#!/usr/bin/env python
"""按**主题检索**采集「女生拍照」素材（Bing 图片搜索 → 免费图库直链）。

对应需求：FR-11（演示素材）/ 用户需求「女生拍照」主题素材库
对应文档：`docs/research/素材图源调研.md`

为什么需要这个脚本
------------------

上一轮实测确认：Pexels / Unsplash 的 **API 需 Key（401）**、网页被
Cloudflare 挡（403），因此当时只能退到 Picsum（993 张随机图库、无主题
标签、人像占比仅 3.4%），**无法按主题检索**。

本轮实测发现两条可行路径：

1. **Bing 图片搜索的 async 接口可访问（200）**，返回结构化 JSON，
   每条含 ``murl``（原图直链）/ ``purl``（来源页）/ ``desc``（描述）
   → **这提供了真正的主题检索能力**。
2. **免费图库的 CDN 直链可访问（200）**，例如
   ``cdn.pixabay.com/photo/...``，虽然网页被 Cloudflare 挡，但图片本身可下载。

把两者结合：用 Bing 按主题搜索 → **过滤出白名单免费图库的直链** →
下载 → 本项目链路复筛。这样既有主题能力，又**不碰付费图库的版权风险**。

合规红线（**硬编码在白名单里，不可绕过**）
----------------------------------------

**只接受白名单内的免费图库域名**。实测 Bing 的原始结果中，
``img.freepik.com`` / ``c8.alamy.com`` / ``thumbs.dreamstime.com`` /
``media.gettyimages.com`` 等**付费图库占比极高（实测首轮 35 条中 28 条）**，
直接抓取有明确版权风险。因此：

- 白名单只含：Pixabay / Pexels / Unsplash / Wikimedia / Flickr /
  PublicDomainPictures / StockSnap / Rawpixel 等**免费或公共领域**源；
- 每条结果**记录 ``purl`` 与 ``desc``**，用于人工复核与溯源；
- 下载后仍要过**本项目感知链路复筛**（见 ``--screen``）。

用法::

    python scripts/fetch_topic_photos.py --probe          # 只探测，不下载
    python scripts/fetch_topic_photos.py --target 30      # 采集 30 张
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from aicg.observability import get_logger, setup_logging  # noqa: E402
from aicg.settings import PROJECT_ROOT  # noqa: E402

log = get_logger("scripts.fetch_topic_photos")

UA = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# ----------------------------------------------------------------------
# 合规白名单：**只接受免版税 / 公共领域图库**
#
# 实测教训：不带此白名单时，Bing 首轮 35 条结果中 28 条来自付费图库
# （Alamy 13 / Freepik 12 / Dreamstime 2 / Getty 1），**不可直接使用**。
# 因此白名单是**必需的安全阀**，不是可选项。
# ----------------------------------------------------------------------
FREE_HOSTS: dict[str, str] = {
    "cdn.pixabay.com": "Pixabay Content License（免费商用、无需署名）",
    "images.pexels.com": "Pexels License（免费商用、无需署名）",
    "images.unsplash.com": "Unsplash License（免费商用、无需署名）",
    "upload.wikimedia.org": "Wikimedia Commons（多为 CC/PD，授权逐图不同）",
    "live.staticflickr.com": "Flickr（授权逐图不同，需查 purl）",
    "www.publicdomainpictures.net": "Public Domain Pictures（公共领域）",
    "cdn.stocksnap.io": "StockSnap（CC0）",
    "images.rawpixel.com": "Rawpixel（部分公共领域，需查 purl）",
    "burst.shopifycdn.com": "Burst by Shopify（免费商用）",
    "picsum.photos": "Unsplash 经 Picsum 分发",
}

# 主题化检索词：严格对齐用户的 6 个场景要求
TOPIC_QUERIES: dict[str, list[str]] = {
    "selfie": [
        "free stock photo woman selfie portrait",
        "free photo girl selfie smiling natural light",
    ],
    "cafe": [
        "free stock photo woman cafe coffee portrait",
        "free photo woman sitting coffee shop window light",
    ],
    "street": [
        "free photo woman street style portrait walking",
        "free stock photo girl city street candid",
    ],
    "travel": [
        "free photo woman travel portrait scenic",
        "free stock photo girl traveling mountain landscape portrait",
    ],
    "lifestyle": [
        "free photo woman lifestyle candid daily life",
        "free stock photo girl reading home natural light portrait",
    ],
    "outdoor": [
        "free photo woman outdoor golden hour portrait",
        "free stock photo girl park sunset portrait",
    ],
}

BING = "https://www.bing.com/images/async?q={q}&first={first}&count={n}&mmasync=1"


@dataclass
class Hit:
    murl: str
    purl: str = ""
    desc: str = ""
    topic: str = ""
    host: str = ""
    license_note: str = ""
    # 下载后
    img: np.ndarray | None = None
    w: int = 0
    h: int = 0
    rejects: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.rejects


def get_bytes(url: str, timeout: int = 30) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def bing_search(query: str, first: int = 1, n: int = 35) -> list[dict]:
    """调用 Bing 图片搜索 async 接口，解析出结构化结果。

    Bing 把每条结果放在 ``m="{...json...}"`` 属性里，键含
    ``murl``（原图）/ ``purl``（来源页）/ ``desc``（描述）/ ``t``（标题）。
    """
    url = BING.format(q=urllib.parse.quote(query), first=first, n=n)
    try:
        body = get_bytes(url, timeout=25).decode("utf-8", "replace")
    except Exception as e:  # noqa: BLE001
        log.warning(f"搜索失败 [{query[:40]}]: {type(e).__name__}")
        return []

    out: list[dict] = []
    for raw in re.findall(r'm="(\{[^"]*?\})"', body):
        try:
            d = json.loads(html.unescape(raw))
        except json.JSONDecodeError:
            continue
        if d.get("murl"):
            out.append(d)
    return out


def collect(topic_filter: set[str] | None, per_topic_pages: int) -> list[Hit]:
    """按主题检索并**只保留白名单图库**的结果。"""
    found: dict[str, Hit] = {}
    stats: Counter[str] = Counter()

    for topic, queries in TOPIC_QUERIES.items():
        if topic_filter and topic not in topic_filter:
            continue
        for q in queries:
            for first in range(1, 1 + per_topic_pages * 35, 35):
                results = bing_search(q, first=first)
                if not results:
                    continue
                for d in results:
                    host = urlparse(d["murl"]).netloc.lower()
                    stats[host] += 1
                    if host not in FREE_HOSTS:
                        continue          # 合规闸门：非白名单一律丢弃
                    if d["murl"] in found:
                        continue
                    found[d["murl"]] = Hit(
                        murl=d["murl"],
                        purl=d.get("purl", ""),
                        desc=d.get("desc", "") or d.get("t", ""),
                        topic=topic,
                        host=host,
                        license_note=FREE_HOSTS[host],
                    )
                print(f"    [{topic}] '{q[:44]}' first={first} "
                      f"→ 本轮 {len(results)} 条，累计白名单命中 {len(found)}")

    print()
    print("  原始结果域名分布（**说明为何必须白名单**）:")
    for h, c in stats.most_common(12):
        mark = "  ✅白名单" if h in FREE_HOSTS else "  ❌付费/未知"
        print(f"    {h:36s} {c:4d}{mark}")
    return list(found.values())


def download(h: Hit, min_long_edge: int) -> Hit:
    try:
        raw = get_bytes(h.murl, timeout=40)
    except Exception as e:  # noqa: BLE001
        h.rejects.append(f"下载失败:{type(e).__name__}")
        return h
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        h.rejects.append("解码失败")
        return h
    h.img = img
    h.h, h.w = img.shape[:2]
    if max(h.w, h.h) < min_long_edge:
        h.rejects.append(f"尺寸不足({h.w}x{h.h})")
    return h


def main() -> int:
    ap = argparse.ArgumentParser(description="按主题检索采集「女生拍照」素材")
    ap.add_argument("--target", type=int, default=30)
    ap.add_argument("--topics", default="", help="逗号分隔，默认全部")
    ap.add_argument("--pages", type=int, default=2, help="每主题翻页数")
    ap.add_argument("--min-long-edge", type=int, default=900)
    ap.add_argument("--probe", action="store_true", help="只探测不下载")
    ap.add_argument("--out", default=str(PROJECT_ROOT / "assets" / "topic_photos"))
    args = ap.parse_args()

    setup_logging("INFO")
    out_dir = Path(args.out)
    topic_filter = {t.strip() for t in args.topics.split(",") if t.strip()} or None

    print("=" * 76)
    print("  按主题检索采集素材（Bing 搜索 → 白名单免费图库直链）")
    print("=" * 76)
    print(f"  主题     : {sorted(topic_filter) if topic_filter else sorted(TOPIC_QUERIES)}")
    print(f"  目标     : {args.target} 张")
    print(f"  白名单   : {len(FREE_HOSTS)} 个免费图库域（付费图库一律丢弃）")
    print()

    hits = collect(topic_filter, args.pages)
    print()
    print(f"  白名单命中合计: {len(hits)} 条唯一直链")
    by_topic = Counter(h.topic for h in hits)
    for t, c in by_topic.most_common():
        print(f"    {t:12s} {c:4d}")
    print()
    print("  来源分布（合规）:")
    for h, c in Counter(x.host for x in hits).most_common():
        print(f"    {h:34s} {c:4d}  {FREE_HOSTS[h][:38]}")
    print()

    if args.probe or not hits:
        if not hits:
            print("  [警告] 白名单命中为 0，无法采集。")
        return 0

    # 并发下载（注意：Pixabay CDN 不限速，与 tpdne 不同）
    print(f"  下载中（并发 8）…")
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(download, h, args.min_long_edge) for h in hits]
        done = 0
        for f in as_completed(futs):
            f.result()
            done += 1
            if done % 20 == 0:
                print(f"    {done}/{len(hits)}")

    good = [h for h in hits if h.ok and h.img is not None]
    print(f"  下载成功且尺寸达标: {len(good)}/{len(hits)}")

    # 去重（像素指纹，防止同图不同 URL）
    import hashlib
    seen: set[str] = set()
    uniq: list[Hit] = []
    for h in good:
        fp = hashlib.md5(cv2.resize(h.img, (32, 32)).tobytes()).hexdigest()[:16]
        if fp in seen:
            continue
        seen.add(fp)
        uniq.append(h)
    print(f"  像素级去重后: {len(uniq)}")

    # 按主题配额选取
    picked: list[Hit] = []
    quota = max(1, args.target // max(1, len(by_topic)))
    per: Counter[str] = Counter()
    for h in uniq:
        if len(picked) >= args.target:
            break
        if per[h.topic] >= quota:
            continue
        per[h.topic] += 1
        picked.append(h)
    # 未选够则放宽配额补齐
    if len(picked) < args.target:
        for h in uniq:
            if len(picked) >= args.target:
                break
            if h not in picked:
                picked.append(h)

    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for i, h in enumerate(sorted(picked, key=lambda x: (x.topic, x.murl)), 1):
        fn = f"{i:02d}_{h.topic}_{h.w}x{h.h}_{h.host.split('.')[0]}.jpg"
        ok, buf = cv2.imencode(".jpg", h.img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        if not ok:
            continue
        with open(out_dir / fn, "wb") as f:
            f.write(buf.tobytes())
        gray = cv2.cvtColor(h.img, cv2.COLOR_BGR2GRAY)
        records.append({
            "file": fn,
            "topic_requested": h.topic,
            "source_url": h.murl,
            "source_page": h.purl,
            "source_desc": h.desc,
            "source_host": h.host,
            "license": h.license_note,
            "width": h.w, "height": h.h,
            "sharpness_laplacian_var": round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 1),
            "mean_brightness": round(float(gray.mean()) / 255.0, 3),
            "note": "topic 来自检索关键词，**非图库官方标签**；仍需人工复核",
        })

    (out_dir / "_index.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print()
    print(f"  入库 {len(records)} 张 -> {out_dir}")
    for t, c in Counter(r["topic_requested"] for r in records).most_common():
        print(f"    {t:12s} {c:3d}")
    print(f"  索引 -> {out_dir / '_index.json'}")
    print("=" * 76)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
