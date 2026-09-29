#!/usr/bin/env python
"""采集 AI 生成人脸素材（``thispersondoesnotexist``），用于**检测链路**验证。

对应需求：FR-02（感知层）/ NFR-O3（可复现）

用途边界（**必读，勿误用**）
--------------------------

本源的输出**恒为「正面免冠照」构图**：面部居中、正对镜头、背景虚化。
因此它**不能**用于本项目的构图指导演示——构图指导的核心是
主体位置 / 留白比例 / 三分法 / 视线方向 / 景别的差异，而这里全都没有。

它**能**做的事：

1. 验证感知层的人脸 / 人像检测链路在人脸密集输入上的稳定性；
2. 作为「构图恒定」的**对照组**，反证评分器对构图变化的敏感度
   （同一构图、不同人脸 → 分数应几乎不变）；
3. 补齐「人物年龄与穿搭多样化」这一条要求（生成器会产出不同
   年龄 / 性别 / 族裔 / 穿着的面孔）。

**采集约束（实测得出，见 docs/research/素材图源调研.md §4.2）**

该服务对 `random-person.jpeg` 有**缓存**：

   并发 6 次请求 -> 仅 2 张唯一内容（严重重复）
   串行 4 次（间隔 3s） -> 4 张唯一内容

因此本脚本**必须串行 + 加延迟**，并在下载后用**像素级指纹去重**，
把重复的图丢弃重试。这是实测结论，不是保守估计。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from aicg.observability import get_logger, setup_logging  # noqa: E402
from aicg.settings import PROJECT_ROOT  # noqa: E402

log = get_logger("scripts.fetch_face_samples")

URL = "https://thispersondoesnotexist.com/random-person.jpeg"
HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 Chrome/122 Safari/537.36"),
    # 该站校验 Referer，缺失可能被拒
    "Referer": "https://thispersondoesnotexist.com/",
}


def pixel_fingerprint(img: np.ndarray) -> str:
    """像素级指纹：降采样后哈希，**排除 EXIF/压缩噪声干扰**。

    注意：不能用**文件字节**哈希判重——实测同一张图字节级 md5 不同
    （EXIF 差异）但像素内容一致。必须降到像素空间比对。
    """
    small = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)
    return hashlib.md5(small.tobytes()).hexdigest()[:16]


def fetch(timeout: int = 40) -> np.ndarray | None:
    try:
        req = urllib.request.Request(URL, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        return img
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
        log.warning(f"下载失败: {type(e).__name__}")
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description="采集 AI 人脸样本")
    ap.add_argument("--target", type=int, default=20, help="目标张数")
    ap.add_argument("--delay", type=float, default=3.0,
                    help="串行间隔秒数（实测需 >=3s 才不命中缓存）")
    ap.add_argument("--max-attempts", type=int, default=120,
                    help="最大尝试次数（含重复丢弃）")
    ap.add_argument("--out", default=str(PROJECT_ROOT / "assets" / "demo_photos" / "faces"))
    args = ap.parse_args()

    setup_logging("INFO")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 74)
    print("  AI 生成人脸素材采集（检测链路用，非构图演示）")
    print("=" * 74)
    print(f"  目标   : {args.target} 张")
    print(f"  间隔   : {args.delay}s（实测 <3s 会命中服务端缓存导致重复）")
    print(f"  输出   : {out_dir}")
    print()

    seen: set[str] = set()
    records: list[dict] = []
    dup = fail = 0
    t0 = time.perf_counter()

    for attempt in range(1, args.max_attempts + 1):
        if len(records) >= args.target:
            break
        img = fetch()
        time.sleep(args.delay)          # 先睡再判，确保下一次请求间隔足够

        if img is None:
            fail += 1
            continue

        fp = pixel_fingerprint(img)
        if fp in seen:
            dup += 1
            print(f"  [{attempt:3d}] 重复内容，丢弃（累计去重 {dup}）")
            continue
        seen.add(fp)

        idx = len(records) + 1
        # 规范命名：与照片素材一致的风格，但加 faces_ 前缀并标注 ai_generated
        fn = f"faces_{idx:02d}_ai_generated_1024.jpg"
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        if not ok:
            fail += 1
            continue
        with open(out_dir / fn, "wb") as f:
            f.write(buf.tobytes())

        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        records.append({
            "file": fn,
            "width": w,
            "height": h,
            "source": URL,
            "source_kind": "ai_generated",
            "license_note": ("AI 生成人脸，无真实人物肖像权问题；"
                             "但该源未提供明确授权条款，公开使用应注明生成性质"),
            "purpose": ("感知层检测链路验证 / 构图恒定的对照组；"
                        "**不可用于构图多样性演示**（构图恒为正面免冠照）"),
            "pixel_fingerprint": fp,
            "sharpness_laplacian_var": round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 1),
            "mean_brightness": round(float(gray.mean()) / 255.0, 3),
        })
        print(f"  [{attempt:3d}] 收录 {fn}  ({len(records)}/{args.target})")

    (out_dir / "_index.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    el = time.perf_counter() - t0
    print()
    print("-" * 74)
    print(f"  收录     : {len(records)} 张")
    print(f"  重复丢弃 : {dup}")
    print(f"  请求失败 : {fail}")
    print(f"  尝试总数 : {dup + fail + len(records)}")
    print(f"  有效产出率 = {len(records)/max(1, dup+fail+len(records))*100:.0f}%"
          f"（缓存导致的重试成本）")
    print(f"  总耗时   : {el:.0f}s")
    print(f"  索引     -> {out_dir / '_index.json'}")
    if len(records) < args.target:
        print(f"  [注意] 未达目标（{len(records)}/{args.target}）。"
              f"可加大 --max-attempts 或减小 --delay 重试。")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
