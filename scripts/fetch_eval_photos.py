#!/usr/bin/env python
"""真实人像评测集素材采集（突破 n=5 局限）。

对应需求：AC-03（构图分与人工一致性）、NFR-O3（指标可复跑）
对应文档：《测试与验收.md》§4.1 —— 构图评分相关性（SRCC）

**这个脚本为什么存在**

``configs/eval/annotations.json`` 里的 5 张图是从 ``zidane.jpg`` 裁切构造的，
其 ``known_limitation`` 已如实记录了一个致命的构造缺陷：裁切出来的
"主体过小 / 主体贴边" 图，YOLO 仍会把可见人体扩张到接近整幅画面，
导致 ``subject_scale`` 对全部样本都判 1.00 —— 想验证的维度在感知层就被抹平了。

要得到有统计效力的 SRCC（n ≥ 30），必须换用**真实拍摄的整图**：
主体占比随取景距离自然变化，而不是靠裁切伪造。

**素材来源与授权（必须诚实记录）**

本脚本从 `Lorem Picsum <https://picsum.photos>`_ 获取素材。Picsum 的图源是
**Unsplash**，且其 ``/v2/list`` 接口会返回每张图的**作者名**与
**Unsplash 原始照片页 URL**，这正是署名所需的全部信息。

Unsplash License 要点（https://unsplash.com/license）：
- 免费用于商业与非商业用途，**无需**获得许可；
- **无需**署名，但**强烈建议**署名（本脚本一律署名，不占这个便宜）；
- 不得将照片原样转售，或用于构建竞品图库服务；
- 照片中可识别人物的肖像权未被涵盖 —— 用于**内部算法评测**属合理使用，
  但若对外展示需另行评估。本脚本采集的图**仅用于本地评测，不对外再分发**。

因此本脚本会为每张采集到的图写一条 provenance 记录（作者、原始页、
采集日期、许可），并落到 ``configs/eval/images_real/PROVENANCE.md``。

**两阶段采集（避免无谓带宽）**

800 张候选图若全部按全分辨率下载约 1GB+。因此：

1. **Stage 1 筛图**：按少量宽（如 400px）拉取，跑 YOLO，只保留
   能检出 ``person`` 的图，并记录其归一化主体框；
2. **Stage 2 取图**：仅对通过筛选的图按目标长边重新拉取。

筛选结果（含被淘汰图及原因）会完整写入 ``screen_report.json``，
不隐藏失败样本。

用法::

    python scripts/fetch_eval_photos.py --screen-only       # 只筛，不落全图
    python scripts/fetch_eval_photos.py --limit 120         # 默认扫前 120 张候选
    python scripts/fetch_eval_photos.py --target 40         # 目标保留 40 张全图
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from aicg.observability import get_logger, setup_logging  # noqa: E402
from aicg.perception import perception_from_settings  # noqa: E402
from aicg.settings import PROJECT_ROOT, load_settings  # noqa: E402

log = get_logger("scripts.fetch_eval")

REAL_DIR = PROJECT_ROOT / "configs" / "eval" / "images_real"
CACHE_DIR = PROJECT_ROOT / "outputs" / "_fetch_cache"
PROVENANCE = REAL_DIR / "PROVENANCE.md"
SCREEN_REPORT = PROJECT_ROOT / "outputs" / "reports" / "screen_report.json"
CATALOG = PROJECT_ROOT / "outputs" / "_probe" / "catalog800.json"

PICSUM = "https://picsum.photos"
UNSPLASH_LICENSE = "https://unsplash.com/license"
UA = "aicg-eval-fetcher/1.0 (local research; contact: project owner)"


# ----------------------------------------------------------------------
def _get(url: str, timeout: int = 30) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def download_thumb(photo: dict, width: int) -> tuple[str, np.ndarray | None]:
    """拉一张缩略图并解码。失败返回 (id, None)，不抛。"""
    pid = photo["id"]
    url = f"{PICSUM}/id/{pid}/{width}/{int(width * 3 / 4)}"
    try:
        raw = _get(url)
    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
        log.debug(f"缩略图失败 id={pid}: {e}")
        return pid, None
    buf = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return pid, img


def download_full(photo: dict, long_edge: int) -> tuple[str, bytes | None]:
    """按目标长边拉全图（保持原始宽高比，由 Picsum 裁剪）。"""
    pid = photo["id"]
    w, h = int(photo["width"]), int(photo["height"])
    if w >= h:
        tw, th = long_edge, max(1, round(long_edge * h / w))
    else:
        th, tw = long_edge, max(1, round(long_edge * w / h))
    url = f"{PICSUM}/id/{pid}/{tw}/{th}"
    try:
        return pid, _get(url, timeout=60)
    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
        log.warning(f"全图失败 id={pid}: {e}")
        return pid, None


def ndarray_to_jpeg_bytes(img: np.ndarray, quality: int = 92) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("cv2.imencode 失败")
    return buf.tobytes()


def check_write(path: Path, data: bytes) -> None:
    """用二进制写，避开 Windows 下非 ASCII 路径的编码坑。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)


# ----------------------------------------------------------------------
def screen(photos: list[dict], width: int, workers: int) -> list[dict]:
    """Stage 1：并发拉缩略图 → YOLO 筛选 → 返回通过者（含主体框）。"""
    from aicg.perception import perception_from_settings as _pfs

    cfg = load_settings(overrides={"perception.backend": "yolo"})
    perception = _pfs(cfg)
    perception.warmup((360, 480))

    results: list[dict] = []
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(download_thumb, p, width): p for p in photos}
        for i, fut in enumerate(cf.as_completed(futs), 1):
            photo = futs[fut]
            pid, img = fut.result()
            if img is None:
                results.append({"id": pid, "kept": False, "reason": "download_failed"})
                continue
            if img.shape[0] < 64 or img.shape[1] < 64:
                results.append({"id": pid, "kept": False, "reason": "too_small"})
                continue

            h, w = img.shape[:2]
            res = perception.infer(img, 0, 0)
            subj = res.primary_subject
            if subj is None:
                results.append({"id": pid, "kept": False, "reason": "no_person"})
                continue

            x1, y1, x2, y2 = subj.bbox
            results.append(
                {
                    "id": pid,
                    "kept": True,
                    "reason": "person_detected",
                    "score": round(float(subj.confidence), 3),
                    "bbox": [round(float(v), 4) for v in (x1, y1, x2, y2)],
                    "area_ratio": round(float((x2 - x1) * (y2 - y1)), 4),
                    "aspect": round(float(w / h), 3),
                }
            )
            if i % 20 == 0:
                kept = sum(1 for r in results if r["kept"])
                log.info(f"筛图进度 {i}/{len(photos)}，已保留 {kept}")

    return results


def pick_diverse(kept: list[dict], target: int) -> list[dict]:
    """从通过筛图的图里挑出**主体占比有真实分布**的子集。

    这是本脚本的方法论核心。只取"检出人"是不够的——若全是大头照，
    ``subject_scale`` 又会退化成常数。因此按归一化主体框面积分桶，
    每桶按配额取样，使主体占比覆盖 极小 / 小 / 中 / 中大 / 大 五个区间。

    配额按各桶的**可得数量**成比例分配（先保底各 4 张，剩余按比例），
    这样既保证最稀有的中间桶不被饿死，又不伪造不存在的分布。
    """
    if len(kept) <= target:
        return kept

    EDGES = (0.02, 0.06, 0.15, 0.35)
    names = ("xs", "s", "m", "l", "xl")
    buckets: dict[str, list[dict]] = {n: [] for n in names}
    for r in kept:
        a = r["area_ratio"]
        idx = 0
        for e in EDGES:
            if a >= e:
                idx += 1
        buckets[names[idx]].append(r)

    # 每桶内按"置信度降序 + id"排序，偏好检出更确定的样本
    for v in buckets.values():
        v.sort(key=lambda r: (-r["score"], int(r["id"])))

    MIN_PER = 4
    avail = {n: len(buckets[n]) for n in names}
    taken = {n: min(MIN_PER, avail[n]) for n in names}
    used = sum(taken.values())

    # 剩余配额按可得数量比例分配，余数按桶序补齐
    if used < target:
        pool = {n: avail[n] - taken[n] for n in names}
        total_pool = sum(pool.values())
        if total_pool > 0:
            quota_left = target - used
            for n in names:
                share = int(quota_left * pool[n] / total_pool)
                taken[n] += min(share, pool[n])
            # 余数补齐
            i = 0
            while sum(taken.values()) < target and i < 1000:
                n = names[i % len(names)]
                if taken[n] < avail[n]:
                    taken[n] += 1
                i += 1

    picked: list[dict] = []
    for n in names:
        picked.extend(buckets[n][: taken[n]])
    return picked[:target]


# ----------------------------------------------------------------------
def write_provenance(rows: list[dict], long_edge: int) -> None:
    today = dt.date.today().isoformat()
    lines = [
        "# 评测集素材来源与授权记录（images_real/）",
        "",
        "> 本文件由 `scripts/fetch_eval_photos.py` 自动生成，请勿手改。",
        "> 记录目的：使评测集的素材来源**可追溯、可复核、授权清晰**。",
        "",
        "## 授权说明",
        "",
        "素材经 [Lorem Picsum](https://picsum.photos) 获取，图源为 **Unsplash**。"
        f"适用 [Unsplash License]({UNSPLASH_LICENSE})：",
        "",
        "- 免费用于商业与非商业用途，无需获得许可；",
        "- 无需署名，但强烈建议署名 —— 本项目**一律署名**；",
        "- 不得原样转售照片，或用于构建竞品图库服务；",
        "- 照片中可识别人物的肖像权未被该许可涵盖。本评测集**仅用于本地算法评估，"
        "不再分发**；若需对外展示，须另行评估肖像权。",
        "",
        f"- 采集日期：**{today}**",
        f"- 采集分辨率长边：**{long_edge}px**",
        f"- 素材数量：**{len(rows)}**",
        "",
        "## 逐图记录",
        "",
        "| 文件名 | Picsum ID | 作者 | 原始照片页 |",
        "|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| `{r['file']}` | {r['id']} | {r['author']} | <{r['unsplash_url']}> |"
        )
    lines += [
        "",
        "## 关于「作者」字段的说明",
        "",
        "Picsum 只保证其 `author` 字段可查，未提供逐图的 license 元数据。"
        "本项目按 Unsplash License 的**最保守条款**处理（署名 + 不转售 + 不再分发）。",
        "",
        "## 已知局限",
        "",
        "1. **素材不是评测用的「构图质量标注集」**。本目录只保证「真实拍摄 + "
        "可检出主体 + 主体占比有分布」，人工分由 `annotations_real.json` 另行给出，"
        "其标注方法必须在该文件中如实声明。",
        "2. **题材偏人像**。因当前评分器以人像构图为设计目标，非人像图（风光/静物）"
        "在筛图阶段被主动剔除，这会限制结论的外推范围。",
        "3. **分辨率被统一压到长边 "
        f"{long_edge}px**。构图比例不受影响，但细节纹理与原始素材不同。",
    ]
    PROVENANCE.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ----------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="真实人像评测集素材采集")
    ap.add_argument("--catalog", default=str(CATALOG), help="Picsum 元数据目录 JSON")
    ap.add_argument("--limit", type=int, default=200, help="扫描多少张候选")
    ap.add_argument("--target", type=int, default=40, help="保留多少张全图")
    ap.add_argument("--thumb-width", type=int, default=400)
    ap.add_argument("--long-edge", type=int, default=1024)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--screen-only", action="store_true", help="只筛图，不下载全图")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    setup_logging("WARNING" if args.quiet else "INFO")

    cat = Path(args.catalog)
    if not cat.exists():
        print(f"[错误] 目录文件不存在: {cat}", file=sys.stderr)
        print("       请先运行 catalog 抓取（见本脚本 docstring 的两阶段说明）。",
              file=sys.stderr)
        return 2

    photos = json.loads(cat.read_text(encoding="utf-8"))
    photos = photos[: args.limit]
    log.info(f"候选 {len(photos)} 张，开始 Stage 1 筛图（YOLO）")

    results = screen(photos, args.thumb_width, args.workers)
    kept = [r for r in results if r["kept"]]
    log.info(f"Stage 1 完成：检出主体 {len(kept)}/{len(results)}")

    SCREEN_REPORT.parent.mkdir(parents=True, exist_ok=True)
    SCREEN_REPORT.write_text(
        json.dumps(
            {
                "catalog_size": len(photos),
                "screened": len(results),
                "kept": len(kept),
                "kept_ratio": round(len(kept) / max(1, len(results)), 4),
                "thumb_width": args.thumb_width,
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    log.info(f"筛选明细 → {SCREEN_REPORT}")

    if args.screen_only:
        return 0

    picked = pick_diverse(kept, args.target)
    log.info(f"Stage 2：按主体占比分桶挑选 {len(picked)} 张，开始取全图")

    by_id = {p["id"]: p for p in photos}
    rows: list[dict] = []
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(download_full, by_id[r["id"]], args.long_edge): r for r in picked}
        for fut in cf.as_completed(futs):
            r = futs[fut]
            pid, raw = fut.result()
            if raw is None:
                continue
            photo = by_id[pid]
            fname = f"p{pid}.jpg"
            check_write(REAL_DIR / fname, raw)
            rows.append(
                {
                    "file": fname,
                    "id": pid,
                    "author": photo["author"],
                    "unsplash_url": photo["url"],
                    "native": f"{photo['width']}x{photo['height']}",
                    "subject_bbox": r["bbox"],
                    "area_ratio": r["area_ratio"],
                }
            )

    rows.sort(key=lambda r: int(r["id"]))
    REAL_DIR.mkdir(parents=True, exist_ok=True)
    (REAL_DIR / "_index.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_provenance(rows, args.long_edge)

    print("=" * 70)
    print("  真实评测素材采集完成")
    print("=" * 70)
    print(f"  候选扫描   : {len(photos)}")
    print(f"  检出主体   : {len(kept)}")
    print(f"  实际落地   : {len(rows)}")
    print(f"  目录       : {REAL_DIR}")
    print(f"  授权记录   : {PROVENANCE}")
    print(f"  筛选明细   : {SCREEN_REPORT}")
    if rows:
        ar = [r["area_ratio"] for r in rows]
        print(f"  主体占比   : 最小 {min(ar):.4f} / 中位 {np.median(ar):.4f} / 最大 {max(ar):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
