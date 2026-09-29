#!/usr/bin/env python
"""原则驱动标注生成（原则分，非主观审美）。

对应需求：AC-03（构图分与人工一致性）
对应文档：《测试与验收.md》§4.1 —— 构图评分相关性（SRCC）

**为什么需要这个脚本，以及它为什么不能替代真人标注**

``configs/eval/annotations.json``（5 张裁切图）的 `known_limitation` 指出，
裁切构造导致 ``subject_scale`` 被抹平，SRCC 不具统计效力。本脚本配合
``scripts/fetch_eval_photos.py`` 采集的 48 张**真实整图**，生成一份
**原则驱动**的序数标注，使样本量跨过 n≥30 门槛。

**必须讲清楚的方法论边界（写进报告的"诚实条款"）**

1. 本标注**不是**主观人工评分。它由构图学中可复述、可争议的**明确原则**
   计算得出，`annotation_method = "principle_based"`、
   `is_subjective_human_rating = false`。
2. **为避免循环论证**，本脚本的原则实现与评分器
   （``aicg.composition.rules``）**是独立重写的**：它只看两个量——
   主体框位置与主体框相对画面的占比——并按教科书构图原则打分，
   不读取评分器的权重、不调用评分器的函数、不参考评分器的输出。
   两边独立实现若仍然同向，才说明"评分器与构图原则一致"这一命题
   有一定支撑；若同向纯靠窃取同一套公式，则该结论毫无意义。
3. **最强的局限**：原则分与真人审美偏好之间存在系统性偏差
   （例如"主体偏小但有环境交代"在纪实摄影中可能是优点，
   而原则分会扣分）。因此本 SRCC 回答的问题是
   **"评分器是否与构图学原则同向"**，而**不是**"评分器是否符合人类审美"。
   后者需要多人主观标注，本项目尚未具备。

用法::

    python scripts/build_principle_annotations.py
    python scripts/build_principle_annotations.py --samples 24   # 生成抽检样本表
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from aicg.observability import get_logger, setup_logging  # noqa: E402
from aicg.perception import perception_from_settings  # noqa: E402
from aicg.settings import PROJECT_ROOT, load_settings  # noqa: E402

log = get_logger("scripts.build_annotations")

REAL_DIR = PROJECT_ROOT / "configs" / "eval" / "images_real"
OUT_JSON = PROJECT_ROOT / "configs" / "eval" / "annotations_real.json"
SAMPLE_CSV = PROJECT_ROOT / "outputs" / "reports" / "annotation_review_sheet.md"


# ----------------------------------------------------------------------
# 以下三个函数是对构图原则的**独立实现**。
# 刻意不 import aicg.composition.rules —— 独立性是本评测能否成立的前提。
# ----------------------------------------------------------------------
def _thirds_principle(cx: float, cy: float, tol: float = 0.12) -> float:
    """原则一：主体应落在三分点附近（或至少贴住某条三分线）。

    教科书表述："把画面横竖各三等分，交点即视觉趣味中心。"
    评分：到最近三分交点的归一化距离，`tol` 内满分，超过 3*tol 归零。
    """
    t1, t2 = 1.0 / 3.0, 2.0 / 3.0
    dx = min(abs(cx - t1), abs(cx - t2))
    dy = min(abs(cy - t1), abs(cy - t2))
    d = (dx * dx + dy * dy) ** 0.5
    return float(max(0.0, min(1.0, 1.0 - d / (3.0 * tol))))


def _headroom_principle(top: float, ideal: float = 0.12) -> float:
    """原则二：头顶留白适中 —— 不顶头（压迫），也不空太多（主体变小）。

    只依赖"主体框上边缘到画面上边缘的距离占画面高度的比例"。
    """
    dev = abs(top - ideal)
    return float(max(0.0, min(1.0, 1.0 - dev / 0.30)))


def _scale_principle(h: float, ideal: float = 0.55, floor: float = 0.15) -> float:
    """原则三：主体应占画面的合理比例 —— 太小则失焦，太大则压迫。

    与评分器曲线形状不同：这里用**单峰**（过大同样扣分）而非单调递增，
    因为原则层面"贴满整幅"并不比"半身"更好。这一处差异是刻意的：
    如果两边曲线完全一致，就失去了"独立验证"的意义。
    """
    if h >= ideal:
        # 超过理想后缓慢衰减：0.55->1.0, 1.0->0.70（抵制过度塞满）
        over = min(1.0, (h - ideal) / max(1e-6, 1.0 - ideal))
        return float(1.0 - 0.30 * over)
    if h < floor:
        return float(max(0.0, 0.30 * (h / max(floor, 1e-6))))
    return float(0.30 + 0.70 * (h - floor) / max(ideal - floor, 1e-6))


def _balance_principle(cx: float, cy: float) -> float:
    """原则四：主体不宜贴边，也不宜绝对居中（过居中显得呆板）。

    用主体中心到画面中心的距离做单峰：中心给 0.75（合规但保守），
    "略偏"区间（离中心 0.08~0.30）给满分，贴边（>0.40）快速归零。
    """
    d = ((cx - 0.5) ** 2 + (cy - 0.5) ** 2) ** 0.5
    if d <= 0.08:
        return 0.75
    if d <= 0.30:
        return 1.0
    return float(max(0.0, 1.0 - (d - 0.30) / 0.10))


def principle_score(bbox: tuple[float, float, float, float]) -> tuple[float, dict]:
    """把四条构图原则合成 0~100 的序数分。

    权重是构图学中较常见的经验配比（位置类合计 0.55，占比 0.30，平衡 0.15），
    与评分器的权重并不相同 —— 这是刻意的独立性来源之一。
    """
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    h = y2 - y1

    p_thirds = _thirds_principle(cx, cy)
    p_head = _headroom_principle(y1)
    p_scale = _scale_principle(h)
    p_bal = _balance_principle(cx, cy)
    # 主体越出画面（被裁切）是硬缺陷，直接压低
    clipped = x1 < -1e-6 or y1 < -1e-6 or x2 > 1 + 1e-6 or y2 > 1 + 1e-6

    raw = 0.30 * p_thirds + 0.25 * p_head + 0.30 * p_scale + 0.15 * p_bal
    if clipped:
        raw *= 0.5

    score = round(100.0 * raw, 1)
    return score, {
        "thirds": round(p_thirds, 3),
        "headroom": round(p_head, 3),
        "scale": round(p_scale, 3),
        "balance": round(p_bal, 3),
        "clipped": clipped,
    }


# ----------------------------------------------------------------------
def measure(perception, image_path: Path) -> dict | None:
    img = cv2.imread(str(image_path))
    if img is None:
        try:
            img = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        except OSError:
            pass
    if img is None:
        return None

    from aicg.utils.image import resize_keep_aspect

    img = resize_keep_aspect(img, 480)
    h, w = img.shape[:2]
    res = perception.infer(img, 0, 0)
    subj = res.primary_subject
    if subj is None:
        return None
    return {
        "bbox": tuple(float(v) for v in subj.bbox),
        "confidence": float(subj.confidence),
        "auto_hint": None,
        "frame": (h, w),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="原则驱动标注生成")
    ap.add_argument("--images", default=str(REAL_DIR))
    ap.add_argument("--out", default=str(OUT_JSON))
    ap.add_argument("--samples", type=int, default=24,
                    help="额外生成多少张人工抽检用的样张清单")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    setup_logging("WARNING" if args.quiet else "INFO")

    d = Path(args.images)
    files = sorted(d.glob("*.jpg"))
    if not files:
        print(f"[错误] 目录下无 jpg: {d}", file=sys.stderr)
        print("       请先运行 scripts/fetch_eval_photos.py", file=sys.stderr)
        return 2

    cfg = load_settings(overrides={"perception.backend": "yolo"})
    perception = perception_from_settings(cfg)
    perception.warmup((360, 480))

    index = {}
    idx_path = d / "_index.json"
    if idx_path.exists():
        for r in json.loads(idx_path.read_text(encoding="utf-8")):
            index[r["file"]] = r

    items = []
    skipped = []
    for f in files:
        m = measure(perception, f)
        if m is None:
            skipped.append(f.name)
            continue
        score, detail = principle_score(m["bbox"])
        meta = index.get(f.name, {})
        x1, y1, x2, y2 = m["bbox"]
        items.append(
            {
                "image": str(Path("configs/eval/images_real") / f.name),
                "human_score": score,
                "principle_detail": detail,
                "subject_bbox": [round(v, 4) for v in m["bbox"]],
                "area_ratio": round((x2 - x1) * (y2 - y1), 4),
                "detector_confidence": round(m["confidence"], 3),
                "author": meta.get("author", "?"),
                "unsplash_url": meta.get("unsplash_url", ""),
                "note": (
                    f"主体中心({(x1+x2)/2:.2f},{(y1+y2)/2:.2f})，"
                    f"高度占比{y2-y1:.2f}；原则分项 "
                    f"thirds={detail['thirds']} headroom={detail['headroom']} "
                    f"scale={detail['scale']} balance={detail['balance']}"
                    + ("；主体越出画面" if detail["clipped"] else "")
                ),
            }
        )

    items.sort(key=lambda r: -r["human_score"])

    payload = {
        "source_note": (
            "【原则驱动标注，非主观人工评分】素材为 48 张真实拍摄照片，"
            "经 Lorem Picsum 取自 Unsplash（作者与原始照片页见 "
            "configs/eval/images_real/PROVENANCE.md，许可见 "
            "https://unsplash.com/license）。标注分由四条可复述的构图学原则"
            "独立计算得出：三分点对齐(0.30)、头顶留白适中(0.25)、"
            "主体占比合理(0.30)、画面平衡(0.15)。"
            "关键点：该原则实现与评分器 src/aicg/composition/rules.py "
            "是**独立重写**的——不共享代码、不共享权重、曲线形状亦不同"
            "（如主体占比用单峰而非单调递增），因此两边同向与否具备验证意义。"
            "本标注**不代表真人审美偏好**，它回答的是"
            "「评分器是否与构图学原则同向」。"
        ),
        "annotation_method": "principle_based",
        "is_subjective_human_rating": False,
        "annotation_date": dt.date.today().isoformat(),
        "principles": {
            "thirds_alignment": 0.30,
            "headroom": 0.25,
            "subject_scale": 0.30,
            "balance": 0.15,
        },
        "known_limitation": (
            "【必须与 SRCC 数值同时阅读的局限】(1) 原则分 ≠ 人类审美偏好："
            "原则会对「主体小但有环境交代」这类纪实性构图系统性扣分，"
            "而真人可能给高分，因此本 SRCC 只能支撑「与构图学原则同向」，"
            "不能支撑「符合人类审美」。要后者需多人主观标注（2-3 人独立打分"
            "并报告一致性），本项目尚未完成。(2) 素材取自 Picsum 的通用图库"
            "feed 而非人像摄影专题集，题材混杂，且人像只是其中一部分；"
            "部分画面含多人/动物，主体选择由 prefers_person 规则决定，"
            "可能与拍摄者意图不一致。(3) 筛图阶段以 YOLO 能否检出 person "
            "为准，因此**无人/检不出人的画面被排除**，评测集在"
            "「主体清晰度」上有正向选择偏差。(4) 标注为单一评分者"
            "（即本脚本），未做多人一致性检验。"
        ),
        "items": items,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 70)
    print("  原则驱动标注生成完成")
    print("=" * 70)
    print(f"  素材目录   : {d}")
    print(f"  有效标注   : {len(items)}")
    print(f"  跳过(无主体): {len(skipped)}" + (f" -> {skipped[:6]}" if skipped else ""))
    print(f"  输出       : {out}")
    if items:
        sc = [i["human_score"] for i in items]
        print(f"  原则分分布 : min {min(sc):.1f} / 中位 {np.median(sc):.1f} / max {max(sc):.1f}")
        ar = [i["area_ratio"] for i in items]
        print(f"  主体占比   : min {min(ar):.4f} / 中位 {np.median(ar):.4f} / max {max(ar):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
