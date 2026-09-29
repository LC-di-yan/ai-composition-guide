"""量化验证 D-10 修复效果：真实素材的主体占高分布 vs 「距离合适」窗口。

**为什么需要这个脚本**：D-10 不是靠"读代码"发现的，而是靠真实素材跑出来的。
同样，修复是否真的解决了问题，也不能靠"代码看起来对了"来判断——必须重新
测一遍真实分布，确认：

1. 修复前的窗口（0.60~0.76）与素材分布的**重叠率**是多少（预期极低）；
2. 修复后的窗口（0.55±15% → 0.4675~0.6325）重叠率是多少；
3. 动作分布是否从"几乎全是 move_closer"变成有意义的分布。

用法::

    PYTHONPATH=src python scripts/check_occupancy_window.py

输出 ``outputs/reports/d10_occupancy_window.json``。
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402

from aicg.perception.factory import perception_from_settings  # noqa: E402
from aicg.settings import load_settings  # noqa: E402
from aicg.utils.image import bbox_height  # noqa: E402

ASSETS = PROJECT_ROOT / "assets" / "topic_photos"
OUT = PROJECT_ROOT / "outputs" / "reports" / "d10_occupancy_window.json"


def imread_any(path: Path) -> np.ndarray | None:
    """读图，兼容非 ASCII 路径。

    ``cv2.imread`` 在中文路径下**静默返回 None**（不抛异常），必须走
    ``np.fromfile`` + ``imdecode``；且要捕获 ``Exception`` 而非仅 ``OSError``。
    """
    import cv2

    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
        if buf.size == 0:
            return None
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except Exception:  # noqa: BLE001
        return None


def main() -> int:
    settings = load_settings()
    det = perception_from_settings(settings)

    backend = getattr(det, "backend_name", None) or type(det).__name__
    if backend == "rule" or type(det).__name__ == "RulePerception":
        print("[错误] 感知后端降级为 rule，占高分布无意义，拒绝继续。")
        print("       请确认 models/yolov8n-seg.pt 存在。")
        return 1
    print(f"[信息] 感知后端 = {backend}")

    sc = settings.composition.scoring
    ideal = sc.ideal_subject_height
    tol = sc.derived_occupancy_tolerance
    new_lo, new_hi = ideal - tol, ideal + tol
    old_lo, old_hi = 0.60, 0.76  # 修复前：0.68 ± 0.08

    ratios: list[dict] = []
    files = sorted(p for p in ASSETS.glob("*.jpg"))
    if not files:
        files = sorted(p for p in ASSETS.glob("*.png"))
    print(f"[信息] 待测素材 {len(files)} 张")

    for p in files:
        img = imread_any(p)
        if img is None:
            print(f"  [跳过] 无法读取 {p.name}")
            continue
        res = det.infer(img, frame_id=len(ratios), timestamp_ms=len(ratios) * 33)
        if getattr(res, "degraded", False):
            print(f"  [跳过] 感知降级 {p.name}")
            continue
        subs = getattr(res, "subjects", None) or []
        if not subs:
            print(f"  [跳过] 未检出主体 {p.name}")
            continue
        # 取置信度最高的主体
        best = max(subs, key=lambda s: getattr(s, "confidence", 0.0))
        h = bbox_height(best.bbox)
        ratios.append({"file": p.name, "height_ratio": round(h, 4)})
        print(f"  {p.name:42s} 占高 {h:.3f}")

    if not ratios:
        print("[错误] 没有任何可用的占高样本。")
        return 1

    vals = np.array([r["height_ratio"] for r in ratios], dtype=float)

    def overlap(lo: float, hi: float) -> float:
        return float(((vals >= lo) & (vals <= hi)).mean())

    # 动作分布：用修复后的阈值判定（尺度优先）
    actions = Counter()
    for v in vals:
        d_scale = v / ideal
        if abs(1.0 - d_scale) > sc.occupancy_tolerance_ratio:
            actions["move_closer" if d_scale < 1.0 else "move_back"] += 1
        else:
            actions["in_window"] += 1

    report = {
        "n": len(vals),
        "backend": backend,
        "ideal_subject_height": ideal,
        "occupancy_tolerance_ratio": sc.occupancy_tolerance_ratio,
        "window_before_fix": {"lo": old_lo, "hi": old_hi,
                              "overlap": round(overlap(old_lo, old_hi), 4)},
        "window_after_fix": {"lo": round(new_lo, 4), "hi": round(new_hi, 4),
                             "overlap": round(overlap(new_lo, new_hi), 4)},
        "distribution": {
            "min": round(float(vals.min()), 4),
            "p25": round(float(np.percentile(vals, 25)), 4),
            "median": round(float(np.median(vals)), 4),
            "p75": round(float(np.percentile(vals, 75)), 4),
            "max": round(float(vals.max()), 4),
            "mean": round(float(vals.mean()), 4),
        },
        "action_distribution_after_fix": dict(actions),
        "samples": ratios,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 62)
    print(f"样本数 n = {len(vals)}")
    print(f"占高分布   min={vals.min():.3f}  p25={np.percentile(vals,25):.3f}  "
          f"median={np.median(vals):.3f}  p75={np.percentile(vals,75):.3f}  "
          f"max={vals.max():.3f}")
    print(f"修复前窗口 [{old_lo:.3f}, {old_hi:.3f}]  重叠率 = {overlap(old_lo,old_hi)*100:.1f}%")
    print(f"修复后窗口 [{new_lo:.4f}, {new_hi:.4f}] 重叠率 = {overlap(new_lo,new_hi)*100:.1f}%")
    print(f"动作分布   {dict(actions)}")
    print("=" * 62)
    print(f"[完成] 已写出 {OUT.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
