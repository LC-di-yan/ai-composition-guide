#!/usr/bin/env python
"""把主题素材过一遍本项目感知 + 构图链路，判断能否作为演示素材。

对应需求：FR-11 / NFR-O3

**为什么这一步不能省**

``fetch_topic_photos.py`` 只保证「域名合规」+「文件能解码」，
``curate_topic_photos.py`` 保证「人眼看着合适」。但这两步都不知道
**本项目自己的检测器能不能在这张图上工作**——而演示素材的价值恰恰在于
它能跑通链路。三个必查项：

1. **主体检出**：YOLO 能否找到 person？找不到 → 构图规则全部无法生效；
2. **主体面积**：面积比过小（如远景 <0.05）→ 检出但不稳定，演示会闪；
3. **构图得分**：落在哪个区间 → 决定它适合当「正例」还是「反例」演示。

本脚本只读不改，输出 JSON + 终端分布表。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from aicg.observability import get_logger, setup_logging  # noqa: E402
from aicg.settings import PROJECT_ROOT, load_settings  # noqa: E402

log = get_logger("scripts.screen_topic_photos")


def imread_any(p: Path) -> np.ndarray | None:
    if not p.exists():
        return None
    img = cv2.imread(str(p))
    if img is not None:
        return img
    try:
        buf = np.fromfile(str(p), dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR) if buf.size else None
    except Exception:  # noqa: BLE001
        return None


def main() -> int:
    setup_logging("WARNING")
    settings = load_settings()
    root = PROJECT_ROOT / "assets" / "topic_photos"
    idx = root / "_index_curated.json"
    if not idx.exists():
        print(f"[错误] 找不到 {idx}，请先跑 curate_topic_photos.py")
        return 1
    records = json.loads(idx.read_text(encoding="utf-8"))

    # 延迟导入：感知层会加载 YOLO 权重，导入耗时且重
    #
    # **踩坑记录（本脚本第一版踩过）**：不要用 ``YoloPerception(settings)``
    # 直接构造——它的第一个参数是 ``weights: str``，传整个 ``AppConfig``
    # 会让权重路径变成一坨 repr 字符串，抛 FileNotFoundError 后**静默降级**
    # 到规则后端，返回 subjects=[]。结果就是"23 张全都没检出人"这种
    # 看起来很严重、其实完全是假的结论。
    # 正解是用 ``perception_from_settings``——它负责读配置里的权重/阈值/device，
    # 且保证返回可用对象（NFR-R1）。
    from aicg.perception.factory import perception_from_settings  # noqa: E402

    det = perception_from_settings(settings)
    if getattr(det, "backend_name", None) == "rule" or type(det).__name__ == "RulePerception":
        print("[错误] 感知后端降级为 rule，检测结果不可用于筛查，拒绝继续。")
        print("       请检查 models/yolov8n-seg.pt 是否存在。")
        return 1
    print("=" * 74)
    print(f"  主题素材过本项目感知链路（后端: {type(det).__name__}）")
    print("=" * 74)

    rows, no_subject = [], []
    for i, r in enumerate(records):
        img = imread_any(root / r["file"])
        if img is None:
            print(f"  [跳过] {r['file']} 读取失败")
            continue
        h, w = img.shape[:2]
        res = det.infer(img, frame_id=i, timestamp_ms=i * 33)
        subs = list(getattr(res, "subjects", []) or [])
        # 自身也要检查 degraded——上游降级时返回空结果，不能当"没检出人"
        if getattr(res, "degraded", False):
            print(f"  [降级] {r['file']} 感知返回 degraded=True，结果不可信")
        area = 0.0
        conf = 0.0
        if subs:
            s = max(subs, key=lambda x: float(getattr(x, "area", 0.0)))
            area = float(getattr(s, "area", 0.0))
            conf = float(getattr(s, "confidence", 0.0))
        row = {
            "file": r["file"],
            "topic": r["topic_requested"],
            "n_person": len(subs),
            "subject_area_ratio": round(area, 4),
            "subject_confidence": round(conf, 3),
            "width": w, "height": h,
            "degraded": bool(getattr(res, "degraded", False)),
            "usable": bool(subs and area >= 0.02 and not getattr(res, "degraded", False)),
        }
        rows.append(row)
        if not subs:
            no_subject.append(r["file"])

    (PROJECT_ROOT / "outputs" / "reports" / "topic_photos_screen.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    ok = [r for r in rows if r["usable"]]
    print(f"  复核后素材 : {len(rows)}")
    print(f"  检出主体   : {len(rows) - len(no_subject)}/{len(rows)}"
          f"  ({(len(rows)-len(no_subject))/max(1,len(rows)):.0%})")
    print(f"  可用作演示 : {len(ok)}")
    print()
    print("  逐主题:")
    topics = sorted({r["topic"] for r in rows})
    for t in topics:
        sub = [r for r in rows if r["topic"] == t]
        n_ok = sum(1 for r in sub if r["usable"])
        areas = [r["subject_area_ratio"] for r in sub if r["subject_area_ratio"] > 0]
        a = f"{min(areas):.2f}~{max(areas):.2f}" if areas else "—"
        print(f"    {t:11s} 共{len(sub):3d}  检出{n_ok:3d}  面积比 {a}")
    if no_subject:
        print()
        print("  ⚠ 未检出主体的图（不能作为演示素材）:")
        for f in no_subject:
            print(f"      {f}")
    print()
    print(f"  明细 -> {PROJECT_ROOT / 'outputs' / 'reports' / 'topic_photos_screen.json'}")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
