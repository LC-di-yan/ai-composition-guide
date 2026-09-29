#!/usr/bin/env python
"""为演示素材生成人工复核拼图（含实测标注）。

对应需求：FR-11 / NFR-O3（可复现）

**为什么需要人工复核**

``fetch_demo_photos.py`` 的属性全部由自动检测得出，自动检测会出错
（YOLO 可能把雕像/海报/远景剪影判为 person；Haar 人脸检测也会有误报）。
本脚本把素材拼成一张总览图，供**人眼过一遍**，确认：

1. 画面里确实有清晰可辨的**人物**（不是雕像 / 海报 / 纯风景）；
2. 画面**不低俗、不敏感**（合规红线，自动检测判不了）；
3. 自动标注的景别与人脸朝向是否与肉眼一致。

标注使用 ASCII（OpenCV Hershey 字体无 CJK 字形），因此写
``SHOT`` / ``FACE`` / ``AREA`` 而非中文。这是既有约束，不是疏漏。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from aicg.observability import get_logger, setup_logging  # noqa: E402
from aicg.settings import PROJECT_ROOT  # noqa: E402

log = get_logger("scripts.make_demo_sheet")


def imread_any(p: Path) -> np.ndarray | None:
    """读图，兼容非 ASCII 路径。

    **踩坑记录**：本项目的所有路径都含中文（``AI实时构图指导Agent``），
    ``cv2.imread`` 在这类路径上**静默返回 None**（只打一条 WARN 到 stderr，
    不抛异常）。因此必须用 ``np.fromfile`` + ``cv2.imdecode`` 兜底。
    此处捕获 ``Exception`` 而非仅 ``OSError``——``np.fromfile`` 对参数
    类型敏感，异常类型不固定，收窄捕获会漏掉失败、让调用方拿到 None。
    """
    if not p.exists():
        return None
    img = cv2.imread(str(p))
    if img is not None:
        return img
    try:
        buf = np.fromfile(str(p), dtype=np.uint8)
        if buf.size == 0:
            return None
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except Exception:  # noqa: BLE001
        return None


def main() -> int:
    setup_logging("WARNING")
    root = PROJECT_ROOT / "assets" / "demo_photos"
    items = json.loads((root / "_index.json").read_text(encoding="utf-8"))

    CELL_W, CELL_H = 260, 347
    COLS = 5
    rows = (len(items) + COLS - 1) // COLS
    PAD, LABEL_H = 8, 34
    sheet = np.full(
        (rows * (CELL_H + LABEL_H + PAD) + PAD,
         COLS * (CELL_W + PAD) + PAD, 3), 245, np.uint8)

    for i, it in enumerate(items):
        img = imread_any(root / it["file"])
        if img is None:
            log.warning(f"{it['file']} 读取失败")
            continue
        cell = cv2.resize(img, (CELL_W, CELL_H), interpolation=cv2.INTER_AREA)
        r, c = divmod(i, COLS)
        y = PAD + r * (CELL_H + LABEL_H + PAD)
        x = PAD + c * (CELL_W + PAD)
        sheet[y:y + CELL_H, x:x + CELL_W] = cell

        # 标注：序号 + 实测属性（ASCII）
        lines = [
            f"#{i+1} {it['shot_class']}",
            f"{it['face_kind']} n={it['n_person_detected']} area={it['subject_area_ratio']:.2f}",
            f"score={it['composition_score']:.0f} sharp={it['sharpness_laplacian_var']:.0f}",
        ]
        for j, t in enumerate(lines):
            cv2.putText(sheet, t, (x + 3, y + CELL_H + 11 + j * 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, (40, 40, 40), 1, cv2.LINE_AA)

    out = PROJECT_ROOT / "outputs" / "reports" / "demo_photos_contact_sheet.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", sheet)
    if not ok:
        print("[错误] 拼图编码失败")
        return 1
    with open(out, "wb") as f:
        f.write(buf.tobytes())
    print(f"拼图 -> {out}  ({sheet.shape[1]}x{sheet.shape[0]}, {len(items)} 张)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
