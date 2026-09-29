#!/usr/bin/env python
"""为主题素材生成人工复核拼图。

对应需求：FR-11 / NFR-O3（可复现）

**为什么必须人工复核**

``fetch_topic_photos.py`` 只能保证两件事：**域名在白名单内**、**文件能解码**。
它保证不了——也保证不了——下面这三条：

1. 画面里**确实有清晰可辨的女生人物**（检索词是关键词，不是图库官方标签；
   ``woman street style`` 会返回空街景、橱窗模特、男性背影）；
2. 画面**不低俗、不敏感**（合规红线，任何自动检测都判不了）；
3. 画面**构图与光线**足以当演示素材（糊、过曝、杂乱都不合格）。

本脚本把素材拼成总览图供人眼过一遍。标注用 ASCII（OpenCV Hershey 字体
无 CJK 字形，这是既有约束）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from aicg.observability import get_logger, setup_logging  # noqa: E402
from aicg.settings import PROJECT_ROOT  # noqa: E402

log = get_logger("scripts.make_topic_sheet")


def imread_any(p: Path) -> np.ndarray | None:
    """读图，兼容非 ASCII 路径（cv2.imread 在中文路径上静默返回 None）。

    捕获 ``Exception`` 而非仅 ``OSError``：``np.fromfile`` 异常类型不固定。
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
    ap = argparse.ArgumentParser(description="主题素材复核拼图")
    ap.add_argument("--dir", default=str(PROJECT_ROOT / "assets" / "topic_photos"))
    ap.add_argument("--out", default="topic_photos_contact_sheet.png")
    ap.add_argument("--cols", type=int, default=5)
    args = ap.parse_args()

    setup_logging("WARNING")
    root = Path(args.dir)
    index_path = root / "_index.json"
    if not index_path.exists():
        print(f"[错误] 找不到索引 {index_path}")
        return 1
    items = json.loads(index_path.read_text(encoding="utf-8"))

    CELL_W, CELL_H = 300, 300
    COLS = args.cols
    rows = (len(items) + COLS - 1) // COLS
    PAD, LABEL_H = 8, 40
    sheet = np.full(
        (rows * (CELL_H + LABEL_H + PAD) + PAD,
         COLS * (CELL_W + PAD) + PAD, 3), 245, np.uint8)

    for i, it in enumerate(items):
        img = imread_any(root / it["file"])
        if img is None:
            log.warning(f"{it['file']} 读取失败")
            continue
        # 等比缩放 + 灰边填充，避免拉伸变形导致误判构图
        h, w = img.shape[:2]
        s = min(CELL_W / w, CELL_H / h)
        nw, nh = max(1, int(w * s)), max(1, int(h * s))
        cell = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        r, c = divmod(i, COLS)
        y = PAD + r * (CELL_H + LABEL_H + PAD)
        x = PAD + c * (CELL_W + PAD)
        oy, ox = y + (CELL_H - nh) // 2, x + (CELL_W - nw) // 2
        sheet[oy:oy + nh, ox:ox + nw] = cell

        lines = [
            f"#{i+1} {it['topic_requested']}",
            f"{it['width']}x{it['height']} sharp={it['sharpness_laplacian_var']:.0f}",
            f"bri={it['mean_brightness']:.2f} {it['source_host'].split('.')[0]}",
        ]
        for j, t in enumerate(lines):
            cv2.putText(sheet, t, (x + 3, y + CELL_H + 12 + j * 11),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (35, 35, 35), 1, cv2.LINE_AA)

    out = PROJECT_ROOT / "outputs" / "reports" / args.out
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
