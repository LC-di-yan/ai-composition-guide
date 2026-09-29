#!/usr/bin/env python
"""生成人工抽检校准表 + 拼图，用于对原则驱动标注做交叉验证。

用法::

    python scripts/make_review_sheet.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from aicg.settings import PROJECT_ROOT  # noqa: E402

ANN = PROJECT_ROOT / "configs" / "eval" / "annotations_real.json"
SRCC = PROJECT_ROOT / "outputs" / "reports" / "srcc_real.json"
IMG_DIR = PROJECT_ROOT / "configs" / "eval" / "images_real"
OUT_MD = PROJECT_ROOT / "outputs" / "reports" / "annotation_review_sheet.md"
OUT_PNG = PROJECT_ROOT / "outputs" / "reports" / "annotation_review_contact_sheet.png"


def main() -> int:
    ann = json.loads(ANN.read_text(encoding="utf-8"))
    srcc = json.loads(SRCC.read_text(encoding="utf-8"))
    per = {Path(p["image"]).name: p for p in srcc["per_item"]}

    items = ann["items"]
    random.seed(20260928)
    by_score = sorted(items, key=lambda x: -x["human_score"])

    segments = [
        ("高分（原则分 ≥ 70）", by_score[:12]),
        ("中分（原则分 50–70）", [x for x in by_score if 50 <= x["human_score"] < 70]),
        ("低分（原则分 < 50）", [x for x in by_score if x["human_score"] < 50]),
    ]
    picks: list[tuple[str, list[dict]]] = []
    for label, seg in segments:
        n = min(6, len(seg))
        picks.append((label, random.sample(seg, n)))

    # ---- Markdown ----
    lines = [
        "# 人工抽检校准表（18 张）",
        "",
        "> **用途**：对「原则驱动标注」做人工交叉验证 —— 检验原则分与你**主观判断**是否同向。",
        "> 这是把 SRCC 从「与构图原则同向」升级到「符合人类审美」的唯一途径。",
        "",
        "## 怎么用",
        "",
        "1. 打开 `outputs/reports/annotation_review_contact_sheet.png`（已按分组拼好，编号 1–18，",
        "   每格下方标了文件名），或直接打开 `configs/eval/images_real/<文件名>`；",
        "2. **先不看表格里的「原则分」「自动分」**，凭直觉给每张图打一个 0–100 的主观分",
        "   （标准自定，只要前后一致即可）；",
        "3. 填进「你的分」列；把表发回给我，我会计算：",
        "   - 你 vs 原则分 的 SRCC（检验原则分是否贴近真人）",
        "   - 你 vs 自动分 的 SRCC（检验评分器是否贴近真人）",
        "   - 三者两两对照，据实更新结论与局限说明。",
        "",
        "> 提示：不必追求「专业摄影师水准」。你**稳定且诚实**的直觉，就是有效的参照。",
        "",
        "---",
        "",
    ]

    ordered: list[tuple[str, dict]] = []
    for label, grp in picks:
        lines += [f"## {label}", "",
                  "| # | 图片 | 原则分 | 自动分 | 主体占比 | 你的分（0–100） | 一句话理由 |",
                  "|---|---|---|---|---|---|---|"]
        for it in grp:
            fn = Path(it["image"]).name
            auto = per.get(fn, {}).get("auto_score", "-")
            idx = len(ordered) + 1
            ordered.append((fn, it))
            lines.append(
                f"| {idx} | `{fn}` | {it['human_score']} | {auto} | "
                f"{it['area_ratio']} | | |"
            )
        lines.append("")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---- Contact sheet ----
    CELL_W, CELL_H = 300, 260
    COLS = 6
    ROWS = (len(ordered) + COLS - 1) // COLS
    sheet = np.full((ROWS * CELL_H, COLS * CELL_W, 3), 245, dtype=np.uint8)

    for i, (fn, it) in enumerate(ordered):
        r, c = divmod(i, COLS)
        p = IMG_DIR / fn
        img = cv2.imread(str(p))
        if img is None:
            img = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        # 等比缩放进格子（留出底部文字带）
        tw, th = CELL_W - 12, CELL_H - 34
        h, w = img.shape[:2]
        s = min(tw / w, th / h)
        nw, nh = max(1, int(w * s)), max(1, int(h * s))
        small = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

        y0, x0 = r * CELL_H + 6, c * CELL_W + 6
        sheet[y0:y0 + nh, x0:x0 + nw] = small
        # 边框
        cv2.rectangle(sheet, (x0 - 1, y0 - 1), (x0 + nw, y0 + nh), (170, 170, 170), 1)

        # 文字带（ASCII only：Hershey 字体无 CJK 字形）
        ty = r * CELL_H + nh + 20
        cv2.putText(sheet, f"#{i+1} {fn}", (x0, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (30, 30, 30), 1, cv2.LINE_AA)
        cv2.putText(sheet, f"principle={it['human_score']:.0f} area={it['area_ratio']:.3f}",
                    (x0, ty + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (100, 100, 100), 1, cv2.LINE_AA)

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", sheet)
    if ok:
        with open(OUT_PNG, "wb") as f:
            f.write(buf.tobytes())

    print(f"抽检表    -> {OUT_MD}")
    print(f"拼图      -> {OUT_PNG}  ({len(ordered)} 张)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
