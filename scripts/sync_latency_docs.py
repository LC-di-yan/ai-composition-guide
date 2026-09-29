#!/usr/bin/env python
"""把文档中的延迟数字统一校正为「多次实测区间」。

背景：早期记录的 P95 = 24.4ms 无法在后续复测中复现（复测为 29.5~31.6ms 均值、
P95 33.7~37.2ms）。为避免文档记录一个挑出来的最优点值，统一改为
**区间表述 + 注明测量日期与条件**。

用法::

    python scripts/sync_latency_docs.py --dry-run
    python scripts/sync_latency_docs.py
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aicg.settings import PROJECT_ROOT  # noqa: E402

# 实测区间（2026-09-28，yolo/cuda，3 次独立复测 + 1 次 3 轮中位数）
NEW = "P95 33.7–37.2ms（均值 29.5–31.6ms）"
NEW_SHORT = "P95 ~34–37ms"
PCT = "10.1–11.2%"
PCT_SHORT = "~10–11%"

DOCS = [
    "README.md",
    "CHANGELOG.md",
    "开发计划.md",
    "技术方案.md",
    "测试与验收.md",
    "需求说明.md",
]


def patch(text: str) -> tuple[str, int]:
    n = 0
    subs = [
        # 最常见：P95 24.4ms（7.3%）一类
        (r"P95\s*24\.4\s*ms（7\.3%）", f"{NEW}（{PCT}）"),
        (r"P95\s*24\.4ms（7\.3%）", f"{NEW}（{PCT}）"),
        (r"\*\*24\.4\s*ms\*\*（预算 333ms 的 \*\*7\.3%\*\*；多次运行区间 24–28ms）",
         f"**{NEW_SAFE}**"),
        (r"P95 \*\*24\.4\s*ms\*\* = 预算 7\.3%", f"P95 **{NEW_SAFE}** = 预算 {PCT_SHORT}"),
        (r"P95 \*\*24\.4ms\*\*（预算 333ms 的 7\.3%）", f"P95 **{NEW_SHORT}**（预算 333ms 的 {PCT_SHORT}）"),
        (r"\*\*P95 24\.4ms（7\.3%）\*\*", f"**{NEW_SAFE}（{PCT}）**"),
        (r"P95 \*\*24\.4ms\*\* = 预算\(333ms\) 的 \*\*7\.3%\*\*；均值 19\.4ms；P50 20\.2ms",
         f"P95 **{NEW_SHORT}** = 预算(333ms) 的 **{PCT_SHORT}**；均值 29.5–31.6ms"),
        (r"P95 24\.4ms", NEW_SHORT),
        (r"实测 24\.4ms", f"实测 {NEW_SHORT}"),
        (r"实测 P95 \*\*24\.4ms\*\*（\*\*7\.3%\*\*，多次运行 24–28ms）",
         f"实测 **{NEW_SAFE}**（**{PCT}**）"),
        (r"P95 28\.2ms（8\.4%） \| P95 24\.4ms（7\.3%），多次运行 24–28ms",
         f"P95 28.2ms（8.4%） | **{NEW_SAFE}（{PCT}）**，已按复测校正"),
    ]
    for pat, rep in subs:
        text, k = re.subn(pat, rep, text)
        n += k
    return text, n


NEW_SAFE = "P95 33.7–37.2 ms"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    total = 0
    for name in DOCS:
        p = PROJECT_ROOT / name
        if not p.exists():
            print(f"[跳过] 不存在: {name}")
            continue
        raw = p.read_text(encoding="utf-8")
        new, n = patch(raw)
        if n:
            total += n
            print(f"{name}: {n} 处")
            if not args.dry_run:
                p.write_text(new, encoding="utf-8")
        else:
            print(f"{name}: 0 处")
    print(f"\n合计 {total} 处" + ("（dry-run，未写入）" if args.dry_run else "，已写入"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
