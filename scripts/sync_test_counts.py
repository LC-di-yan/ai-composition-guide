#!/usr/bin/env python
"""把"全量测试通过数 / 收集总数"一次性同步到所有文档与门禁基线。

为什么要有这个脚本
------------------
本项目里「测试数」漂移已经手工同步过 **三次**（通过数 331→334→355→363→367，
总数 335→356→370→374），每次都要在 README / 测试与验收 / 开发计划 /
CHANGELOG / check_doc_consistency.py 之间来回改，且每一处都可能漏改。
漏改的结果不是"不好看"，而是**文档互相打脸**——这正是
`check_doc_consistency.py` 存在的理由，但它只能**抓**，不能**修**。

所以这里补上"修"的一半：**一个可执行的一次性同步**，干三件事：

1. 把实测数字写进唯一真源 `scripts/test_counts.json`；
2. 把所有文档里**符合口径**（与 check_doc_consistency.py 用同一套正则）
   的旧数字替换为新数字；
3. 打印改了哪些文件、改了几处，提醒复跑门禁确认。

刻意**不做**的事：

* 不去猜数字：数字必须由调用方给（来自真实 pytest 输出，不是估算）。
* 不改历史快照：`outputs/` 与 CHANGELOG 这类流水账豁免（旧值在当时是对的，
  改成现值等于伪造历史）。
* 不顺手改别的事实：只动测试口径两个数。

用法：
    # 依据真实 pytest 输出同步（推荐：先把 pytest 结果存成文件）
    python scripts/sync_test_counts.py --passed 367 --collected 374

    # 先看会改什么，不动手
    python scripts/sync_test_counts.py --passed 367 --collected 374 --dry-run

退出码：0 = 同步完成；1 = 参数不自洽或写入失败。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE = PROJECT_ROOT / "scripts" / "test_counts.json"

# 扫描范围与 check_doc_consistency.py 保持一致（同一套口径）。
DOC_GLOBS = ["*.md", "docs/**/*.md", "web/**/*.html"]
EXCLUDE_PARTS = {
    "outputs", ".venv", "venv", "__pycache__", ".pytest_cache", "node_modules",
    "images_real", "images", "demo_photos", "topic_photos",
}
# 历史快照 / 流水账：其中旧值是当时的事实，改了就是伪造历史。
EXEMPT_FILES = {
    "CHANGELOG.md",
    "构图指令一致性与防抖闩锁缺陷_D-10_D-11.md",
}

# 两个口径的正则与门禁脚本严格一致：
#   通过数 → "N passed / N 用例"
#   总数   → "N 项 /"
PASSED_RE = re.compile(r"(\d{3})\s*(?:passed|用例通过|用例)")
COLLECTED_RE = re.compile(r"(\d{3})\s*项\s*/")


def _load_baseline() -> dict:
    if BASELINE.exists():
        try:
            return json.loads(BASELINE.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[warn] 基线文件读不出来（{exc}），退回内置默认值", file=sys.stderr)
    return {"passed": 363, "collected": 370, "skipped": 7}


def _iter_docs():
    for pattern in DOC_GLOBS:
        for path in sorted(PROJECT_ROOT.glob(pattern)):
            if not path.is_file():
                continue
            rel = path.relative_to(PROJECT_ROOT)
            if any(part in EXCLUDE_PARTS for part in rel.parts):
                continue
            if path.name in EXEMPT_FILES:
                continue
            yield path


def _replace_in_file(path: Path, old_passed: str, new_passed: str,
                     old_collected: str, new_collected: str, dry_run: bool):
    """替换单文件中的两个口径；返回（通过数处数, 总数处数）。"""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"[warn] 读不到 {path.name}：{exc}", file=sys.stderr)
        return 0, 0

    n_passed = 0

    def _sub_passed(m: re.Match) -> str:
        nonlocal n_passed
        if m.group(1) != old_passed:
            return m.group(0)
        n_passed += 1
        return m.group(0).replace(old_passed, new_passed, 1)

    n_collected = 0

    def _sub_collected(m: re.Match) -> str:
        nonlocal n_collected
        if m.group(1) != old_collected:
            return m.group(0)
        n_collected += 1
        return m.group(0).replace(old_collected, new_collected, 1)

    new_text = PASSED_RE.sub(_sub_passed, text)
    new_text = COLLECTED_RE.sub(_sub_collected, new_text)

    if not dry_run and new_text != text:
        try:
            path.write_text(new_text, encoding="utf-8", newline="\n")
        except OSError as exc:
            print(f"[error] 写回失败 {path.name}：{exc}", file=sys.stderr)
            return 0, 0
    return n_passed, n_collected


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="同步全量测试计数到所有文档与门禁基线")
    ap.add_argument("--passed", type=int, required=True, help="真实 pytest 的 passed 数")
    ap.add_argument("--collected", type=int, required=True, help="真实 pytest 的收集总数")
    ap.add_argument("--skipped", type=int, default=None,
                    help="skipped 数（默认 = collected - passed）")
    ap.add_argument("--source", default="python -m pytest -q 实测输出",
                    help="数字来源说明（写进基线文件备查）")
    ap.add_argument("--dry-run", action="store_true", help="只报告会改什么，不落盘")
    args = ap.parse_args(argv)

    if args.passed <= 0 or args.collected <= 0 or args.passed > args.collected:
        print("[error] 数字不自洽：需满足 0 < passed <= collected", file=sys.stderr)
        return 1
    skipped = args.skipped if args.skipped is not None else args.collected - args.passed
    if skipped < 0:
        print("[error] skipped 为负数，检查传入数字", file=sys.stderr)
        return 1

    old = _load_baseline()
    old_passed, old_collected = str(old.get("passed", 0)), str(old.get("collected", 0))
    new_passed, new_collected = str(args.passed), str(args.collected)

    print(f"口径变更：通过数 {old_passed} → {new_passed}｜总数 {old_collected} → {new_collected}"
          f"（skipped {skipped}）")
    if args.dry_run:
        print("（dry-run：不会写入任何文件）")

    # 1) 文档替换
    changed: list[tuple[str, int, int]] = []
    for path in _iter_docs():
        np, nc = _replace_in_file(path, old_passed, new_passed,
                                  old_collected, new_collected, args.dry_run)
        if np or nc:
            changed.append((str(path.relative_to(PROJECT_ROOT)), np, nc))

    if changed:
        print("\n受影响文件：")
        for name, np, nc in changed:
            print(f"  - {name}：通过数 {np} 处，总数 {nc} 处")
    else:
        print("\n[warn] 没有任何文档命中这两个口径——"
              "若你确实新增了用例，说明文档里的写法已跳出正则，需回头检查口径写法。")

    # 2) 基线文件
    baseline = {
        "_comment": "全量 pytest 计数的唯一真源。由 scripts/sync_test_counts.py "
                    "依据真实 pytest 输出写入，请勿手工编辑——手工同步正是本文件要消灭的动作。",
        "passed": args.passed,
        "collected": args.collected,
        "skipped": skipped,
        "note": old.get("note", ""),
        "updated": date.today().isoformat(),
        "source": args.source,
    }
    if not args.dry_run:
        BASELINE.write_text(json.dumps(baseline, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8", newline="\n")
        print(f"\n已更新唯一真源：{BASELINE.relative_to(PROJECT_ROOT)}")
    else:
        print(f"\n将更新唯一真源：{BASELINE.relative_to(PROJECT_ROOT)}")

    print("\n下一步（必须做，别偷懒跳过）：")
    print("  python scripts/check_all.py   # 确认门禁不会因为这次同步变红")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
