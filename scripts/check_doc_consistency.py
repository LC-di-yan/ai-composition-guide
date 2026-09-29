#!/usr/bin/env python
"""文档口径一致性校验：让"改了一处、忘了三处"再也藏不住。

本脚本源于本项目反复出现的一类真实缺陷：**同一个数字/路径散落在多个文档里，
改了源头却漏改副本**，导致文档之间互相打脸。已经踩过四次的坑：

1. 延迟 P95 = 24.4ms 被写进多个文档，后续复测无法复现（只登记了最优读数）。
2. API 路径/字段名在 8 处文档与运行时 schema 不符。
3. 测试用例数 331 → 334 后，README / 测试与验收 / 开发计划 三处未同步。
4. 缺陷文档重命名后，10 处引用未同步会直接变成死链。

因此本校验器做三件事：

* **死链检查**：扫描所有 markdown/源码里的 `docs/...`、`scripts/...`、
  `src/...` 反引号路径，确认文件真实存在。
* **口径检查**：对每个"易漂移事实"（测试数、抖动指标、文件名）声明
  **允许出现的取值集合**，并检查文档里出现的数字是否在集合内。
* **真源单一性**：报告每个事实在哪些文件出现，方便确认"改了源头"时
  需要连带修改哪些副本。

用法：
    python scripts/check_doc_consistency.py            # 全部检查
    python scripts/check_doc_consistency.py --paths    # 只查死链
    python scripts/check_doc_consistency.py --facts    # 只查口径
    python scripts/check_doc_consistency.py --quiet    # 只在失败时输出

退出码：0 = 全部通过；1 = 发现问题（可接 CI）。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# 扫描范围：只扫"活的"文档与源码。
# outputs/reports/acceptance_*.json 之类的**历史快照刻意排除**——
# 历史记录不该被改写，它们的旧数字是正确的。
# ---------------------------------------------------------------------------
DOC_GLOBS = ["*.md", "docs/**/*.md", "web/**/*.html"]
SRC_GLOBS = ["src/**/*.py", "scripts/**/*.py", "tests/**/*.py", "configs/**/*.yaml"]

# 历史快照 / 生成物目录：跳过（内容不可改，且含大量旧数字）
EXCLUDE_PARTS = {
    "outputs",           # 含历史 acceptance_*.json 与评测报告快照
    ".venv", "venv", "__pycache__", ".pytest_cache", "node_modules",
    "images_real", "images", "demo_photos", "topic_photos",  # 素材目录
}

# ---------------------------------------------------------------------------
# 测试口径的**唯一真源**：scripts/test_counts.json
#
# 这两个数字以前写死在本文件的 FACTS 里，于是每次新增用例都要手工改一堆地方
# （本文件 2 处 + 若干文档），到第三次终于改成现在的分工：
#     scripts/sync_test_counts.py  ← 负责**写**（依据真实 pytest 输出）
#     本文件                        ← 负责**读**（这里是守卫，不是账本）
# 基线文件缺失/损坏时退回内置值，并明确告警——宁可噪音提醒，也不能让门禁静默失效。
# ---------------------------------------------------------------------------
_COUNTS_PATH = PROJECT_ROOT / "scripts" / "test_counts.json"
_FALLBACK_COUNTS = {"passed": 367, "collected": 374, "skipped": 7}


def _load_test_counts() -> dict:
    if _COUNTS_PATH.exists():
        try:
            data = json.loads(_COUNTS_PATH.read_text(encoding="utf-8"))
            if "passed" in data and "collected" in data:
                return data
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[warn] 读不到 {_COUNTS_PATH.name}（{exc}），退回内置基线",
                  file=sys.stderr)
    print(f"[warn] {_COUNTS_PATH.name} 不可用，退回内置基线——请先跑 "
          f"scripts/sync_test_counts.py 写入真源", file=sys.stderr)
    return dict(_FALLBACK_COUNTS)


_TEST_COUNTS = _load_test_counts()
PASSED = str(_TEST_COUNTS["passed"])
COLLECTED = str(_TEST_COUNTS["collected"])
SKIPPED = int(_TEST_COUNTS.get("skipped", 0))

# ---------------------------------------------------------------------------
# 事实登记表：每个"易漂移事实"声明**全部合法取值**。
# 只要文档里出现的数字不在集合内，就报错——等于把"口径变更"强制显式化。
#
# 字段：
#   pattern  : 捕获组 1 是数字或字符串
#   allowed  : 允许出现的合法值集合（字符串比较）
#   why      : 为什么这些值都合法（写给人看，避免后人误删）
# ---------------------------------------------------------------------------
FACTS: list[dict] = [
    {
        "name": "测试通过数",
        "pattern": r"(\d{3})\s*(?:passed|用例通过|用例)",
        "allowed": {PASSED},
        "why": f"唯一当前值 {PASSED}（= 收集 {COLLECTED} − skipped {SKIPPED}），\n"
               f"来自唯一真源 `scripts/test_counts.json`，\n"
               f"**不要在这里改数字**：跑 scripts/sync_test_counts.py 写入真源即可。\n"
               "历史值（331 / 334 / 355 / 363…）出现在 CHANGELOG 与缺陷快照文档里"
               "是**对的**，已豁免；其余文档只允许写当前值。",
        # 两个豁免文件都是**流水账/历史快照**：其中的旧值是当时的事实，
        # 改成当前值等于伪造历史。其余文档必须只写当前值。
        # - CHANGELOG.md：版本流水账
        # - D-10/D-11 缺陷文档：修复当时的回归测试快照（331 → 334）
        "exempt_files": {
            "CHANGELOG.md",
            "构图指令一致性与防抖闩锁缺陷_D-10_D-11.md",
        },
    },
    {
        "name": "测试收集总数",
        "pattern": r"(\d{3})\s*项\s*/",
        "allowed": {COLLECTED},
        "why": f"{COLLECTED} = {PASSED} passed + {SKIPPED} skipped，\n"
               f"同样来自唯一真源 `scripts/test_counts.json`。\n"
               "与「测试通过数」是**两个口径**：总数用「N 项 /」后缀，通过数用\n"
               "「N passed / N 用例」后缀——禁止混用（曾因写「335 用例」被抓）。",
        "exempt_files": {"CHANGELOG.md"},
    },
    {
        "name": "防抖降幅",
        "pattern": r"降幅?\s*\**\s*(\d{2}\.\d{2})%|切\w*\s*-?(\d{2}\.\d{2})%",
        "allowed": {"77.38", "93.02"},
        "why": "77.38% = acceptance_report.py 官方验收基线（84→19）；"
               "93.02% = 修 D-11 闩锁后实测（84→6）。**两个都必须保留**，"
               "只留一个就是在藏口径变更。",
    },
    {
        "name": "缺陷文档路径",
        "pattern": r"docs/research/([\w\u4e00-\u9fff\-]+\.md)",
        "allowed": {
            "构图指令一致性与防抖闩锁缺陷_D-10_D-11.md",
            "素材图源调研.md",
            "Milvus本地方案调研.md",
        },
        "why": "重命名后旧名 `距离建议一致性缺陷_D-10.md` 已作废，出现即为死链。\n"
               "新增调研文档必须登记进本集合——否则代码注释里引用它就报死链"
               "（这正是本门禁的**设计意图**：引用即须存在）。",
        "is_path": True,
    },
]

# ---------------------------------------------------------------------------
# 路径存在性检查
# ---------------------------------------------------------------------------
PATH_RE = re.compile(
    r"`(?P<p>(?:docs|scripts|src|tests|configs|web|assets|outputs)/[\w\u4e00-\u9fff./\-]+?)`"
)
# 这些是"示意路径"（通配符/占位/示例），不做存在性校验
# 例：`tests/fixtures/x.jpg`（文档里用 x.jpg 表示"随便一个图片"）、
#     `src/foo/bar.py`、`data:image/...` 等
PATH_PLACEHOLDER = re.compile(
    r"[*<>{}]|\bxxx\b|待确认|TODO|"
    r"/x\.\w+$|"          # 形如 tests/fixtures/x.jpg 的示意文件名
    r"/foo/|/bar\.|/baz"  # 常见的占位命名
)

# 「方案 vs 实际」对照行：这一行的路径是**被否决/被重命名/未实现**的对象，
# 它"不存在"恰恰是正确状态。判据是同一行里出现下列任一标记。
NEGATIVE_MARKERS = re.compile(
    r"更名为|合并进|修正|简化|未实现|未创建|不作为|改为|废弃|已移除|"
    r"⏸|❌|~~|→ \*\*|禁止|不要用|旧名|原名|"
    r"已随|已清理|临时文件|草案|曾经|未落地"
)


def _has_negative_marker(line: str) -> bool:
    """该行是否在描述一个「故意不存在」的路径。"""
    return bool(NEGATIVE_MARKERS.search(line))


# 可再生产物前缀：这些路径由脚本在运行时生成（outputs/ 由验收与演示脚本
# 重建，models/ 由 download_weights.py 重建），**刻意不进版本库**。
# 文档引用它们是合法的——但在全新克隆（CI runner / 评审者机器）上必然
# "不存在"。存在性校验只对「应当随仓库分发的文件」有意义；
# 本地 76MB outputs/ 在时这条豁免看不出来，CI 首跑（全新克隆）才暴露。
REGENERABLE_PREFIXES = ("outputs/", "models/")


def iter_files(globs: list[str]) -> list[Path]:
    out: list[Path] = []
    for g in globs:
        for p in PROJECT_ROOT.glob(g):
            if not p.is_file():
                continue
            if any(part in EXCLUDE_PARTS for part in p.relative_to(PROJECT_ROOT).parts):
                continue
            if p.name == "check_doc_consistency.py":  # 跳过自检（含示例路径）
                continue
            out.append(p)
    return sorted(set(out))


def scan_paths(files: list[Path]) -> list[str]:
    problems: list[str] = []
    for f in files:
        try:
            text = f.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        rel = f.relative_to(PROJECT_ROOT)
        lines = text.splitlines()
        for m in PATH_RE.finditer(text):
            raw = m.group("p")
            if PATH_PLACEHOLDER.search(raw):
                continue
            line_no = text[: m.start()].count("\n") + 1
            line = lines[line_no - 1] if line_no <= len(lines) else ""
            # 示例/对照行不做存在性校验（"不存在"是其正确语义）
            if _has_negative_marker(line):
                continue
            # 可再生产物：不进版本库，新克隆上必然不存在，豁免存在性校验
            if raw.startswith(REGENERABLE_PREFIXES):
                continue
            target = PROJECT_ROOT / raw.rstrip("/")
            hit = target.exists() or target.with_suffix(".py").exists()
            if not hit:
                problems.append(f"{rel}:{line_no}  死链 → `{raw}`")
    return problems


def scan_facts(files: list[Path]) -> tuple[list[str], dict[str, list[str]]]:
    problems: list[str] = []
    occurrences: dict[str, list[str]] = {f["name"]: [] for f in FACTS}
    for f in files:
        try:
            text = f.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        rel = f.relative_to(PROJECT_ROOT)
        rel_s = rel.as_posix()
        for fact in FACTS:
            # CHANGELOG 等流水账：历史段旧值合法，整文豁免（但仍统计出现位置）
            # 豁免名支持**路径或文件名**两种写法：早期只比完整相对路径，
            # 导致 docs/research/ 下的文件写了文件名却不生效（CHANGELOG
            # 因为在根目录才碰巧命中）。任一匹配即可，避免位置耦合。
            exempt_files = fact.get("exempt_files", set())
            exempt = rel_s in exempt_files or Path(rel_s).name in exempt_files
            for m in re.finditer(fact["pattern"], text):
                val = next((g for g in m.groups() if g), None)
                if val is None:
                    continue
                occurrences[fact["name"]].append(rel_s)
                if exempt:
                    continue
                if val not in fact["allowed"]:
                    line = text[: m.start()].count("\n") + 1
                    problems.append(
                        f"{rel_s}:{line}  「{fact['name']}」出现非法值 {val!r} "
                        f"（合法：{sorted(fact['allowed'])}）"
                    )
    return problems, occurrences


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="文档口径一致性校验")
    ap.add_argument("--paths", action="store_true", help="只查死链")
    ap.add_argument("--facts", action="store_true", help="只查口径")
    ap.add_argument("--quiet", action="store_true", help="只在失败时输出")
    args = ap.parse_args(argv)

    do_paths = args.paths or not (args.paths or args.facts)
    do_facts = args.facts or not (args.paths or args.facts)

    files = iter_files(DOC_GLOBS + SRC_GLOBS)
    problems: list[str] = []

    if not args.quiet:
        print(f"[扫描] {len(files)} 个文件（已排除 outputs/ 等历史快照与素材目录）")

    if do_paths:
        p = scan_paths(files)
        problems += p
        if not args.quiet:
            n = len(PATH_RE.findall("\n".join(
                f.read_text(encoding="utf-8", errors="ignore") for f in files)))
            print(f"[死链] 检查 ~{n} 个行内路径引用 → "
                  f"{'通过' if not p else f'{len(p)} 个问题'}")

    if do_facts:
        p, occ = scan_facts(files)
        problems += p
        if not args.quiet:
            print("[口径] 易漂移事实登记：")
            for name, where in occ.items():
                uniq = sorted(set(where))
                print(f"  · {name}: 出现于 {len(uniq)} 个文件")
            print(f"[口径] {'通过' if not p else f'{len(p)} 个问题'}")

    if problems:
        print("\n发现问题：")
        for x in problems:
            print("  ✗ " + x)
        print(f"\n合计 {len(problems)} 个问题（退出码 1）")
        return 1

    if not args.quiet:
        print("\n全部通过（退出码 0）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
