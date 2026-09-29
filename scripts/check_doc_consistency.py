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
        "allowed": {"334"},
        "why": "唯一当前值。331 是 0.6.0 的历史值。",
        # CHANGELOG 是**流水账**：历史版本段落里的旧值是正确的，整文放行。
        # 其余文档（README/测试与验收/开发计划…）必须只写当前值。
        "exempt_files": {"CHANGELOG.md"},
    },
    {
        "name": "测试收集总数",
        "pattern": r"(\d{3})\s*项\s*/",
        "allowed": {"335"},
        "why": "335 = 334 passed + 1 skipped（junit tests 属性实测）。\n"
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
        "allowed": {"构图指令一致性与防抖闩锁缺陷_D-10_D-11.md", "素材图源调研.md"},
        "why": "重命名后旧名 `距离建议一致性缺陷_D-10.md` 已作废，出现即为死链。",
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
            exempt = rel_s in fact.get("exempt_files", set())
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
