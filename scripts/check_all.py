#!/usr/bin/env python
"""一键质量门禁：把全部「秒级静态门禁」收敛到单一入口。

为什么需要它：

项目的门禁分散在三个脚本里（文档口径 / Docker 编排 / CI 工作流），
每个都要单独跑、单独看退出码 —— 结果就是**总有人忘记跑其中某个**。
本项目反复出现"改了源头漏改副本"、"标了已备但没测过"的教训，
对策是把"跑门禁"这件事本身也变成一条命令。

本脚本**不包含 pytest / 验收报告**（那些是分钟级、需要模型环境），
只跑零依赖、秒级的静态门禁，适合：

* 写完文档 / 改完编排后随手跑一次
* 提交前跑（`git pre-commit` 也可接，见 `.pre-commit-config.yaml`）
* CI 的 gate job（`.github/workflows/ci.yml` 逐项调用同一批脚本）

用法：
    python scripts/check_all.py                # 全部静态门禁
    python scripts/check_all.py --with-tests   # 追加全量 pytest（分钟级）
    python scripts/check_all.py --list         # 只列出门禁清单

退出码：0 = 全部通过；1 = 任一门禁失败。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# (名称, 脚本相对路径, 传参, 说明)
GATES: list[tuple[str, str, list[str], str]] = [
    ("文档口径", "scripts/check_doc_consistency.py", [],
     "死链 + 易漂移事实取值（测试数/指标/文件路径）"),
    ("Docker 编排", "scripts/check_docker_static.py", ["--verbose"],
     "入口/COPY源/构建参数/命令依赖/挂载/ignore 六类静态校验"),
    ("CI 工作流", "scripts/check_ci_static.py", ["--verbose"],
     "workflow YAML 结构 + 脚本引用与参数签名"),
]


def run_gate(name: str, script: str, args: list[str]) -> bool:
    cmd = [sys.executable, str(PROJECT_ROOT / script), *args]
    r = subprocess.run(cmd, capture_output=True, text=True,
                       cwd=PROJECT_ROOT, errors="replace")
    ok = r.returncode == 0
    mark = "✅" if ok else "❌"
    print(f"{mark} {name}（{'通过' if ok else '失败'}）")
    out = (r.stdout or "") + (r.stderr or "")
    out = out.strip()
    if out and (not ok or "-v" in args or "--verbose" in args):
        for line in out.splitlines():
            print("    " + line)
    return ok


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="一键静态质量门禁")
    ap.add_argument("--with-tests", action="store_true",
                    help="追加全量 pytest（分钟级，需完整依赖环境）")
    ap.add_argument("--list", action="store_true", help="只列出门禁清单")
    args = ap.parse_args(argv)

    if args.list:
        print("静态门禁清单：")
        for name, script, _, desc in GATES:
            print(f"  · {name:8} {script}  — {desc}")
        if args.with_tests:
            print("  · 全量 pytest（--with-tests 时追加）")
        return 0

    print(f"一键静态门禁（{len(GATES)} 项）\n" + "-" * 46)
    results: list[tuple[str, bool]] = []
    for name, script, gate_args, _ in GATES:
        results.append((name, run_gate(name, script, gate_args)))

    if args.with_tests:
        print("-" * 46)
        print("追加：全量 pytest ...")
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", "--no-header"],
                           cwd=PROJECT_ROOT)
        results.append(("pytest", r.returncode == 0))

    print("-" * 46)
    failed = [n for n, ok in results if not ok]
    for name, ok in results:
        print(f"  {'✅' if ok else '❌'} {name}")
    if failed:
        print(f"\n{len(failed)} 项失败：{failed}（退出码 1）")
        return 1
    print(f"\n全部 {len(results)} 项通过（退出码 0）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
