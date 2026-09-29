#!/usr/bin/env python
"""CI 工作流静态校验：不跑 Actions 也能抓出必然失败项。

为什么需要：`.github/workflows/` 里的 YAML 在本机**无法执行**（需要 GitHub
runner），因此它属于"未实测交付物"。而它的失败代价很隐蔽 —— 你可能几周后
才发现 CI 一直在红，或者更糟：CI 一直绿着但**其实没在检查任何东西**
（例如 `if: always()` 写错、step 里命令拼错但被 `|| true` 吞掉）。

本脚本把已知的坑固化为检查项。诞生原因就是实打实踩到的两类问题：

1. **引用了不存在的脚本参数**：Docker workflow 的冒烟步骤里我写了
   `run_demo.py --limit 5`，但真实参数是 `--max-frames`。
   → 检查项 D 会比对 `scripts/*.py` 的 argparse 定义。
2. **step 名/结构写错**：YAML 能解析但语义无效。

检查项：
  A. YAML 可解析；每个 job 有 runs-on + steps；每个 step 有 uses 或 run
  B. 引用的 action（uses）版本格式合法
  C. 引用的本地脚本路径存在
  D. step 里调用的 `python scripts/xxx.py --flag` 的 flag 在该脚本的
     argparse 里真实存在
  E. 引用的 artifact 路径 glob 至少能匹配到东西（软检查）
  F. `on` 触发器存在且合法

用法：
    python scripts/check_ci_static.py
    python scripts/check_ci_static.py --verbose

退出码：0 = 通过；1 = 发现必然失败项。
"""

from __future__ import annotations

import argparse
import glob
import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS_DIR = PROJECT_ROOT / ".github" / "workflows"

VALID_EVENTS = {
    "push", "pull_request", "workflow_dispatch", "schedule", "release",
    "workflow_call", "workflow_run", "pull_request_target", "issues",
    "issue_comment", "merge_group", "repository_dispatch", "create", "delete",
}


class Finding:
    def __init__(self, level: str, code: str, msg: str) -> None:
        self.level, self.code, self.msg = level, code, msg

    def __str__(self) -> str:
        return f"[{self.level}] {self.code}  {self.msg}"


def iter_workflows() -> list[Path]:
    if not WORKFLOWS_DIR.exists():
        return []
    return sorted(list(WORKFLOWS_DIR.glob("*.yml")) + list(WORKFLOWS_DIR.glob("*.yaml")))


def load(path: Path):
    import yaml  # type: ignore
    return yaml.safe_load(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# D. 比对脚本参数
# ---------------------------------------------------------------------------
# 缓存各脚本的合法 flag 集合
_FLAG_CACHE: dict[str, set[str] | None] = {}


def script_flags(script_rel: str) -> set[str] | None:
    """返回脚本 argparse 支持的所有 flag；无法确定时返回 None。"""
    if script_rel in _FLAG_CACHE:
        return _FLAG_CACHE[script_rel]
    p = PROJECT_ROOT / script_rel
    if not p.exists():
        _FLAG_CACHE[script_rel] = None
        return None
    try:
        # 用 --help 输出解析，比静态 AST 更贴近真实行为
        r = subprocess.run(
            [sys.executable, str(p), "--help"],
            capture_output=True, text=True, timeout=60, cwd=PROJECT_ROOT,
        )
        text = r.stdout + r.stderr
        if "--help" in text or "usage:" in text:
            flags = set(re.findall(r"(--[a-zA-Z][\w-]*)", text))
            _FLAG_CACHE[script_rel] = flags
            return flags
    except Exception:  # noqa: BLE001
        pass
    _FLAG_CACHE[script_rel] = None
    return None


PY_CALL_RE = re.compile(r"python[3]?\s+(scripts/[\w/]+\.py)((?:\s+--?[\w-]+(?:[=\s]\S+)?)*)")


def normalize_shell(text: str) -> str:
    """把 shell 续行折成一行。

    关键：YAML 里的 run 几乎总是多行 shell（每行以 `\\` 续行），
    而参数只能跨行提取。实测踩过：正则里用 `[^\\s\\\\]+` 会在反斜杠处截断，
    导致 `--limit`（跨行出现在第 4 行）完全提取不到，门禁静默漏判。
    """
    return re.sub(r"\\\s*\n\s*", " ", text)


def check_script_flags(text: str, where: str, findings: list[Finding]) -> None:
    text = normalize_shell(text)
    for m in PY_CALL_RE.finditer(text):
        script = m.group(1)
        args_blob = m.group(2) or ""
        if not (PROJECT_ROOT / script).exists():
            findings.append(Finding(
                "FAIL", "CI-SCRIPT-001",
                f"{where} 调用的脚本不存在：`{script}`",
            ))
            continue
        flags = script_flags(script)
        if flags is None:
            continue  # 无法确定（脚本可能不支持 --help），跳过
        for flag in re.findall(r"(--[\w-]+)", args_blob):
            if flag in ("--help",):
                continue
            if flag not in flags:
                # 可能是缩写或子命令，给出建议
                close = sorted(f for f in flags if f.startswith(flag[:4]))
                hint = f"（相近的合法参数：{close}）" if close else ""
                findings.append(Finding(
                    "FAIL", "CI-SCRIPT-002",
                    f"{where} 给 `{script}` 传了未定义的参数 `{flag}`{hint}",
                ))


# ---------------------------------------------------------------------------
# 主检查
# ---------------------------------------------------------------------------
def check_workflow(path: Path, findings: list[Finding]) -> dict | None:
    rel = path.relative_to(PROJECT_ROOT).as_posix()
    try:
        doc = load(path)
    except Exception as e:  # noqa: BLE001
        findings.append(Finding("FAIL", "CI-YAML-001", f"{rel} YAML 解析失败：{e}"))
        return None
    if not isinstance(doc, dict):
        findings.append(Finding("FAIL", "CI-YAML-002", f"{rel} 顶层不是映射"))
        return None

    # F. 触发器
    trigger = doc.get("on", doc.get(True))
    if trigger is None:
        findings.append(Finding("FAIL", "CI-TRIG-001", f"{rel} 缺少 `on` 触发器"))
    elif isinstance(trigger, str):
        if trigger not in VALID_EVENTS:
            findings.append(Finding("FAIL", "CI-TRIG-002",
                                    f"{rel} 触发器 `{trigger}` 不是合法事件名"))
    elif isinstance(trigger, dict):
        for ev in trigger:
            if ev not in VALID_EVENTS:
                findings.append(Finding("FAIL", "CI-TRIG-003",
                                        f"{rel} 触发器 `{ev}` 不是合法事件名"))

    # A. jobs / steps 结构
    jobs = doc.get("jobs") or {}
    if not jobs:
        findings.append(Finding("FAIL", "CI-JOB-001", f"{rel} 没有任何 job"))
    for jname, job in jobs.items():
        if not isinstance(job, dict):
            findings.append(Finding("FAIL", "CI-JOB-002", f"{rel} job `{jname}` 不是映射"))
            continue
        if "runs-on" not in job:
            findings.append(Finding("FAIL", "CI-JOB-003", f"{rel} job `{jname}` 缺 runs-on"))
        if "uses" in job:
            continue  # reusable workflow
        steps = job.get("steps")
        if not steps:
            findings.append(Finding("FAIL", "CI-JOB-004", f"{rel} job `{jname}` 缺 steps"))
            continue
        for i, st in enumerate(steps):
            if not isinstance(st, dict):
                findings.append(Finding("FAIL", "CI-STEP-001",
                                        f"{rel} {jname}.steps[{i}] 不是映射"))
                continue
            has_uses, has_run = st.get("uses"), st.get("run")
            if not (has_uses or has_run):
                findings.append(Finding("FAIL", "CI-STEP-002",
                                        f"{rel} {jname}.steps[{i}] 既无 uses 也无 run"))
            # B. action 版本格式
            if has_uses:
                if "@" not in has_uses:
                    findings.append(Finding("WARN", "CI-ACTION-001",
                                            f"{rel} action `{has_uses}` 未固定版本（建议 @v4）"))
            # D. 脚本参数
            if has_run:
                check_script_flags(has_run, f"{rel} `{jname}` step#{i+1}", findings)
            # E. upload-artifact 路径
            if has_uses and "upload-artifact" in str(has_uses):
                for pth in str(st.get("with", {}).get("path", "")).splitlines():
                    pth = pth.strip()
                    if not pth:
                        continue
                    if glob.glob(str(PROJECT_ROOT / pth)):
                        continue
                    if st.get("with", {}).get("if-no-files-found") == "warn":
                        continue
                    findings.append(Finding(
                        "WARN", "CI-ART-001",
                        f"{rel} upload-artifact 的路径当前无匹配：`{pth}`"
                        f"（测试前为空属正常；若始终为空则说明产物名变更）",
                    ))
    return doc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="CI 工作流静态校验")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    wfs = iter_workflows()
    if not wfs:
        print("未找到 .github/workflows/*.yml")
        return 0

    findings: list[Finding] = []
    for wf in wfs:
        check_workflow(wf, findings)

    fails = [f for f in findings if f.level == "FAIL"]
    warns = [f for f in findings if f.level == "WARN"]

    if args.verbose or fails or warns:
        print(f"CI 工作流静态校验（{len(wfs)} 个文件）")
        for f in fails + warns:
            print("  " + str(f))

    if fails:
        print(f"\n合计 {len(fails)} 个必然失败项、{len(warns)} 个警告（退出码 1）")
        return 1
    if not (args.verbose or warns):
        print(f"CI 工作流静态校验：通过（{len(wfs)} 个文件，退出码 0）")
    else:
        print(f"\n无必然失败项；{len(warns)} 个警告（退出码 0）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
