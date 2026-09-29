#!/usr/bin/env python
"""Docker 编排静态校验：在**不构建镜像**的前提下找出必然失败的错误。

为什么需要这个脚本：

本项目的 `docker/Dockerfile` 与 `docker-compose.yml` **从未实测构建过**
（开发机 Docker daemon 未运行，见《技术方案.md》的诚实标注）。而"未实测的
交付物"恰恰最容易藏确定性错误 —— 事实上本脚本的诞生就是因为在静态审查里
发现了三处必错项：

1. `CMD ["uvicorn","aicg.api.app:app",...]` —— 但该模块只暴露工厂
   `create_app()`，没有模块级 `app`，容器**启动必然失败**；
2. compose 的 `test` profile 跑 `python -m pytest`，但镜像根本没装 pytest；
3. 缺 `.dockerignore`，构建上下文 114MB 全量传给 daemon。

本脚本把这些检查固化下来，避免同类错误再次潜伏到"以为能跑"。

检查项：
  A. Dockerfile 的 COPY/ADD 源路径在上下文中真实存在
  B. uvicorn 入口与模块实际导出一致（区分 app 对象 / factory）
  C. compose 各 service 的 build args 与 Dockerfile ARG 匹配
  D. compose 的 command 所需可执行/模块在镜像依赖里存在（如 pytest）
  E. compose 的 volume 挂载源存在，且**不遮蔽**镜像内已 COPY 的关键目录
  F. .dockerignore 存在，且不排除 Dockerfile 需要的路径

用法：
    python scripts/check_docker_static.py
    python scripts/check_docker_static.py --verbose

退出码：0 = 无问题；1 = 发现必然失败项。
"""

from __future__ import annotations

import argparse
import fnmatch
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = PROJECT_ROOT / "docker" / "Dockerfile"
COMPOSE = PROJECT_ROOT / "docker" / "docker-compose.yml"
DOCKERIGNORE = PROJECT_ROOT / ".dockerignore"


class Finding:
    def __init__(self, level: str, code: str, msg: str) -> None:
        self.level = level   # FAIL / WARN
        self.code = code
        self.msg = msg

    def __str__(self) -> str:
        return f"[{self.level}] {self.code}  {self.msg}"


# ---------------------------------------------------------------------------
# 极简 YAML 解析：只提取本校验关心的结构，避免引入 PyYAML 之外的假设。
# 若 PyYAML 可用则优先使用（项目已依赖它）。
# ---------------------------------------------------------------------------
def load_compose() -> dict:
    try:
        import yaml  # type: ignore
    except ImportError:
        return {}
    if not COMPOSE.exists():
        return {}
    return yaml.safe_load(COMPOSE.read_text(encoding="utf-8")) or {}


def read_dockerfile() -> str:
    return DOCKERFILE.read_text(encoding="utf-8") if DOCKERFILE.exists() else ""


def strip_comments(text: str) -> str:
    """去掉 Dockerfile 注释行。

    必须去注释：注释里常出现**示例代码**（如
    `# 那里用 uvicorn.run("aicg.api.app:create_app", factory=True)`），
    若参与解析会被当成真实入口，产生"CMD 缺 --factory"的误报
    （实测踩过：加了这条注释后门禁立刻误报）。
    """
    out = []
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("#"):
            continue
        out.append(line)
    return "\n".join(out)


# ---------------------------------------------------------------------------
# A. COPY/ADD 源路径存在性
# ---------------------------------------------------------------------------
def check_copy_sources(df: str, findings: list[Finding]) -> None:
    for m in re.finditer(r"^\s*(COPY|ADD)\s+(.+)$", df, re.MULTILINE):
        raw = m.group(2).strip()
        if raw.startswith("--from="):          # 多阶段产物，非上下文路径
            continue
        parts = raw.split()
        # 去掉 COPY 的 --flags
        parts = [p for p in parts if not p.startswith("--")]
        srcs, dst = parts[:-1], parts[-1]
        for s in srcs:
            if "$" in s:
                continue
            # 上下文根 = 项目根（compose 里 context: ..）
            cand = PROJECT_ROOT / s.rstrip("/")
            if not cand.exists():
                findings.append(Finding(
                    "FAIL", "DOCKER-COPY-001",
                    f"Dockerfile COPY 源路径不存在：`{s}`（解析为 {cand}）",
                ))


# ---------------------------------------------------------------------------
# B. uvicorn 入口与模块导出的一致性
# ---------------------------------------------------------------------------
def check_uvicorn_entry(df: str, findings: list[Finding]) -> None:
    """`CMD [...]` 或 `ENTRYPOINT [...]` 里的 uvicorn 目标必须真实存在。

    注意：CMD 常常跨多行（用反斜杠续行），或写成 shell 形式（无方括号）。
    因此不能只抓 `[...]` —— 还要处理续行与普通 CMD 行，否则会漏判
    `--factory` 是否存在（实测踩过：CMD 里明明有 --factory 却报缺失）。
    """
    # 先去掉注释行（注释里的示例代码会被误判为真实入口），
    # 再做续行拼接，最后匹配。
    df = strip_comments(df)
    folded = re.sub(r"\\\s*\n\s*", " ", df)
    candidates: list[str] = []
    # JSON 形式："CMD ["uvicorn", ...]"
    for m in re.finditer(r'(?:CMD|ENTRYPOINT)\s*\[(.*?)\]', folded):
        candidates.append(m.group(1))
    # shell 形式："CMD uvicorn ..."（无方括号）
    for m in re.finditer(r'^\s*(?:CMD|ENTRYPOINT)\s+(?!\[)(.+)$', folded, re.MULTILINE):
        candidates.append(m.group(1))
    # Dockerfile 里内联的 uvicorn.run(...)
    for m in re.finditer(r'uvicorn\.run\(\s*[\'"]([\w.:]+)[\'"]([^)]*)', folded):
        candidates.append(m.group(1) + " " + m.group(2))

    for blob in candidates:
        if ":" not in blob:
            continue
        for target in re.findall(r'[\'"]?([\w.]+:[\w]+)[\'"]?', blob):
            module_name, _, attr = target.partition(":")
            mod_path = PROJECT_ROOT / "src" / (module_name.replace(".", "/") + ".py")
            if not mod_path.exists():
                findings.append(Finding(
                    "FAIL", "DOCKER-ENTRY-001",
                    f"uvicorn 入口模块不存在：`{module_name}`（{mod_path}）",
                ))
                continue
            text = mod_path.read_text(encoding="utf-8")
            has_module_attr = bool(re.search(
                rf"^(?:{attr}\s*=|def\s+{attr}\s*\(|class\s+{attr}\b)",
                text, re.MULTILINE))
            is_factory = f"def {attr}(" in text
            has_flag = "--factory" in blob
            if not has_module_attr:
                findings.append(Finding(
                    "FAIL", "DOCKER-ENTRY-002",
                    f"`{target}` 在 {mod_path.name} 中不存在 —— 容器启动会报 "
                    f"Attribute \"{attr}\" not found",
                ))
            elif is_factory and not has_flag:
                findings.append(Finding(
                    "FAIL", "DOCKER-ENTRY-003",
                    f"`{target}` 是**工厂函数**，必须加 `--factory`（当前 CMD 未加）",
                ))
            elif not is_factory and has_flag and attr != "create_app":
                findings.append(Finding(
                    "WARN", "DOCKER-ENTRY-004",
                    f"`{target}` 看起来不是工厂，却用了 `--factory`",
                ))


# ---------------------------------------------------------------------------
# C. compose build args 与 Dockerfile ARG 匹配
# ---------------------------------------------------------------------------
def check_build_args(compose: dict, df: str, findings: list[Finding]) -> None:
    declared = set(re.findall(r"^\s*ARG\s+(\w+)", df, re.MULTILINE))
    services = (compose or {}).get("services") or {}
    for name, svc in services.items():
        if not isinstance(svc, dict):
            continue
        build = svc.get("build")
        if not isinstance(build, dict):
            continue
        for arg in (build.get("args") or {}):
            if arg not in declared:
                findings.append(Finding(
                    "WARN", "DOCKER-ARG-001",
                    f"service `{name}` 传了 build arg `{arg}`，"
                    f"但 Dockerfile 未声明该 ARG（会被静默忽略）",
                ))
    # Dockerfile 里声明但无人传的关键 arg（提示）
    for a in sorted(declared):
        passed = any(
            a in ((s.get("build") or {}).get("args") or {})
            for s in services.values() if isinstance(s, dict)
        )
        if not passed:
            findings.append(Finding(
                "WARN", "DOCKER-ARG-002",
                f"Dockerfile 声明 ARG `{a}`，但没有任何 service 传值"
                f"（将取默认值，确认是否符合预期）",
            ))


# ---------------------------------------------------------------------------
# D. compose command 所需模块在镜像依赖中是否存在
# ---------------------------------------------------------------------------
# 命令可执行名 → 需要安装的 pip 包
# 注意：**每个包只保留一条规则**。曾经同时写 'pytest' 与 'python -m pytest'，
# 导致同一条命令命中两次、报告出现重复项（实测踩过）。
# 用最宽松的子串（'pytest'）即可覆盖其所有调用形式。
CMD_REQUIREMENTS = {
    "pytest": "pytest",
    "uvicorn": "uvicorn",
}


def check_command_deps(compose: dict, df: str, findings: list[Finding]) -> None:
    """compose 里 command 调用的工具，必须**在无条件路径上**被安装。

    关键：不能只 grep 包名是否出现在 Dockerfile 里 —— 它可能只出现在
    注释里，或出现在一个**条件 RUN**（如 `if [ "$WITH_TEST" = "1" ]`）中，
    而对应 build arg 传的是 0，此时容器里其实没有这个包。实测踩过：
    仅查字符串导致 `WITH_TEST: "0"` 的配置被漏判。
    """
    body = strip_comments(df)   # ← 去掉注释，避免注释里的包名蒙混过关
    services = (compose or {}).get("services") or {}

    for name, svc in services.items():
        if not isinstance(svc, dict):
            continue
        cmd = svc.get("command")
        if not cmd:
            continue
        if isinstance(cmd, list):
            cmd = " ".join(cmd)
        build_args = (svc.get("build") or {}).get("args") or {}

        for trigger, pkg in CMD_REQUIREMENTS.items():
            if trigger not in cmd:
                continue
            if pkg not in body:
                findings.append(Finding(
                    "FAIL", "DOCKER-CMD-001",
                    f"service `{name}` 的 command 调用了 `{trigger}`，"
                    f"但 Dockerfile 未安装 `{pkg}` —— 容器内会 No module named {pkg}",
                ))
                continue
            # 进一步：该包是否只被"条件 RUN"安装，而对应 ARG 未开启？
            guard = _conditional_guard_for(body, pkg)
            if guard is None:
                continue  # 无条件安装，安全
            arg_name, arg_on_value = guard
            passed = str(build_args.get(arg_name, "")).strip()
            if passed != arg_on_value:
                findings.append(Finding(
                    "FAIL", "DOCKER-CMD-002",
                    f"service `{name}` 需要 `{pkg}`，但它只在 "
                    f"`{arg_name}={arg_on_value}` 时才安装；"
                    f"当前传入 `{arg_name}={passed or '(未传)'}` —— "
                    f"容器内会 No module named {pkg}",
                ))


def _conditional_guard_for(body: str, pkg: str) -> tuple[str, str] | None:
    """判断 `pkg` 是否只在一个 `if [ "$ARG" = "V" ]` 块内被安装。

    返回 (ARG, V)；若为无条件安装则返回 None。

    实现方式：逐行扫描，维护"当前是否处于条件块内"的状态，
    找到含 pkg 的 pip install 行时看它处于哪个条件块。
    """
    cond_re = re.compile(r'if\s*\[\s*"\$(\w+)"\s*=\s*"([^"]*)"\s*\]')
    current: tuple[str, str] | None = None
    for line in body.splitlines():
        m = cond_re.search(line)
        if m:
            current = (m.group(1), m.group(2))
            # 单行 if（含 fi 或 &&）视为该行内生效，下一行恢复
            if "fi" in line or line.rstrip().endswith(";"):
                if "pip install" not in line:
                    current = None
            continue
        if current is not None:
            if re.match(r"^\s*fi\s*;?\s*$", line):
                current = None
                continue
            if "pip install" in line and pkg in line:
                return current
            if "pip install" in line and pkg not in line:
                continue
        elif "pip install" in line and pkg in line:
            return None   # 找到无条件安装
    return None


# ---------------------------------------------------------------------------
# E. volume 挂载：源存在 + 不遮蔽镜像内已 COPY 的关键目录
# ---------------------------------------------------------------------------
def check_volumes(compose: dict, df: str, findings: list[Finding]) -> None:
    copied_dirs = {
        m.group(2).strip("/").split("/")[0]
        for m in re.finditer(r"^\s*COPY\s+(?:--\S+\s+)*(\S+)\s+\./(\w+)", df, re.MULTILINE)
    }
    copied_dirs |= {
        m.group(1).rstrip("/")
        for m in re.finditer(r"COPY\s+(?:--\S+\s+)*(\w+)/\s+/app/\1/?", df)
    }
    services = (compose or {}).get("services") or {}
    for name, svc in services.items():
        if not isinstance(svc, dict):
            continue
        for vol in (svc.get("volumes") or []):
            if not isinstance(vol, str) or ":" not in vol:
                continue
            src, dst = vol.split(":")[:2]
            if not src.startswith((".", "/")):
                continue  # named volume
            real_src = (PROJECT_ROOT / "docker" / src).resolve()
            if not real_src.exists():
                findings.append(Finding(
                    "WARN", "DOCKER-VOL-001",
                    f"service `{name}` 挂载源不存在：`{src}`（{real_src}）",
                ))
            # 遮蔽检查：挂到 /app/<dir> 而这些目录来自 COPY
            m = re.match(r"/app/(\w+)", dst)
            if m and m.group(1) in copied_dirs:
                findings.append(Finding(
                    "WARN", "DOCKER-VOL-002",
                    f"service `{name}` 把 `{dst}` 挂载覆盖了镜像内已 COPY 的 "
                    f"`{m.group(1)}/` —— 测的/跑的可能不是镜像里的那一份",
                ))


# ---------------------------------------------------------------------------
# F. .dockerignore 存在性 + 是否误排 Dockerfile 需要的东西
# ---------------------------------------------------------------------------
def check_dockerignore(df: str, findings: list[Finding]) -> None:
    if not DOCKERIGNORE.exists():
        findings.append(Finding(
            "FAIL", "DOCKER-IGN-001",
            "缺少 `./.dockerignore`：整个项目目录（含 outputs/assets/models）"
            "都会作为构建上下文传给 daemon（本项目实测 114MB）",
        ))
        return
    rules = [
        ln.strip() for ln in DOCKERIGNORE.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    # Dockerfile 需要的最小集合
    needed = ["docker/Dockerfile", "src/", "configs/", "scripts/", "tests/",
              "requirements.txt", "pyproject.toml", "README.md"]
    for need in needed:
        key = need.rstrip("/")
        for r in rules:
            if r.startswith("!"):
                continue
            pat = r.rstrip("/")
            if not pat:
                continue
            # 用 fnmatch 语义判断，而不是朴素前缀匹配 ——
            # 否则 `configs/*.local.yaml` 会被误判为排除了整个 `configs/`
            # （它只匹配 configs 下的 local 覆盖文件）。
            hits_dir = (
                fnmatch.fnmatch(key, pat)                       # 精确/通配匹配
                or fnmatch.fnmatch(key + "/", pat + "/")
                or (pat.count("/") == 0 and key.split("/")[0] == pat)  # 顶层目录
            )
            if hits_dir:
                findings.append(Finding(
                    "FAIL", "DOCKER-IGN-002",
                    f"`.dockerignore` 规则 `{r}` 会排除 Dockerfile 需要的 "
                    f"`{need}` —— 构建时会找不到它",
                ))
                break


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Docker 编排静态校验（不构建镜像）")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    if not DOCKERFILE.exists():
        print(f"找不到 {DOCKERFILE}", file=sys.stderr)
        return 1

    df = read_dockerfile()
    compose = load_compose()
    findings: list[Finding] = []

    check_copy_sources(df, findings)
    check_uvicorn_entry(df, findings)
    check_build_args(compose, df, findings)
    check_command_deps(compose, df, findings)
    check_volumes(compose, df, findings)
    check_dockerignore(df, findings)

    if not compose:
        findings.append(Finding(
            "WARN", "DOCKER-YAML-001",
            "未能解析 docker-compose.yml（PyYAML 缺失或文件异常），"
            "compose 相关检查已跳过",
        ))

    fails = [f for f in findings if f.level == "FAIL"]
    warns = [f for f in findings if f.level == "WARN"]

    if args.verbose or fails or warns:
        print("Docker 编排静态校验")
        print(f"  目标: {DOCKERFILE.relative_to(PROJECT_ROOT)}")
        print(f"  条目: {len(re.findall(r'^FROM ', df, re.MULTILINE))} 个构建阶段；"
              f"{(compose or {}).get('services') and len(compose['services']) or 0} 个 service")
        print()
        for f in fails + warns:
            print("  " + str(f))

    if fails:
        print(f"\n合计 {len(fails)} 个必然失败项、{len(warns)} 个警告（退出码 1）")
        return 1
    if not (args.verbose or warns):
        print("Docker 编排静态校验：通过（退出码 0）")
    else:
        print(f"\n无必然失败项；{len(warns)} 个警告（退出码 0）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
