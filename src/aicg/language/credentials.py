"""外部服务凭据加载：从 keypool 目录运行时读取，**严禁硬编码**。

对应需求：M4（语言层接入真实模型）
对应文档：《技术方案.md》§2.4 语言层选型

**设计原则**

1. **凭据永不入库**。本模块只负责"从用户指定的本地目录读取"，
   密钥不写进代码、不写进 `configs/`、不进版本库。
2. **只读不写**。绝不修改 keypool 目录下的任何文件。
3. **失败要明确**。目录不存在 / 文件缺失 / 无法解析时，
   抛出带**可执行建议**的异常，而不是静默降级成空密钥——
   静默降级会让调用方以为"已接入真实模型"，实际却在跑 mock，
   这正是本项目在 D-04 里记录过的那类缺陷。
4. **优先走本地代理**。keypool 在本机 `127.0.0.1:8790` 起了
   OpenAI 兼容代理，由它负责密钥轮询与 429 冷却。因此**默认
   不直连上游**，只用一个占位口令访问本地代理即可，
   这样密钥本身根本不需要进入本项目的进程环境。

**两种接入模式**

- ``proxy``（默认）：只连本地代理 ``http://127.0.0.1:8790/v1``。
  本进程**完全不接触真实密钥**。推荐，也是 M4 采用的方式。
- ``direct``：直连上游，需要真正读取密钥。仅在代理不可用时备用，
  且明确告知调用方"当前绕过了密钥池，无轮询与冷却保护"。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

# keypool 目录的候选位置（按顺序探测，第一个存在的用于读取）
DEFAULT_KEYPOOL_DIRS = (
    Path("C:/Users/86134/Desktop/keypool"),
    Path.home() / "Desktop" / "keypool",
)

PROXY_BASE_URL = "http://127.0.0.1:8790/v1"
PROXY_PLACEHOLDER_KEY = "keypool-local"


class CredentialError(RuntimeError):
    """凭据读取失败。消息必须包含可执行的修复建议。"""


@dataclass
class KeyPoolConfig:
    """从 keypool 目录解析出的服务配置（不含密钥值时也可用）。"""

    source_dir: Path | None = None
    mode: str = "proxy"
    base_url: str = PROXY_BASE_URL
    api_key: str = PROXY_PLACEHOLDER_KEY
    pools: list[dict] = field(default_factory=list)
    access_keys: dict[str, str] = field(default_factory=dict)
    """池名 -> 专属口令（占位示例：``tf-pool-<口令>``；真实值只存在于 keypool 目录）。"""

    @property
    def available(self) -> bool:
        return bool(self.base_url and self.api_key)

    def describe(self) -> str:
        """一句话描述当前接入状态（用于日志，**不含任何密钥**）。"""
        if self.mode == "proxy":
            return f"keypool 本地代理 {self.base_url}（密钥由代理池管理，本进程不持有）"
        n = sum(len(p.get("keys", [])) for p in self.pools)
        pools = ", ".join(p.get("name", "?") for p in self.pools) or "无"
        return f"直连上游（池: {pools}，密钥 {n} 把；无轮询/冷却保护）"


# ----------------------------------------------------------------------
def find_keypool_dir(explicit: str | Path | None = None) -> Path:
    """定位 keypool 目录。

    优先用显式传入的路径；否则依次探测默认位置。

    Raises:
        CredentialError: 所有候选位置都不存在，附带排查建议。
    """
    cands: list[Path] = []
    if explicit:
        cands.append(Path(explicit))
    cands.extend(DEFAULT_KEYPOOL_DIRS)

    for d in cands:
        if d.is_dir():
            return d

    tried = "\n".join(f"  - {c}" for c in cands)
    raise CredentialError(
        "未找到 keypool 目录。\n"
        f"已尝试的位置：\n{tried}\n"
        "排查建议：\n"
        "  1) 确认 keypool 目录确实存在（默认在桌面）；\n"
        "  2) 若在别处，用 --keypool-dir 显式指定，或设置环境变量 AICG_KEYPOOL_DIR；\n"
        "  3) 若不需要接入真实模型，可显式使用 provider=mock（模板兜底，不报错）。"
    )


def load_config(explicit_dir: str | Path | None = None, mode: str = "proxy") -> KeyPoolConfig:
    """读取 keypool 配置。

    Args:
        explicit_dir: 显式指定的 keypool 目录。
        mode: ``proxy``（默认，推荐）或 ``direct``。

    Returns:
        :class:`KeyPoolConfig`。``proxy`` 模式下不含任何真实密钥。

    Raises:
        CredentialError: 目录/文件缺失或解析失败。
    """
    if mode not in ("proxy", "direct"):
        raise CredentialError(f"未知 mode: {mode!r}，只能是 'proxy' 或 'direct'")

    d = find_keypool_dir(explicit_dir)
    cfg_path = d / "config.json"

    if mode == "proxy":
        # 代理模式：连本地代理即可，不读取任何密钥文件
        return KeyPoolConfig(
            source_dir=d, mode="proxy",
            base_url=PROXY_BASE_URL, api_key=PROXY_PLACEHOLDER_KEY,
            pools=_peek_pool_names(cfg_path),
        )

    # ---- direct 模式：需要真实读取密钥 ----
    if not cfg_path.exists():
        raise CredentialError(
            f"缺少 {cfg_path}。\n"
            "排查建议：确认 keypool 目录完整；或用 mode='proxy' 走本地代理。"
        )
    try:
        raw = json.loads(cfg_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise CredentialError(
            f"解析 {cfg_path} 失败: {e}\n"
            "排查建议：用 JSON 校验工具检查该文件（常见于尾随逗号、注释未闭合）。"
        ) from e

    pools_out: list[dict] = []
    access: dict[str, str] = {}
    for p in raw.get("pools", []):
        keys = list(p.get("keys") or [])
        for kf in p.get("key_files") or []:
            keys.extend(_read_key_file(d / kf, d))
        name = p.get("name", "?")
        pools_out.append({
            "name": name,
            "base_url": p.get("base_url", ""),
            "models": list(p.get("models") or []),
            "keys": keys,
        })
        for ak in p.get("access_keys") or []:
            access[name] = ak

    total = sum(len(p["keys"]) for p in pools_out)
    if total == 0:
        raise CredentialError(
            f"{cfg_path} 中未解析到任何密钥。\n"
            "排查建议：检查 pools[].keys 与 pools[].key_files；\n"
            "          密钥行应以 sk- 开头且不在 # 注释内。"
        )

    primary = pools_out[0]
    return KeyPoolConfig(
        source_dir=d,
        mode="direct",
        base_url=primary["base_url"],
        api_key=primary["keys"][0],
        pools=pools_out,
        access_keys=access,
    )


def _peek_pool_names(cfg_path: Path) -> list[dict]:
    """只读取池名与模型列表（**不读取 keys**），用于展示。"""
    if not cfg_path.exists():
        return []
    try:
        raw = json.loads(cfg_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return [
        {
            "name": p.get("name", "?"),
            "base_url": p.get("base_url", ""),
            "models": list(p.get("models") or []),
            "keys": [],  # 刻意留空：proxy 模式不持有密钥
        }
        for p in raw.get("pools", [])
    ]


_KEY_RE = re.compile(r"^\s*(?:#.*)?$|^\s*(sk-[A-Za-z0-9_\-]{8,})\s*$")


def _read_key_file(path: Path, base: Path) -> list[str]:
    """解析密钥文件。

    兼容 keypool 的记事本格式：
    - 以 ``#`` 开头 = 注释，跳过；
    - 以 ``sk-`` 开头 = 密钥，取用；
    - 其他行（标签、站点 URL）= 跳过。

    特别处理 ``账号-tierflow.txt`` 的**主/子账号规则**：
    标签含"主"字的密钥属主账号（兜底），此时**默认不启用**
    （该文件里主账号已被注释停用，保持一致语义）。
    """
    if not path.exists():
        raise CredentialError(
            f"密钥文件不存在: {path}\n"
            "排查建议：检查 config.json 中 key_files 的相对路径是否正确。"
        )

    keys: list[str] = []
    pending_is_primary = False
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        t = line.strip()
        if not t:
            continue
        if t.startswith("#"):
            continue
        m = _KEY_RE.match(t)
        if m and m.group(1):
            # 主账号默认跳过（与 keypool 的"优先子账号"策略一致）
            if not pending_is_primary:
                keys.append(m.group(1))
            pending_is_primary = False
            continue
        # 标签行：判断是否主账号
        pending_is_primary = "主" in t and "子" not in t

    # 去重且保序：源文件中确实存在重复密钥
    # （如 账号.txt 里"账号13(与12重复,自动去重)"，文件自身已注明）
    seen: set[str] = set()
    unique: list[str] = []
    for k in keys:
        if k not in seen:
            seen.add(k)
            unique.append(k)
    return unique


def load_from_env() -> KeyPoolConfig | None:
    """从环境变量读取（部署场景用，避免写死路径）。

    识别：
    - ``AICG_KEYPOOL_DIR``：keypool 目录
    - ``AICG_LLM_BASE_URL`` / ``AICG_LLM_API_KEY``：直接指定端点（最高优先级）
    """
    import os

    base = os.environ.get("AICG_LLM_BASE_URL")
    key = os.environ.get("AICG_LLM_API_KEY")
    if base and key:
        return KeyPoolConfig(source_dir=None, mode="direct", base_url=base, api_key=key)

    d = os.environ.get("AICG_KEYPOOL_DIR")
    if d:
        return load_config(d)
    return None
