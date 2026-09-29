"""语言层 LLM 客户端：按角色选模型、按链路降级、绝不静默失败。

对应需求：FR-07 / FR-08 / NFR-E3
对应文档：《技术方案.md》§2.4、§2.4.1

**与既有 ``vlm_client.py`` 的关系**

``vlm_client.py`` 是 M3 阶段的**最小可用实现**（mock 模板兜底）。
本模块是 M4 的**真实接入层**：它会真的去调 keypool，并在失败时
按预设链路逐个换模型，最后才退到模板。

**三条设计纪律（沿用项目既有原则）**

1. **降级不抛异常**：任何网络/鉴权/超时问题都转成
   ``LLMResult(ok=False, degraded=True, reason=...)``，
   由调用方决定是否用模板兜底。绝不把异常抛到每帧循环里。
2. **切换要留痕**：每次换模型都记录到 ``attempts``，
   写明"哪个模型、什么错、耗时多少"。否则线上会以为
   "一直是 glm-5.3 在答"，实际早降级了。
3. **成本要可算**：返回 ``usage``，让上层能累计 token 成本（NFR-E3）。
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from ..observability import get_logger
from .credentials import CredentialError, KeyPoolConfig, load_config, load_from_env
from .model_registry import (
    DEFAULT_CHAIN,
    MODEL_CHAIN_FALLBACK,
    budget_for,
    by_id,
)

log = get_logger("aicg.language.client")


def _join(base: str, path: str) -> str:
    """拼接 base 与 path，避免 base 已含 /v1 时重复。"""
    b = base.rstrip("/")
    p = path if path.startswith("/") else "/" + path
    if b.endswith("/v1") and p.startswith("/v1/"):
        p = p[len("/v1"):]
    return b + p


@dataclass
class LLMResult:
    """一次语言层调用的结果。"""

    ok: bool
    text: str = ""
    model: str = ""
    """真正产出内容的模型（**降级后是降级到的那个**，不是请求的那个）。"""

    requested_model: str = ""
    degraded: bool = False
    reason: str = ""
    latency_ms: float = 0.0
    attempts: list[dict] = field(default_factory=list)
    """逐次尝试记录：{model, status, error, ms, max_tokens}。"""

    usage: dict = field(default_factory=dict)

    def describe(self) -> str:
        if self.ok:
            tag = "（已降级）" if self.degraded else ""
            return f"{self.model}{tag} {self.latency_ms:.0f}ms"
        return f"失败: {self.reason}"


class LLMClient:
    """keypool 接入的 OpenAI 兼容客户端。"""

    def __init__(self, cfg: KeyPoolConfig | None = None, timeout_s: int = 30):
        self.cfg = cfg
        self.timeout_s = timeout_s
        self._init_error: str = ""

        if self.cfg is None:
            try:
                self.cfg = load_from_env() or load_config()
            except CredentialError as e:
                self._init_error = str(e)
                log.warning(f"凭据加载失败，语言层将只能走模板兜底：{e}")

    @property
    def available(self) -> bool:
        return self.cfg is not None and self.cfg.available

    def status(self) -> str:
        if self._init_error:
            return f"不可用（{self._init_error.splitlines()[0]}）"
        if self.cfg is None:
            return "未配置"
        return self.cfg.describe()

    # ------------------------------------------------------------------
    def chat(
        self,
        messages: list[dict],
        role: str = "narration",
        model: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        json_mode: bool = False,
        min_chars: int = 8,
    ) -> LLMResult:
        """按角色调用，失败时沿链路降级。

        Args:
            messages: OpenAI 格式消息列表。
            role: 角色（narration / analysis / bulk），决定候选链路。
            model: 强制指定模型（跳过链路）。
            json_mode: 要求结构化输出（部分模型支持）。
            min_chars: 正文最小长度门槛（汉字数语义上按字符计）。
                低于此长度视为**截断失败**而非有效回答（见 D-09）——
                实测重推理模型在困难样本上会吐出单个字就耗尽预算。
                传 0 可关闭该校验（用于结构化输出等场景）。
        """
        if not self.available:
            return LLMResult(
                ok=False, degraded=True, requested_model=model or "",
                reason=self._init_error or "未配置可用的 LLM 端点",
            )

        chain = (model,) if model else DEFAULT_CHAIN.get(role, MODEL_CHAIN_FALLBACK)
        attempts: list[dict] = []
        t_all = time.perf_counter()

        for m in chain:
            spec = by_id(m)
            if spec is not None and not spec.probe_ok:
                attempts.append({
                    "model": m, "status": 0, "ms": 0.0,
                    "error": "已实测不可用（登记于 model_registry），跳过",
                })
                continue

            # D-08 防御：把预算夹到该模型的实测甜区。
            # 传太小 -> 正文被 reasoning 挤空；传太大 -> 延迟超线性上涨。
            mt = budget_for(m, max_tokens)

            # D-09 防御：截断是**随机**的（同一提示词同一参数，10 次里可能
            # 有 2~3 次推理停不下来）。因此对"截断"这一个失败模式**原地重试
            # 一次**再换模型——换模型要付一次完整延迟，重试通常更划算。
            # 只对截断重试；4xx/5xx 类失败重试无意义，直接换模型。
            got = None
            for try_i in range(2):
                t0 = time.perf_counter()
                status, body = self._post(
                    "/v1/chat/completions",
                    {
                        "model": m,
                        "messages": messages,
                        "max_tokens": mt,
                        "temperature": temperature,
                        **({"response_format": {"type": "json_object"}} if json_mode else {}),
                    },
                )
                ms = (time.perf_counter() - t0) * 1000.0

                if status != 200:
                    err = self._extract_error(body)
                    attempts.append({"model": m, "status": status, "ms": round(ms, 1),
                                     "max_tokens": mt, "error": err})
                    log.info(f"语言层 {m} 失败(status={status})，尝试链路上的下一个模型")
                    got = None
                    break

                try:
                    data = json.loads(body)
                    text = data["choices"][0]["message"]["content"]
                except (json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
                    attempts.append({"model": m, "status": status, "ms": round(ms, 1),
                                     "max_tokens": mt, "error": f"响应解析失败: {e}"})
                    got = None
                    break

                usage = data.get("usage", {}) or {}
                finish = ""
                try:
                    finish = data["choices"][0].get("finish_reason") or ""
                except (KeyError, IndexError, TypeError):
                    pass

                # D-09：正文**非空但严重截断**（实测 glm-5.3 在困难样本上
                # 返回单个"人"字，finish_reason=length，2048 token 全烧在
                # reasoning）。只判空会漏掉它，因此设一个最小长度门槛。
                stripped = (text or "").strip()
                if len(stripped) < min_chars:
                    rt = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
                    note = (
                        f"正文过短/为空（{len(stripped)}字 < 门槛{min_chars}字；"
                        f"max_tokens={mt}, completion_tokens="
                        f"{usage.get('completion_tokens')}, reasoning_tokens={rt}, "
                        f"finish_reason={finish or 'n/a'}）"
                        + ("；预算被推理耗尽，应调大该模型 max_tokens"
                           if finish == "length" else "")
                    )
                    attempts.append({"model": m, "status": status, "ms": round(ms, 1),
                                     "max_tokens": mt, "error": note})
                    if try_i == 0:
                        log.info(f"语言层 {m} 输出截断，原地重试一次（截断是随机的）")
                        continue
                    log.warning(f"语言层 {m} 重试后仍截断；尝试链路上的下一个模型")
                    got = None
                    break

                attempts.append({"model": m, "status": 200, "ms": round(ms, 1),
                                 "max_tokens": mt, "error": ""})
                got = (text, usage)
                break

            if got is None:
                if status in (401, 403):
                    # 鉴权问题换模型也没用，但不同池的密钥不同，仍值得继续
                    continue
                continue

            text, usage = got
            if len(attempts) > 1:
                log.warning(
                    f"语言层已降级：首选 {chain[0]} 不可用，"
                    f"实际使用 {m}（尝试 {len(attempts)} 次）"
                )
            return LLMResult(
                ok=True, text=text, model=m,
                requested_model=chain[0],
                degraded=len(attempts) > 1,
                reason="" if len(attempts) == 1 else f"首选 {chain[0]} 失败，降级至 {m}",
                latency_ms=(time.perf_counter() - t_all) * 1000.0,
                attempts=attempts,
                usage=usage,
            )

        return LLMResult(
            ok=False, degraded=True, requested_model=chain[0],
            reason=f"链路 {list(chain)} 全部失败；最后错误: "
                   f"{attempts[-1]['error'] if attempts else '无可用模型'}",
            latency_ms=(time.perf_counter() - t_all) * 1000.0,
            attempts=attempts,
        )

    # ------------------------------------------------------------------
    def _post(self, path: str, payload: dict) -> tuple[int, str]:
        assert self.cfg is not None
        req = urllib.request.Request(
            _join(self.cfg.base_url, path),
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "content-type": "application/json",
                "authorization": f"Bearer {self.cfg.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
                return r.status, r.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode("utf-8", "replace")
        except Exception as e:  # noqa: BLE001
            return 0, f"{type(e).__name__}: {e}"

    @staticmethod
    def _extract_error(body: str) -> str:
        try:
            d = json.loads(body)
            if isinstance(d, dict):
                err = d.get("error")
                if isinstance(err, dict):
                    return str(err.get("message", err))[:300]
                if err:
                    return str(err)[:300]
        except Exception:  # noqa: BLE001
            pass
        return body[:200]
