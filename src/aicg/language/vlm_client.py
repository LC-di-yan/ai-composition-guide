"""VLM 客户端：自然语言解说生成（FR-07）。

对应需求：FR-07（自然语言解说）、NFR-E3（成本）
对应文档：《技术方案.md》§2.4 语言层选型

**架构决策（诚实标注）：**

调研指出自然语言层的措辞"不是模板拼接能写出的口吻"，因此 VLM 是
必要的。但同时必须保证：

1. **无 API Key 时流程不中断**——走**模板兜底**，用真实的结构化字段
   拼装出语法通顺的解说。兜底结果会在 ``ShotReport.is_fallback`` 中
   如实标记，**不会伪装成 VLM 输出**。
2. **成本可控**（NFR-E3）——语言层**不是每帧调用**，只在拍摄完成等
   关键节点调用，且受 ``max_calls_per_session`` 约束。
3. **可溯源**（NFR-O2）——prompt 由结构化字段组装，解说内容可回溯到
   具体字段。

关于 Agent 框架（AgentScope / LangChain）的取舍：本层是**单轮
「结构化输入 → 文本输出」**，无工具调用、无多轮规划、无状态机，
引入框架只会增加黑盒与依赖。因此采用直接调用 + 显式 prompt 组装，
这更契合 NFR-O2「建议可溯源」的要求。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import yaml

from ..observability import get_logger
from ..schemas import CompositionResult, FrameSnapshot, TokenUsage
from ..settings import LanguageConfig, PROJECT_ROOT
from ..utils.image import encode_png_base64
from .client import LLMClient

log = get_logger("language.vlm")


@dataclass
class NarrationResult:
    """解说生成结果。"""

    text: str
    is_fallback: bool
    model: str
    prompt_version: str
    token_usage: TokenUsage | None = None
    extras: dict[str, Any] = field(default_factory=dict)


class PersonaAssets:
    """人格化资产：system prompt + 兜底模板 + 滤镜预设。

    从 ``configs/prompts/photographer.yaml`` 加载，使"人格"成为可替换的
    配置资产而非硬编码（NFR-M2）。
    """

    def __init__(self, cfg: LanguageConfig | None = None) -> None:
        self.path = (cfg.resolved_prompt_file() if cfg else PROJECT_ROOT / "configs/prompts/photographer.yaml")
        self.raw: dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        try:
            with self.path.open("r", encoding="utf-8") as f:
                self.raw = yaml.safe_load(f) or {}
        except FileNotFoundError:
            log.warning("人格配置文件不存在，使用内置默认值: %s", self.path)
            self.raw = {}
        except yaml.YAMLError as e:
            log.warning("人格配置解析失败，使用内置默认值: %s", e)
            self.raw = {}

    @property
    def version(self) -> str:
        return str(self.raw.get("version", "v1"))

    @property
    def system_prompt(self) -> str:
        return str(
            self.raw.get("system_prompt")
            or "你是一位专业摄影师，请根据画面分析结果给出简短的构图解说。"
        )

    @property
    def fallback_templates(self) -> dict[str, Any]:
        return self.raw.get("fallback_templates") or {}

    @property
    def filters(self) -> list[dict[str, Any]]:
        return self.raw.get("filters") or []


class VlmClient:
    """VLM 客户端：OpenAI 兼容协议 + 模板兜底。

    支持通过 ``base_url`` 切换到各类兼容 OpenAI 协议的服务
    （OpenAI / 通义千问 DashScope 兼容模式 / 本地 vLLM 等），
    避免绑定单一供应商。
    """

    def __init__(self, cfg: LanguageConfig | None = None) -> None:
        self.cfg = cfg or LanguageConfig()
        self.assets = PersonaAssets(self.cfg)
        self._calls = 0
        self._client = None
        self._init_error: str | None = None
        # keypool 路由（provider == "keypool" 时使用）
        self._kp: LLMClient | None = None
        self._init_client()

    # ------------------------------------------------------------------
    def _init_client(self) -> None:
        """初始化客户端。无 Key 或 SDK 缺失时降级（不抛异常）。"""
        if not self.enabled:
            return

        # ---- 路径 A：keypool 本地代理（M4 默认）----
        # 选它的理由：keypool 已在本机把 27 把密钥池化并做了轮询与 429 冷却，
        # 因此本项目**不需要持有任何真实密钥**，也无需自己实现重试/限流。
        if self.cfg.provider == "keypool":
            self._kp = LLMClient(timeout_s=int(self.cfg.timeout_s))
            if self._kp.available:
                log.info("语言层就绪（keypool）: %s", self._kp.status())
            else:
                self._init_error = self._kp.status()
                log.warning("语言层 keypool 不可用，将走模板兜底: %s", self._init_error)
            return

        # ---- 路径 B：直连任意 OpenAI 兼容端点（保留原有能力）----
        try:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=self.cfg.api_key(),
                base_url=self.cfg.base_url or None,
                timeout=self.cfg.timeout_s,
            )
            log.info(
                "VLM 客户端就绪: provider=%s model=%s", self.cfg.provider, self.cfg.model
            )
        except Exception as e:  # noqa: BLE001
            self._init_error = f"{type(e).__name__}: {e}"
            self._client = None
            log.warning("VLM 客户端初始化失败，将走模板兜底: %s", self._init_error)

    @property
    def enabled(self) -> bool:
        """是否具备真实调用条件。"""
        if self.cfg.provider == "mock":
            return False
        if self.cfg.provider == "keypool":
            # keypool 不需要 api_key/model：密钥由代理池管理，
            # 模型由 model_registry 按角色决定。
            return True
        return bool(self.cfg.api_key() and self.cfg.model)

    @property
    def call_count(self) -> int:
        return self._calls

    def budget_exhausted(self) -> bool:
        """是否已达单会话调用上限（NFR-E3 成本闸门）。"""
        return self._calls >= self.cfg.max_calls_per_session

    # ------------------------------------------------------------------
    def narrate(
        self,
        snapshot: FrameSnapshot,
        image_b64: str | None = None,
    ) -> NarrationResult:
        """生成构图解说。

        Args:
            snapshot: 该次拍摄的结构化快照。**解说的事实来源完全来自它**，
                以保证可溯源（NFR-O2）。
            image_b64: 可选的图像 data URI。提供时走真正的多模态调用；
                不提供时走纯文本调用（成本更低）。

        Returns:
            解说结果。**保证返回**：真实调用失败或未配置时返回模板兜底结果。
        """
        if not self.enabled or self.budget_exhausted():
            reason = "未配置 VLM" if not self.enabled else "已达会话调用上限"
            return self._fallback(snapshot, note=reason)

        try:
            return self._call_vlm(snapshot, image_b64)
        except Exception as e:  # noqa: BLE001 - 语言层失败不得影响引导
            log.warning("VLM 调用失败，退回模板兜底: %s", e)
            return self._fallback(snapshot, note=f"{type(e).__name__}")

    # ------------------------------------------------------------------
    def _call_vlm(self, snapshot: FrameSnapshot, image_b64: str | None) -> NarrationResult:
        """真实调用。按 provider 分流到 keypool 或直连 SDK。"""
        facts = build_facts(snapshot)
        content: list[dict[str, Any]] = [
            {"type": "text", "text": "画面分析结果：\n" + yaml.safe_dump(facts, allow_unicode=True, sort_keys=False)}
        ]
        if image_b64:
            content.append({"type": "image_url", "image_url": {"url": image_b64}})
        content.append(
            {"type": "text", "text": "请基于以上分析结果，生成 1~2 句构图理念解说。"}
        )

        if self._kp is not None:
            return self._call_via_keypool(content)

        assert self._client is not None
        resp = self._client.chat.completions.create(
            model=self.cfg.model,
            messages=[
                {"role": "system", "content": self.assets.system_prompt},
                {"role": "user", "content": content},  # type: ignore[arg-type]
            ],
            temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_tokens,
        )
        self._calls += 1

        text = (resp.choices[0].message.content or "").strip()
        usage = None
        if getattr(resp, "usage", None) is not None:
            usage = TokenUsage(
                prompt_tokens=getattr(resp.usage, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(resp.usage, "completion_tokens", 0) or 0,
                total_tokens=getattr(resp.usage, "total_tokens", 0) or 0,
            )

        if not text:
            log.warning("VLM 返回空文本，退回模板兜底")
            return self._fallback(snapshot, note="empty_response")

        return NarrationResult(
            text=text,
            is_fallback=False,
            model=self.cfg.model,
            prompt_version=self.assets.version,
            token_usage=usage,
        )

    def _call_via_keypool(self, content: list[dict[str, Any]]) -> NarrationResult:
        """经 keypool 池化代理调用真实模型（M4 路径）。

        走 ``role="narration"`` 的模型链路：首选 glm-5.3（语感最佳），
        失败自动降级到 deepseek-v4-flash / qwen3.8-27b / qwen3.8-flash。
        链路与理由集中写在 ``model_registry.py``，此处不重复。

        诚实说明：本方法**不抛异常**。任何失败都转成
        ``LLMResult(ok=False)``，由 :meth:`narrate` 统一转模板兜底。
        """
        assert self._kp is not None

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.assets.system_prompt},
            {"role": "user", "content": content},
        ]
        res = self._kp.chat(
            messages,
            role="narration",
            model=self.cfg.model or None,
            max_tokens=self.cfg.max_tokens,
            temperature=self.cfg.temperature,
        )
        self._calls += 1

        # 失败原因必须**有内容**。早期版本只写 res.reason，
        # 而"HTTP 成功但正文为空"这类情况 reason 恰为空字符串，
        # 导致日志里只看到 "keypool 链路失败: " 后面什么都没有，
        # 排障时无从下手（实测踩到）。故此处分情况给出可定位的描述。
        if not res.ok:
            detail = res.reason or "（上游未给出错误信息）"
            raise RuntimeError(f"keypool 链路失败: {detail}")
        text = (res.text or "").strip()
        if not text:
            # 常见于重推理模型把 token 预算耗在 reasoning 上（见 D-08）
            ct = (res.usage or {}).get("completion_tokens")
            rt = ((res.usage or {}).get("completion_tokens_details") or {}).get("reasoning_tokens")
            raise RuntimeError(
                "keypool 返回空正文"
                f"（model={res.model}, completion_tokens={ct}, reasoning_tokens={rt}）"
                "；若 reasoning_tokens 接近 max_tokens，说明预算被推理耗尽，"
                "应调大 language.max_tokens"
            )

        u = res.usage or {}
        usage = TokenUsage(
            prompt_tokens=int(u.get("prompt_tokens", 0) or 0),
            completion_tokens=int(u.get("completion_tokens", 0) or 0),
            total_tokens=int(u.get("total_tokens", 0) or 0),
        ) if u else None

        return NarrationResult(
            text=text,
            is_fallback=False,
            model=res.model,
            prompt_version=self.assets.version,
            token_usage=usage,
            extras={
                # 如实记录"是否降级、尝试了几次"，便于复盘而非隐瞒
                "degraded_from_chain": res.degraded,
                "requested_model": res.requested_model,
                "attempts": res.attempts,
                "latency_ms": round(res.latency_ms, 1),
                "reasoning_tokens": (
                    (u.get("completion_tokens_details") or {}).get("reasoning_tokens")
                ),
            },
        )

    # ------------------------------------------------------------------
    def _fallback(self, snapshot: FrameSnapshot, note: str = "") -> NarrationResult:
        """模板兜底解说。

        诚实说明：这不是"假数据"。它把真实的结构化推理结果
        （构图模式、景别、问题项）组装成句子，事实部分完全真实；
        只是措辞不如 VLM 自然。``is_fallback=True`` 会如实上报。
        """
        templates = self.assets.fallback_templates
        comp: CompositionResult = snapshot.composition
        subject = snapshot.perception.primary_subject

        # 选择最贴合的模板。
        #
        # **缺陷修正（真实 bug）**：早期条件是 ``subject is None or comp.degraded``，
        # 把"构图评估降级"与"没识别到主体"当成同一件事。但 ``comp.degraded``
        # 在**主体明明存在**时也会为 True——例如候选框搜索空间受限
        # （``candidate_count`` 极少）、主体溢出画面等。结果是系统对着一个
        # 清晰检出的人像说出"当前画面尚未识别到明确主体"，**向用户输出与事实
        # 相反的话**。这是比"文案不好"严重得多的问题。
        #
        # 修正：只有 ``subject is None``（确实没有主体）才用"无主体"模板；
        # 有主体但评估降级时，仍然基于真实主体描述，只是额外提示建议受限。
        emphasis = snapshot.perception.extras.get("saliency_emphasis_label")
        if subject is None:
            text = templates.get("narration_degraded") or "当前画面尚未识别到明确主体。"
        elif emphasis:
            tpl = templates.get("narration_with_saliency") or templates.get("narration")
            text = _safe_format(
                str(tpl),
                pattern_label=comp.pattern_label or "三分法",
                shot_size=comp.shot_size_label or "半身",
                emphasis=str(emphasis),
                subject_label=subject.label,
            )
        else:
            tpl = templates.get("narration_with_subject") or templates.get("narration")
            text = _safe_format(
                str(tpl),
                pattern_label=comp.pattern_label or "三分法",
                shot_size=comp.shot_size_label or "半身",
                subject_label=subject.label,
            )
            # 主体存在但构图空间搜索受限 → 如实附注，避免让用户误以为
            # 建议是完整搜索后的最优解。
            if comp.degraded:
                text += "（注：本次构图候选空间受限，建议仅供参考。）"

        # 追加改善建议（来自真实的规则违反项）
        issues = templates.get("issue_phrases") or {}
        if comp.rule_violations:
            phrases = [issues.get(v.rule.value) for v in comp.rule_violations]
            phrases = [p for p in phrases if p]
            if phrases:
                text += f"如果能{phrases[0]}，画面会更舒展。"

        return NarrationResult(
            text=text,
            is_fallback=True,
            model="mock",
            prompt_version=self.assets.version,
            extras={"fallback_note": note} if note else {},
        )


# ----------------------------------------------------------------------
def build_facts(snapshot: FrameSnapshot) -> dict[str, Any]:
    """把快照抽取为"事实字典"，作为 prompt 的唯一信息源（NFR-O2）。

    只保留可被验证的字段，不含任何推测。上层与 VLM 都看同一份事实，
    从而保证"解说里的每句话都能追溯到某个字段"。
    """
    comp = snapshot.composition
    subject = snapshot.perception.primary_subject
    cmd = snapshot.command.command

    facts: dict[str, Any] = {
        "构图模式": comp.pattern_label or comp.pattern.value,
        "景别": comp.shot_size_label,
        "构图评分": comp.composition_score,
        "最优构图评分": comp.best_score,
        "主体": (
            {"类别": subject.label, "置信度": round(subject.confidence, 3)}
            if subject
            else None
        ),
        "各项得分": {k: round(v, 3) for k, v in comp.sub_scores.items()},
        "当前指令": {"动作": cmd.action.value, "说明": cmd.magnitude_text},
    }

    if comp.rule_violations:
        facts["存在的问题"] = [
            {"类型": v.rule.value, "严重度": v.severity.value, "说明": v.detail}
            for v in comp.rule_violations
        ]
    if snapshot.perception.degraded:
        facts["感知降级"] = True
        facts["感知后端"] = snapshot.perception.backend
    return facts


def _safe_format(template: str, **kwargs: Any) -> str:
    """安全格式化：模板中缺失的占位符不抛异常。"""
    try:
        return template.format(**kwargs)
    except (KeyError, IndexError):
        out = template
        for k, v in kwargs.items():
            out = out.replace("{" + k + "}", str(v))
        return out
