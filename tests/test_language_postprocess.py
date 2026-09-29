"""语言层与后处理层测试（FR-07 / FR-08）。

对应文档：《需求说明.md》FR-07/FR-08、NFR-E3（成本）、NFR-O2（可溯源）

**核心验证点**：

1. **无 API Key 时流程绝不中断**（降级是常态）；
2. **兜底结果如实标记** ``is_fallback=True``，不伪装成 VLM 输出；
3. **解说事实可溯源**：``build_facts`` 的每个键都能对应到快照字段；
4. **滤镜推荐必须给出理由**（NFR-O2），且理由指向具体信号。
"""

from __future__ import annotations

import pytest

from aicg.language import PersonaAssets, VlmClient, build_facts
from aicg.language.tts import NullTtsEngine, build_tts_engine
from aicg.postprocess import FilterRecommender
from aicg.schemas import (
    ActionCommand,
    ActionType,
    CompositionPattern,
    CompositionResult,
    FrameSnapshot,
    PerceptionResult,
    RuleName,
    RuleViolation,
    Severity,
    StabilizedCommand,
    Subject,
    SubjectSource,
)
from aicg.settings import LanguageConfig


def _snapshot(
    *,
    score: float = 72.0,
    best: float = 88.0,
    pattern: CompositionPattern = CompositionPattern.RULE_OF_THIRDS,
    occupancy: float = 0.68,
    violations: list | None = None,
    extras: dict | None = None,
    degraded: bool = False,
) -> FrameSnapshot:
    h = occupancy
    w = h * 0.42
    cx = 0.62
    subj = Subject(
        subject_id="s0", label="person",
        bbox=(cx - w / 2, 0.5 - h / 2, cx + w / 2, 0.5 + h / 2),
        confidence=0.92, is_primary=True, source=SubjectSource.AUTO,
    )
    raw = ActionCommand(action=ActionType.MOVE_BACK, magnitude_text="约 0.5 米", magnitude_raw=0.08)
    return FrameSnapshot(
        frame_id=7, timestamp_ms=2331, frame_size=(480, 640),
        perception=PerceptionResult(
            frame_id=7, timestamp_ms=2331, frame_size=(480, 640),
            subjects=[subj], backend="stub", extras=extras or {}, degraded=degraded,
        ),
        composition=CompositionResult(
            composition_score=score, best_score=best,
            best_bbox=(0.1, 0.05, 0.9, 0.95), current_bbox=subj.bbox,
            sub_scores={"thirds": 0.8, "balance": 0.6, "headroom": 0.5, "brightness": 0.55},
            pattern=pattern, pattern_label="右三分", shot_size_label="七分身",
            rule_violations=violations or [],
        ),
        command=StabilizedCommand(command=raw, raw_command=raw, is_changed=True),
    )


# ---------------------------------------------------------------------------
# 人格资产
# ---------------------------------------------------------------------------
class TestPersonaAssets:
    def test_loads_from_config(self):
        """人格资产应从 yaml 加载，使"人格"可替换（NFR-M2）。"""
        a = PersonaAssets()
        assert a.system_prompt
        assert isinstance(a.fallback_templates, dict)
        assert isinstance(a.filters, list)

    def test_missing_file_uses_builtin_default(self, tmp_path):
        """配置文件缺失时必须退回内置默认值，不能崩。"""
        cfg = LanguageConfig(prompt_file=str(tmp_path / "nonexistent.yaml"))
        a = PersonaAssets(cfg)
        assert a.system_prompt
        assert a.version == "v1"

    def test_malformed_yaml_falls_back(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text("system_prompt: [unclosed\n  - broken:", encoding="utf-8")
        a = PersonaAssets(LanguageConfig(prompt_file=str(bad)))
        assert a.system_prompt  # 未抛异常


# ---------------------------------------------------------------------------
# VLM 客户端：降级是第一公民
# ---------------------------------------------------------------------------
class TestVlmFallback:
    def test_mock_provider_is_disabled(self):
        c = VlmClient(LanguageConfig(provider="mock"))
        assert c.enabled is False

    def test_narrate_always_returns(self):
        """无 Key 时也必须返回结果（模板兜底），绝不抛异常。"""
        c = VlmClient(LanguageConfig(provider="mock"))
        r = c.narrate(_snapshot())
        assert r.text
        assert r.is_fallback is True
        assert r.model == "mock"

    def test_fallback_flagged_not_faked(self):
        """兜底必须如实标记，不能伪装成 VLM 输出（诚实性要求）。"""
        c = VlmClient(LanguageConfig(provider="mock"))
        r = c.narrate(_snapshot())
        assert r.is_fallback is True
        assert "fallback_note" in r.extras

    def test_budget_gate(self):
        """超过会话调用上限时应走兜底（NFR-E3 成本闸门）。"""
        cfg = LanguageConfig(provider="openai", model="gpt-x", max_calls_per_session=0)
        c = VlmClient(cfg)
        # 未配置 Key → enabled=False；手工模拟已用尽预算的场景
        c._calls = 999
        assert c.budget_exhausted() is True
        r = c.narrate(_snapshot())
        assert r.is_fallback is True

    def test_call_count_starts_zero(self):
        c = VlmClient(LanguageConfig(provider="mock"))
        assert c.call_count == 0

    def test_degraded_snapshot_template(self):
        """感知降级时兜底文案应切换到"未识别主体"模板。"""
        c = VlmClient(LanguageConfig(provider="mock"))
        snap = _snapshot(degraded=True, occupancy=0.3)
        snap = snap.model_copy(update={"perception": snap.perception.model_copy(update={"subjects": []})})
        r = c.narrate(snap)
        assert r.text
        assert r.is_fallback is True

    def test_violation_appended_to_fallback(self):
        """存在规则违反项时，兜底文案应追加改善建议。"""
        v = [RuleViolation(rule=RuleName.MARGIN_RATIO, severity=Severity.WARN, detail="头顶留白不足")]
        c = VlmClient(LanguageConfig(provider="mock"))
        r = c.narrate(_snapshot(violations=v))
        assert r.text


class TestNarrationHonestyRegression:
    """**回归测试：解说误报"没主体"**（真实修复过的 bug，勿删）。

    历史缺陷：兜底模板的条件是 ``subject is None or comp.degraded``，
    把"构图评估降级"与"没识别到主体"混为一谈。但 ``comp.degraded`` 在
    主体明明存在时也会为 True（候选框搜索空间受限、主体溢出画面等）。

    实测症状：对一张 YOLO 明确检出 ``person`` 的照片，系统输出
    「当前画面尚未识别到明确主体」——**向用户陈述与事实相反的内容**。
    这比"文案不自然"严重得多，属于事实性错误输出。
    """

    def _snap_with_subject_degraded(self):
        """构造：主体存在，但构图评估 degraded=True。

        这正是线上真实触发场景（real_photo_sample.jpg：检出 person，
        但 candidate_count=1 导致 degraded）。
        """
        snap = _snapshot(degraded=True, occupancy=0.7)
        comp = snap.composition.model_copy(update={"degraded": True})
        return snap.model_copy(update={"composition": comp})

    def test_subject_present_never_says_no_subject(self):
        c = VlmClient(LanguageConfig(provider="mock"))
        snap = self._snap_with_subject_degraded()
        # 前提校验：主体确实存在
        assert snap.perception.primary_subject is not None
        r = c.narrate(snap)
        assert "尚未识别到明确主体" not in r.text, (
            f"主体存在时不得输出『未识别到主体』，实际: {r.text}"
        )

    def test_subject_absent_does_use_no_subject_template(self):
        """反向用例：确实没有主体时才用该模板。"""
        c = VlmClient(LanguageConfig(provider="mock"))
        snap = _snapshot(degraded=True, occupancy=0.3)
        empty = snap.model_copy(
            update={"perception": snap.perception.model_copy(update={"subjects": []})}
        )
        assert empty.perception.primary_subject is None
        r = c.narrate(empty)
        assert "尚未识别到明确主体" in r.text

    def test_degraded_with_subject_adds_caveat(self):
        """主体存在但评估受限时，应附带"建议受限"的如实说明。"""
        c = VlmClient(LanguageConfig(provider="mock"))
        r = c.narrate(self._snap_with_subject_degraded())
        assert "受限" in r.text, f"应如实提示建议受限，实际: {r.text}"


# ---------------------------------------------------------------------------
# 事实抽取：可溯源（NFR-O2）
# ---------------------------------------------------------------------------
class TestBuildFacts:
    def test_core_keys_present(self):
        facts = build_facts(_snapshot())
        for k in ("构图模式", "景别", "构图评分", "最优构图评分", "主体", "当前指令"):
            assert k in facts, f"缺少可溯源字段 {k}"

    def test_values_traceable_to_snapshot(self):
        """事实字典的值必须与快照字段一致（可溯源性的直接断言）。"""
        snap = _snapshot(score=66.0, best=91.0)
        facts = build_facts(snap)
        assert facts["构图评分"] == 66.0
        assert facts["最优构图评分"] == 91.0
        assert facts["景别"] == snap.composition.shot_size_label
        assert facts["主体"]["类别"] == "person"

    def test_subject_none_when_absent(self):
        snap = _snapshot()
        snap = snap.model_copy(update={"perception": snap.perception.model_copy(update={"subjects": []})})
        facts = build_facts(snap)
        assert facts["主体"] is None

    def test_violations_included_when_present(self):
        v = [RuleViolation(rule=RuleName.SUBJECT_CUTOFF, severity=Severity.ERROR, detail="主体被裁切")]
        facts = build_facts(_snapshot(violations=v))
        assert "存在的问题" in facts
        assert facts["存在的问题"][0]["类型"] == RuleName.SUBJECT_CUTOFF.value

    def test_degraded_flagged(self):
        facts = build_facts(_snapshot(degraded=True))
        assert facts.get("感知降级") is True


# ---------------------------------------------------------------------------
# 滤镜推荐（FR-08）
# ---------------------------------------------------------------------------
class TestFilterRecommendation:
    def test_always_returns_with_reason(self):
        """推荐**必须**带理由（NFR-O2：建议可溯源）。"""
        rec = FilterRecommender().recommend(_snapshot())
        assert rec.name
        assert rec.reason
        assert 0.0 <= rec.confidence <= 1.0

    def test_signals_recorded(self):
        """推荐结果必须记录触发信号，供用户反驳（"为什么推这个"）。"""
        rec = FilterRecommender().recommend(_snapshot())
        assert "brightness" in rec.signals
        assert "subject_occupancy" in rec.signals
        assert "pattern" in rec.signals

    def test_dark_scene_picks_warm_contrast(self):
        """低亮度场景应推暖调高对比滤镜。"""
        snap = _snapshot()
        snap.composition.sub_scores["brightness"] = 0.15
        rec = FilterRecommender().recommend(snap)
        assert "暗" in rec.reason

    def test_closeup_picks_soft_tone(self):
        """近景/特写应推柔和调，避免肤色过曝。"""
        rec = FilterRecommender().recommend(_snapshot(occupancy=0.85))
        assert "近景" in rec.reason or "特写" in rec.reason

    def test_wide_shot_picks_saturated(self):
        """远景应推高饱和滤镜以强化层次。"""
        rec = FilterRecommender().recommend(_snapshot(occupancy=0.10))
        assert "远景" in rec.reason

    def test_signal_uses_height_not_area(self):
        """占比信号必须取高度占比（景别由高度决定，与宽高比无关）。"""
        from aicg.postprocess.filter_recommend import _WIDE_AREA

        rec = FilterRecommender().recommend(_snapshot(occupancy=0.20))
        h_occ = float(rec.signals["subject_occupancy"])
        # 该 helper 生成的框宽高比 0.42，面积 = h * 0.42h，必小于高度
        assert h_occ > _WIDE_AREA, f"高度占比 {h_occ} 不应被判为远景"

    def test_diagonal_pattern_considered(self):
        rec = FilterRecommender().recommend(_snapshot(pattern=CompositionPattern.DIAGONAL))
        assert "对角线" in rec.reason

    def test_custom_filters_respected(self):
        """人格资产提供的滤镜列表应被优先使用（NFR-M2 可替换）。"""
        rf = FilterRecommender([{"name": "MyLook", "tone": "standard", "scene": "general"}])
        # 取一个不触发任何特殊规则的中性场景：常规占比、常规亮度、三分法
        rec = rf.recommend(_snapshot(occupancy=0.30))
        assert rec.name == "MyLook"

    def test_custom_portrait_filter_used_for_closeup(self):
        rf = FilterRecommender([{"name": "SoftPortrait", "tone": "low_saturation", "scene": "portrait"}])
        rec = rf.recommend(_snapshot(occupancy=0.85))
        assert rec.name == "SoftPortrait"

    def test_falls_back_to_builtin_when_scene_missing(self):
        """自定义列表不含所需场景时应退回内置预设，而不是给出空名字。"""
        rf = FilterRecommender([{"name": "OnlyWide", "tone": "x", "scene": "landscape"}])
        rec = rf.recommend(_snapshot(occupancy=0.85))  # 需要 portrait 场景
        assert rec.name  # 非空

    def test_breakdown_signals_are_strings(self):
        """signals 必须是字符串（保证可 JSON 序列化）。"""
        rec = FilterRecommender().recommend(_snapshot())
        assert all(isinstance(v, str) for v in rec.signals.values())


# ---------------------------------------------------------------------------
# TTS 占位（FR-10，P2）
# ---------------------------------------------------------------------------
class TestTtsPlaceholder:
    def test_null_engine_does_not_block(self):
        """空实现必须在调用时立即返回，不阻塞主回路。"""
        t = NullTtsEngine()
        assert t.speak("测试文本", frame_id=1) is None
        assert len(t.history) == 1

    def test_build_defaults_to_null(self):
        assert isinstance(build_tts_engine(enabled=False), NullTtsEngine)

    def test_history_records_requests(self):
        """占位实现需记录请求，供测试断言而不产生音频。"""
        t = NullTtsEngine()
        for i in range(3):
            t.speak(f"第{i}句", frame_id=i)
        assert [x.frame_id for x in t.history] == [0, 1, 2]


# ---------------------------------------------------------------------------
# 拍后流程集成
# ---------------------------------------------------------------------------
class TestPostShotPipeline:
    def test_generates_complete_report(self, settings_factory):
        from aicg.pipeline import PostShotPipeline

        # 同 test_reports_fallback_status：显式用 mock，测试不得依赖网络。
        cfg = settings_factory(**{"language.provider": "mock"})
        p = PostShotPipeline(cfg)
        out = p.generate(_snapshot())

        r = out.report
        assert r.shot_id
        assert r.composition_narration
        assert r.filter_name
        assert r.filter_reason
        assert r.input_snapshot is not None, "必须保留快照以保证可溯源"
        assert out.language_ms >= 0

    def test_reports_fallback_status(self, settings_factory):
        from aicg.pipeline import PostShotPipeline

        # 显式指定 mock，而**不依赖全局默认值**。
        #
        # [2026-09-28 修订] 原写法是 ``settings_factory()`` 并注释
        # "默认 provider=mock"。当 configs/default.yaml 的默认值改为
        # keypool（M4 真实接入）后，这个测试变成了**真的去打网络**——
        # 它本意是验证"无可用模型时如实标记兜底"，却因此变得
        # 依赖外部服务、且会随机失败（真实模型偶尔能答出来）。
        # 教训：**测试必须显式声明自己的前提**，不能寄生在全局默认值上。
        cfg = settings_factory(**{"language.provider": "mock"})
        p = PostShotPipeline(cfg)
        out = p.generate(_snapshot())
        assert out.report.is_fallback is True
        assert out.report.vlm_model == "mock"
        assert p.vlm_available is False

    def test_tts_does_not_produce_audio_in_m3(self, settings_factory):
        """M3 阶段 TTS 为占位：请求播报也不应生成音频文件（诚实降级）。"""
        from aicg.pipeline import PostShotPipeline

        cfg = settings_factory(**{"language.provider": "mock"})
        p = PostShotPipeline(cfg)
        out = p.generate(_snapshot(), speak=True)
        assert out.report.tts_audio_ref is None
