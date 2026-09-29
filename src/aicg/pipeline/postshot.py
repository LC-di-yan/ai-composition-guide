"""拍后解说流程（FR-07 / FR-08）。

对应需求：FR-07（自然语言解说）、FR-08（滤镜推荐）
对应文档：《数据模型与接口.md》§2.7 ``ShotReport``

**这是语言层唯一的入口**，也是"语言不进实时回路"这条设计约束的落点:

- 实时回路（``guiding_loop``）= 感知 + 决策 + 防抖，**微秒~毫秒级**；
- 拍后解说（本模块）= VLM 调用 + 滤镜推荐，**数百毫秒~数秒级**。

两者物理隔离，因此 VLM 再慢也不会污染 NFR-P1 的延迟指标。

**可溯源（NFR-O2）**：``ShotReport.input_snapshot`` 保留生成解说所依据的
完整快照，"解说里的每句话"都能回溯到具体字段。
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass

import numpy as np

from ..language import VlmClient, build_vlm_client
from ..language.tts import BaseTtsEngine, build_tts_engine
from ..observability import get_logger
from ..postprocess.filter_recommend import FilterRecommender, build_filter_recommender
from ..schemas import FrameSnapshot, ShotReport
from ..settings import Settings, get_settings
from ..utils.image import encode_png_base64

log = get_logger("pipeline.postshot")


@dataclass
class PostShotOutcome:
    """拍后产出。"""

    report: ShotReport
    language_ms: float
    """语言层耗时（ms）—— **不计入实时延迟指标**，单独上报。"""


class PostShotPipeline:
    """拍后解说与滤镜推荐流程。"""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        vlm: VlmClient | None = None,
        recommender: FilterRecommender | None = None,
        tts: BaseTtsEngine | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.vlm = vlm or build_vlm_client(self.settings.language)
        self.recommender = recommender or build_filter_recommender(
            self.vlm.assets.filters
        )
        self.tts = tts or build_tts_engine(enabled=False)

    # ------------------------------------------------------------------
    def generate(
        self,
        snapshot: FrameSnapshot,
        image: np.ndarray | None = None,
        *,
        with_image: bool = False,
        speak: bool = False,
    ) -> PostShotOutcome:
        """基于快照生成拍后报告。

        Args:
            snapshot: 该次拍摄的核心快照。**解说事实的唯一来源**。
            image: 可选原始帧（BGR）。提供且 ``with_image=True`` 时走
                多模态调用（成本更高但解说更"懂画面"）。
            with_image: 是否把图像送入 VLM。
            speak: 是否请求 TTS 播报（M3 阶段为占位，不产生音频）。

        Returns:
            拍后产出。**保证返回**——VLM 失败时走模板兜底。
        """
        t0 = time.perf_counter()

        image_b64: str | None = None
        if with_image and image is not None:
            try:
                image_b64 = encode_png_base64(image)
            except Exception as e:  # noqa: BLE001
                log.warning("图像编码失败，退回纯文本调用: %s", e)

        narration = self.vlm.narrate(snapshot, image_b64=image_b64)
        recommendation = self.recommender.recommend(snapshot)

        audio_ref = None
        if speak:
            text = f"{narration.text}。建议使用{recommendation.name}滤镜。"
            audio_ref = self.tts.speak(text, frame_id=snapshot.frame_id)

        language_ms = (time.perf_counter() - t0) * 1000.0

        report = ShotReport(
            shot_id=f"shot-{snapshot.frame_id}-{uuid.uuid4().hex[:6]}",
            composition_narration=narration.text,
            filter_name=recommendation.name,
            filter_reason=recommendation.reason,
            input_snapshot=snapshot,
            prompt_version=narration.prompt_version,
            vlm_model=narration.model,
            is_fallback=narration.is_fallback,
            token_usage=narration.token_usage,
            tts_audio_ref=audio_ref,
        )

        log.info(
            "拍后报告生成: shot=%s 滤镜=%s 解说=%.0f字 兜底=%s 耗时=%.0fms",
            report.shot_id,
            report.filter_name,
            len(report.composition_narration),
            report.is_fallback,
            language_ms,
        )
        return PostShotOutcome(report=report, language_ms=language_ms)

    # ------------------------------------------------------------------
    @property
    def vlm_available(self) -> bool:
        """真实 VLM 是否可用（供接口层告知前端）。"""
        return self.vlm.enabled and not self.vlm.budget_exhausted()


def build_postshot(settings: Settings | None = None) -> PostShotPipeline:
    """按配置构建拍后流程。"""
    return PostShotPipeline(settings or get_settings())
