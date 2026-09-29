"""编排层：把五层串成"引导回路"与"拍后流程"。

对应需求：FR-05（实时引导循环）、FR-07（拍后解说）、FR-01/02（校准与主体）
对应文档：《技术方案.md》§1 五层架构、《目录结构.md》分层规则

本层是**唯一允许跨层调用**的地方（见《目录结构.md》分层依赖规则）：
底层各层之间禁止横向依赖，所有"跨层组装"都收敛到本包。

模块分工::

    frame_processor.py  单帧五层编排（无会话状态）
    guiding_loop.py     多帧会话循环 + 指标采集 + 防抖对照实验
    calibration.py      FR-01 距离校准流程
    subject.py          FR-02 主体确认流程
    postshot.py         FR-07/08 拍后解说与滤镜推荐（语言层入口）

**热路径与冷路径的物理隔离**（本项目最重要的性能约束）::

    热路径（必须在 100ms 内）: guiding_loop → frame_processor
    冷路径（允许 1~3 秒）:      postshot（VLM）、calibration（偶发）

若把 VLM 放进逐帧回路，端到端延迟会从 ~50ms 直接劣化到 ~1s 量级，
NFR-P1 立即失守。因此本包刻意把它们做成**两个独立入口**。
"""

from __future__ import annotations

from .calibration import CalibrationOutcome, CalibrationPipeline
from .frame_processor import FrameContext, FrameProcessor
from .guiding_loop import GuidingLoop, LoopResult, build_loop, compare_debounce
from .postshot import PostShotOutcome, PostShotPipeline, build_postshot
from .subject import SubjectOutcome, SubjectPipeline

__all__ = [
    "CalibrationOutcome",
    "CalibrationPipeline",
    "FrameContext",
    "FrameProcessor",
    "GuidingLoop",
    "LoopResult",
    "PostShotOutcome",
    "PostShotPipeline",
    "SubjectOutcome",
    "SubjectPipeline",
    "build_loop",
    "build_postshot",
    "compare_debounce",
]
