"""YOLO 目标检测/分割后端（真实模型实现）。

对应需求：FR-02（主体确认）
对应文档：《技术方案.md》§2.2 —— YOLOv8-seg，端侧可 INT8 量化

降级契约（NFR-R1/R2）：权重缺失、加载失败或推理异常时**不抛出**，
而是返回 ``degraded=True`` 的空结果，由上层决定是否切换到规则后端。
"""

from __future__ import annotations

import time

import numpy as np

from ..observability import get_logger
from ..schemas import PerceptionResult, Subject, SubjectSource
from ..utils.image import bbox_px_to_norm
from .base import BasePerception, SubjectSelector

log = get_logger("perception.yolo")

_WARMUP_ITERATIONS = 3
"""预热推理次数。

取 3 的依据（实测，480x270 输入）：预热 1 次后首次真实推理仍需 91ms，
3 次后降到 21ms。再多收益递减（第 4 次起稳定在 16~21ms），
故取 3 作为"足以触发 cuDNN autotune 且不浪费启动时间"的折中。
"""


class YoloPerception(BasePerception):
    """基于 Ultralytics YOLO 的主体检测后端。

    注意：本类不做内部重试；单帧失败即返回降级结果，保证引导循环
    的时间预算可控（NFR-P1）。
    """

    name = "yolo"

    def __init__(
        self,
        weights: str,
        device: str = "auto",
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        subject_labels: list[str] | None = None,
        prefer_person: bool = True,
    ) -> None:
        self.weights = str(weights)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.subject_labels = set(subject_labels or ["person"])
        self.selector = SubjectSelector(prefer_person=prefer_person)

        self._model = None
        self._device = self._resolve_device(device)
        self._load_error: str | None = None
        self._load()

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    def _resolve_device(self, device: str) -> str:
        if device != "auto":
            return device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def _load(self) -> None:
        """加载权重。**在构造时执行**（急于加载，而非惰性）。

        为什么选择急于加载：
            权重加载耗时实测约 83ms（CUDA）。若推迟到首帧，这 83ms 会
            直接叠加进首帧端到端延迟，使 P95 指标出现尖刺（NFR-P1 要求
            稳定而非"平均达标"）。放在构造期则完全落在启动阶段，
            引导循环的每一帧都在同一时间预算内。

        失败处理：只记录不抛出。上层通过 :attr:`available` 判断是否
        需要降级为规则后端（NFR-R1）。
        """
        try:
            from ultralytics import YOLO

            t0 = time.perf_counter()
            self._model = YOLO(self.weights)
            log.info(
                "YOLO 权重加载完成: %s (device=%s, %.0fms)",
                self.weights,
                self._device,
                (time.perf_counter() - t0) * 1000,
            )
        except Exception as e:  # noqa: BLE001 - 加载失败必须降级而非中断
            self._load_error = f"{type(e).__name__}: {e}"
            self._model = None
            log.warning("YOLO 权重加载失败，将降级为规则后端: %s", self._load_error)

    @property
    def available(self) -> bool:
        """权重是否成功加载。"""
        return self._model is not None

    def model_versions(self) -> dict[str, str]:
        info = {"yolo.weights": self.weights, "yolo.device": self._device}
        try:
            import ultralytics

            info["yolo.version"] = ultralytics.__version__
        except Exception:
            pass
        return info

    def warmup(self, image_shape: tuple[int, int] | None = None) -> None:
        """跑一次空推理，触发 CUDA 上下文与算子编译，避免首帧延迟尖峰。

        Args:
            image_shape: 实际推理尺寸 ``(高, 宽)``。**务必传真实尺寸**。

        实测数据（RTX 4050, yolov8n-seg, 480x270）：

        ==================================================== ==========
        场景                                                  首次推理
        ==================================================== ==========
        完全不预热                                            7099 ms
        预热但参数与 infer 不一致（少传 iou/retina_masks）      91~120 ms
        预热且参数与 infer 完全一致                            21~34 ms
        ==================================================== ==========

        **关键教训**：预热必须用与真实推理**完全相同的参数**。Ultralytics
        会根据 ``iou`` / ``retina_masks`` 等参数走进不同分支（是否额外做
        NMS、是否生成分割掩膜），参数不一致时预热的根本是另一条路径，
        真实推理仍要重新规划。这个坑很隐蔽——预热"看起来跑了"，
        但指标依然虚高，容易误判为"模型就是这么慢"。

        残余的第一次开销来自 cuDNN 卷积算法在首次遇到该 shape 时的
        基准测试（autotune）；CUDA 是异步的，故预热后显式同步一次。
        """
        if self._model is None:
            return
        try:
            h, w = image_shape if image_shape else (480, 640)
            dummy = np.zeros((int(h), int(w), 3), dtype=np.uint8)
            # 参数必须与 infer() 一字不差，否则预热的不是同一条路径
            for _ in range(_WARMUP_ITERATIONS):
                self._model.predict(
                    dummy,
                    verbose=False,
                    device=self._device,
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    retina_masks=False,
                )
            self._sync_device()
            log.debug("YOLO 预热完成 (shape=%dx%d, %d 次)", w, h, _WARMUP_ITERATIONS)
        except Exception as e:  # noqa: BLE001
            log.warning("YOLO 预热失败（不影响运行）: %s", e)

    def _sync_device(self) -> None:
        """等待 GPU 上已排队的工作真正完成。

        CUDA 调用是异步的：``predict()`` 返回不代表计算结束。不显式同步，
        预热阶段的 kernel 会"泄漏"到第一次真实推理的计时里（实测约 70ms）。
        """
        try:
            import torch

            if self._device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------
    # 推理
    # ------------------------------------------------------------------
    def infer(self, image: np.ndarray, frame_id: int, timestamp_ms: int) -> PerceptionResult:
        h, w = image.shape[:2]
        t0 = time.perf_counter()

        base = PerceptionResult(
            frame_id=frame_id,
            timestamp_ms=timestamp_ms,
            frame_size=(w, h),
            backend=self.name,
            model_versions=self.model_versions(),
        )

        if self._model is None:
            base.degraded = True
            base.extras["load_error"] = self._load_error or "unknown"
            base.perception_ms = (time.perf_counter() - t0) * 1000.0
            return base

        try:
            results = self._model.predict(
                image,
                verbose=False,
                device=self._device,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                retina_masks=False,
            )
        except Exception as e:  # noqa: BLE001 - 单帧推理异常不得中断循环
            log.warning("YOLO 推理异常（帧 %d）: %s", frame_id, e)
            base.degraded = True
            base.extras["infer_error"] = f"{type(e).__name__}: {e}"
            base.perception_ms = (time.perf_counter() - t0) * 1000.0
            return base

        subjects = self._extract_subjects(results[0], (w, h))
        base.subjects = subjects
        base.perception_ms = (time.perf_counter() - t0) * 1000.0

        if not subjects:
            # 无主体是"可预期"结果而非故障：置降级标记，交由上层走人工兜底
            base.degraded = True
            base.extras["no_subject"] = True
        return base

    def _extract_subjects(self, result, size: tuple[int, int]) -> list[Subject]:
        """把 ultralytics 结果转换为契约对象，并完成主体选定。"""
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        names = getattr(result, "names", {}) or {}
        raw: list[tuple[str, tuple[float, float, float, float], float]] = []

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)

        for (x1, y1, x2, y2), conf, cls_id in zip(xyxy, confs, clss):
            label = str(names.get(int(cls_id), str(cls_id)))
            # 只保留关注类别，避免背景物体干扰构图判断
            if self.subject_labels and label not in self.subject_labels:
                continue
            bbox = bbox_px_to_norm((float(x1), float(y1), float(x2), float(y2)), size)
            raw.append((label, bbox, float(conf)))

        if not raw:
            return []

        primary_idx = self.selector.select(raw)
        subjects: list[Subject] = []
        for i, (label, bbox, conf) in enumerate(raw):
            subjects.append(
                Subject(
                    subject_id=f"s{i}",
                    label=label,
                    bbox=bbox,
                    confidence=conf,
                    is_primary=(i == primary_idx),
                    source=SubjectSource.AUTO,
                )
            )
        return subjects
