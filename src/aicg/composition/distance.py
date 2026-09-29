"""距离估算。

对应需求：FR-01（距离校准）
对应文档：《数据模型与接口.md》§2.2、《开发计划.md》§6 简化处清单

核心公式（源自调研）：

.. code-block:: text

    distance ≈ (真实人体高度 H) / (框高像素 / 画面高像素) × (焦距相关常数)

**诚实标注的简化处：**
    本实现采用纯视觉的单目框占比法，**不依赖 LiDAR / ToF**（多数机型不具备）。
    因此误差在 ±20%~30% 属正常范围，这也是产品只给出"约 2.1 米"这类
    粗粒度数字的原因——**这是估算，不是测量**。

    生产环境做法：引入 Depth-Anything-V2 / MiDaS 深度图，或使用双摄/LiDAR。
    详见 ``EstimationMethod.DEPTH_MAP``。
"""

from __future__ import annotations

from ..schemas import CalibrationResult, EstimationMethod
from ..schemas.perception import BBox
from ..settings import DistanceConfig
from ..utils.image import bbox_height


class DistanceEstimator:
    """基于主体框高度占比的单目距离估算器。"""

    _MIN_TRUSTWORTHY_RATIO = 0.15
    """可信任的框高占比下界。

    低于此值（约对应 12m 以上）时，公式外推的数值已不可信，直接降级。
    取值依据：占比 0.15 ≈ 该配置下的 12.6m，此时主体高仅占画面 15%，
    标注框的 1~2px 抖动会造成结果数十厘米的跳变。
    """

    def __init__(self, cfg: DistanceConfig | None = None) -> None:
        self.cfg = cfg or DistanceConfig()

    # ------------------------------------------------------------------
    def estimate(
        self,
        subject_bbox: BBox | None,
        assumed_height_m: float | None = None,
        nominal_focal_mm: float | None = None,
    ) -> CalibrationResult:
        """估算拍摄距离并生成区间建议。

        Args:
            subject_bbox: 主体框（归一化）。None 或高度为 0 时走降级路径。
            assumed_height_m: 主体真实高度假设（m）。None 用配置默认值。
                **注意**：这是整个估算中最大的误差来源——对儿童、坐姿、
                半身入镜的场景都会显著偏低。
            nominal_focal_mm: 等效焦距（35mm 口径，mm）。None 用配置默认值。

        Returns:
            距离校准结果。**保证返回**（无法估算时 ``current_distance_m=None``
            且 ``degraded=True``）。
        """
        cfg = self.cfg
        H = assumed_height_m if assumed_height_m is not None else cfg.assumed_person_height_m
        f = nominal_focal_mm if nominal_focal_mm is not None else cfg.focal_constant

        # 主体框高度占比
        box_ratio = bbox_height(subject_bbox) if subject_bbox is not None else 0.0
        if box_ratio <= 0.02:
            # 无法估算：主体过小或不存在，仅给方向不给距离
            return CalibrationResult(
                current_distance_m=None,
                min_distance_m=cfg.min_distance_m,
                max_distance_m=cfg.max_distance_m,
                position_ratio=0.0,
                is_in_range=False,
                advice_text="看不到明确主体，请对准想拍的人",
                estimation_method=EstimationMethod.BBOX_HEIGHT_RATIO,
                error_margin=None,
                degraded=True,
            )

        # 可信上限兜底：框高占比过小时，公式外推会给出荒谬数值
        # （实测占比 0.05 → 38m）。此时主体已不足画面 1/10，估算失去意义，
        # 因此直接降级为"看不到明确主体"，而不是报一个假精度的大数。
        if box_ratio < self._MIN_TRUSTWORTHY_RATIO:
            return CalibrationResult(
                current_distance_m=None,
                min_distance_m=cfg.min_distance_m,
                max_distance_m=cfg.max_distance_m,
                position_ratio=0.0,
                is_in_range=False,
                advice_text="人物太小，请靠近一些再开始构图",
                estimation_method=EstimationMethod.BBOX_HEIGHT_RATIO,
                error_margin=None,
                degraded=True,
            )

        # 距离推导（小孔成像，用"等效视角"而非物理传感器尺寸）：
        #
        #   垂直视角 FOV_v = 2 * atan(sensor_height_mm / (2 * f_eq))
        #   主体张角 theta = box_ratio * FOV_v
        #   distance = (H / 2) / tan(theta / 2)
        #
        # 为什么不用经典的 (H * f) / image_height：
        #   f 取的是 **等效焦距**（相对 35mm 全画幅，26mm），而手机真实传感器
        #   高度仅约 6~8mm，两者不成比例——直接相除会得到 0.00m 的错误结果。
        #   用「等效焦距 + 全画幅高度」先算视角，再反推距离，量纲自洽。
        #
        # 校验：等效焦距 26mm → FOV_v ≈ 49.6°；框高占比 0.90 → 约 2.0m，
        #       与调研给出的"人体占画面 70% 高度时约在 2m 左右"一致。
        import math

        fov_v_deg = 2.0 * math.degrees(math.atan(cfg.sensor_height_mm / (2.0 * f)))
        fov_v_deg = min(fov_v_deg, 150.0)  # 防止异常参数导致视角过大
        theta = math.radians(box_ratio * fov_v_deg)
        # theta 可能超过 180°，tan 会变号；限制到 (0, 170°)
        theta = min(max(theta, 1e-4), math.radians(170.0))
        distance_m = (H / 2.0) / math.tan(theta / 2.0)

        # 误差范围：按调研给出的 ±20%~30%，取 ±25% 作为标称
        error_margin = round(distance_m * 0.25, 2)

        in_range = cfg.min_distance_m <= distance_m <= cfg.max_distance_m
        position_ratio = self._position_ratio(
            distance_m, cfg.min_distance_m, cfg.max_distance_m
        )
        advice = self._advice(distance_m, cfg, in_range, box_ratio)

        return CalibrationResult(
            current_distance_m=round(distance_m, 2),
            min_distance_m=cfg.min_distance_m,
            max_distance_m=cfg.max_distance_m,
            position_ratio=position_ratio,
            is_in_range=in_range,
            advice_text=advice,
            estimation_method=EstimationMethod.BBOX_HEIGHT_RATIO,
            error_margin=error_margin,
            degraded=False,
        )

    # ------------------------------------------------------------------
    def _position_ratio(self, d: float, lo: float, hi: float) -> float:
        """把当前距离映射到区间内的 0~1 位置，供滑条可视化（FR-01）。"""
        if hi <= lo:
            return 0.5
        return float(max(0.0, min(1.0, (d - lo) / (hi - lo))))

    def _advice(
        self, d: float, cfg: DistanceConfig, in_range: bool, box_ratio: float
    ) -> str:
        """生成面向用户的距离提示文案。

        文案风格对齐调研中观察到的产品表达：
        "距离合适 · 约 2.1 米 · 可继续后退，人物会更完整"
        """
        d_txt = f"约 {d:.1f} 米"
        if in_range:
            if d < cfg.ideal_distance_m:
                return f"距离合适 · {d_txt} · 可继续后退，人物会更完整"
            return f"距离合适 · {d_txt} · 当前距离很好，可以开始构图"
        if d < cfg.min_distance_m:
            return f"距离太近 · {d_txt} · 建议后退到 {cfg.min_distance_m:.1f} 米以外"
        return f"距离偏远 · {d_txt} · 建议靠近到 {cfg.max_distance_m:.1f} 米以内"

    # ------------------------------------------------------------------
    def meters_to_delta_text(self, delta_m: float) -> str:
        """把后退/前进的差分距离转成自然语言片段。

        用于 FR-04 的 ``magnitude_text``，例如 ``"约 1.7 米"``。
        """
        return f"约 {abs(delta_m):.1f} 米"
