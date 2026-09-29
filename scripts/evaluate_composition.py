#!/usr/bin/env python
"""构图评分相关性评估（SRCC 基线测量）。

对应需求：AC-03（构图分与人工一致性）、NFR-O3（指标可复跑）
对应文档：《测试与验收.md》§4.1 —— 构图评分相关性（SRCC）

**本脚本的核心纪律（来自文档的硬性要求）**：

    > 调研中该指标为空白待填项（``SRCC = ___``）。
    > **必须先测出基线，再设定目标，禁止先定数字。**

因此本脚本**只测量、不预设**：它算出 SRCC 就是多少，然后如实报告，
并把"这个数字能不能用"的判断也一并讲清楚（样本量、标注来源、
统计显著性）。绝不为了让指标好看而挑选样本或调参。

----

**标注集从哪来？**

这是最容易被糊弄的地方。诚实的现状是：

- 权威做法：用公开构图质量数据集（如 CADB / AVA），或组织多人标注；
- **本项目现状**：尚未获取此类数据集。因此脚本内置一个小规模
  **自标注样例集**（``configs/eval/annotations.json``），用于
  **打通评估流程、验证脚本正确性**，其 SRCC 数值**不具备统计效力**，
  不得写入简历作为结论。

脚本会在样本量不足时**明确警告**，避免误读。

----

**为什么要用 Spearman 而非 Pearson**：

构图评分是**序数尺度**（"这张比那张好"是有意义的，"好 3 分"没有
绝对零点）。Spearman 基于秩次，不假设线性关系与正态分布，正适合
这种"只要相对排序对就行"的评估目标。

用法::

    python scripts/evaluate_composition.py                  # 用内置样例集
    python scripts/evaluate_composition.py --annotations my.json
    python scripts/evaluate_composition.py --backend yolo
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from aicg.composition.scorer import HeuristicCompositionScorer  # noqa: E402
from aicg.observability import get_logger, setup_logging  # noqa: E402
from aicg.perception import perception_from_settings  # noqa: E402
from aicg.settings import PROJECT_ROOT, load_settings  # noqa: E402

log = get_logger("scripts.eval")

DEFAULT_ANNOTATIONS = PROJECT_ROOT / "configs" / "eval" / "annotations.json"

MIN_MEANINGFUL_N = 30
"""SRCC 具备初步统计意义的最小样本量（经验值）。

低于此值时相关系数的置信区间极宽（n=10 时 95%CI 可宽达 ±0.6），
数值本身几乎不说明问题。因此脚本会在低于该值时显著警告。
"""


# ----------------------------------------------------------------------
def spearman_rank_correlation(x: list[float], y: list[float]) -> float:
    """计算 Spearman 秩相关系数（含并列名次处理）。

    不依赖 scipy：pandas/scipy 都可用，但本指标只需几十行，自己实现
    可避免为一个指标引入重依赖，且能精确控制并列秩的处理方式。

    并列处理：取平均秩（average rank）。这是 Spearman 的标准做法——
    若用"任意次序"打破并列，结果会随输入顺序漂移，不可复现。

    Args:
        x, y: 等长数值序列。

    Returns:
        rho ∈ [-1, 1]。样本量 < 2 或某一侧全为常数时返回 0.0。
    """
    n = len(x)
    if n != len(y):
        raise ValueError(f"长度不一致: {len(x)} vs {len(y)}")
    if n < 2:
        return 0.0

    def _ranks(vals: list[float]) -> list[float]:
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        ranks = [0.0] * len(vals)
        i = 0
        while i < len(order):
            # 找出同值区间，赋予平均秩
            j = i
            while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0  # 秩从 1 开始
            for k in range(i, j + 1):
                ranks[order[k]] = avg
            i = j + 1
        return ranks

    rx, ry = _ranks(x), _ranks(y)
    mx = sum(rx) / n
    my = sum(ry) / n

    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5

    if dx == 0.0 or dy == 0.0:
        # 某一侧完全没有区分度（全同分）——此时相关性无定义
        return 0.0
    return num / (dx * dy)


# ----------------------------------------------------------------------
def load_annotations(path: Path) -> list[dict]:
    """读取标注集。

    格式::

        {
          "source_note": "标注来源说明（必须诚实填写）",
          "items": [
            {"image": "相对或绝对路径", "human_score": 0-100 的人工评分}
          ]
        }

    ``human_score`` 必须是**人工给出的序数评分**。若无此字段，
    评估无从谈起——脚本会拒绝继续，而不是编一个分数。
    """
    if not path.exists():
        raise FileNotFoundError(f"标注文件不存在: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("标注文件缺少非空的 'items' 数组")
    for i, it in enumerate(items):
        if "image" not in it:
            raise ValueError(f"items[{i}] 缺少 'image'")
        if "human_score" not in it:
            raise ValueError(
                f"items[{i}] 缺少 'human_score' —— 没有人工评分就无法算相关性，"
                "本脚本拒绝用自动分充数"
            )
    data["_path"] = str(path)
    return data


def score_image(scorer, perception, image_path: Path) -> dict:
    """对单张图跑完整打分流程，返回评分与中间信息。"""
    p = Path(image_path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    if not p.exists():
        return {"ok": False, "error": f"图像不存在: {p}"}

    img = cv2.imread(str(p))
    if img is None:
        try:
            img = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
        except OSError:
            pass
    if img is None:
        return {"ok": False, "error": f"无法解码: {p}"}

    # 与引导链路保持一致的降采样
    from aicg.utils.image import resize_keep_aspect

    img = resize_keep_aspect(img, 480)
    h, w = img.shape[:2]

    res = perception.infer(img, 0, 0)
    subj = res.primary_subject
    if subj is None:
        # 无主体时无法评构图（本评分器以人像构图为主）——如实标记跳过，
        # 而不是给 0 分（那会污染相关性计算）
        return {"ok": False, "error": "未检出主体", "skipped": True}

    comp = scorer.score_frame(
        subject_bbox=subj.bbox,
        saliency=res.extras.get("saliency_map") if isinstance(res.extras.get("saliency_map"), np.ndarray) else None,
        frame_shape=(h, w),
    )
    return {
        "ok": True,
        "auto_score": comp.composition_score,
        "pattern": comp.pattern.value,
        "shot_size": comp.shot_size_label,
        "backend": res.backend,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="构图评分相关性评估（SRCC）")
    ap.add_argument("--annotations", default=str(DEFAULT_ANNOTATIONS),
                    help="标注集 JSON 路径")
    ap.add_argument("--backend", default=None, choices=["auto", "rule", "yolo"])
    ap.add_argument("--out", default=None, help="结果输出路径（JSON）")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    setup_logging("WARNING" if args.quiet else "INFO")

    try:
        data = load_annotations(Path(args.annotations))
    except (FileNotFoundError, ValueError) as e:
        print(f"[错误] {e}", file=sys.stderr)
        print(
            "\n提示：SRCC 评估必须有人工评分的有标注数据集。\n"
            "      若尚未准备，请先查看 configs/eval/annotations.json 的格式说明。",
            file=sys.stderr,
        )
        return 2

    cfg = load_settings()
    if args.backend:
        from aicg.settings import load_settings as _ls

        cfg = _ls(overrides={"perception.backend": args.backend})
    perception = perception_from_settings(cfg)
    perception.warmup((270, 480))
    scorer = HeuristicCompositionScorer(cfg.composition.scoring)

    items = data["items"]
    auto: list[float] = []
    human: list[float] = []
    skipped: list[str] = []
    per_item: list[dict] = []

    for it in items:
        r = score_image(scorer, perception, it["image"])
        if not r["ok"]:
            skipped.append(f"{it['image']} ({r.get('error')})")
            continue
        auto.append(float(r["auto_score"]))
        human.append(float(it["human_score"]))
        per_item.append(
            {
                "image": it["image"],
                "human_score": it["human_score"],
                "auto_score": round(r["auto_score"], 2),
                "pattern": r["pattern"],
                "shot_size": r["shot_size"],
            }
        )

    print("=" * 70)
    print("  构图评分相关性评估（SRCC 基线测量）")
    print("=" * 70)
    print(f"  感知后端   : {perception.name}")
    print(f"  标注文件   : {data['_path']}")
    print(f"  标注来源   : {data.get('source_note', '（未说明）')}")
    print(f"  标注总数   : {len(items)}")
    print(f"  有效样本   : {len(auto)}")
    if skipped:
        print(f"  跳过       : {len(skipped)}")
        for s in skipped[:5]:
            print(f"      - {s}")
        if len(skipped) > 5:
            print(f"      ... 另有 {len(skipped) - 5} 条")

    if len(auto) < 2:
        print("\n  [无法评估] 有效样本不足 2，无法计算相关系数。")
        return 1

    rho = spearman_rank_correlation(auto, human)

    print("\n" + "-" * 70)
    print("  逐项对比")
    print("-" * 70)
    print(f"  {'图像':<38s} {'人工':>6s} {'自动':>7s}")
    for it in sorted(per_item, key=lambda d: -d["human_score"]):
        name = Path(it["image"]).name[:36]
        print(f"  {name:<38s} {it['human_score']:>6.1f} {it['auto_score']:>7.1f}")

    print("\n" + "-" * 70)
    print(f"  SRCC (Spearman) : {rho:+.4f}")
    print("=" * 70)

    interpretation = _interpret(rho, len(auto))

    # 诚实提示：样本量不足时必须说清楚，防止误用
    if len(auto) < MIN_MEANINGFUL_N:
        print(
            f"\n  [重要警示] 样本量 {len(auto)} < {MIN_MEANINGFUL_N}，"
            "该 SRCC 不具备统计效力。\n"
            f"              n={len(auto)} 时相关系数的 95% 置信区间极宽，\n"
            "              此数值仅用于验证评估脚本本身可用，\n"
            "              **不得作为项目结论或写入简历**。\n"
            "              要做到有效结论，需补充到" + str(MIN_MEANINGFUL_N) + "+ 张有标注图像\n"
            "              （建议来源：CADB / AVA 等公开构图数据集，或多人标注）。"
        )

    print(f"\n  解读: {interpretation}")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
    else:
        out = PROJECT_ROOT / "outputs" / "reports" / "srcc_evaluation.json"
        out.parent.mkdir(parents=True, exist_ok=True)

    out.write_text(
        json.dumps(
            {
                "metric": "spearman_rank_correlation",
                "srcc": round(rho, 4),
                "n_samples": len(auto),
                "n_annotations": len(items),
                "n_skipped": len(skipped),
                "statistically_meaningful": len(auto) >= MIN_MEANINGFUL_N,
                "min_meaningful_n": MIN_MEANINGFUL_N,
                "annotation_source": data.get("source_note", ""),
                "annotation_method": data.get("annotation_method", ""),
                "is_subjective_human_rating": data.get("is_subjective_human_rating"),
                "known_limitation": data.get("known_limitation", ""),
                "backend": perception.name,
                "interpretation": interpretation,
                "per_item": per_item,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n  报告已保存: {out}")
    return 0


def _interpret(rho: float, n: int) -> str:
    """把 rho 翻译成人话，并显式标注置信度限制。"""
    a = abs(rho)
    if a >= 0.7:
        level = "强相关"
    elif a >= 0.4:
        level = "中等相关"
    elif a >= 0.2:
        level = "弱相关"
    else:
        level = "几乎不相关"

    direction = "同向" if rho > 0 else "反向"
    note = "" if n >= MIN_MEANINGFUL_N else "（但样本量不足，结论不可靠）"
    return f"{level}、{direction}（|rho|={a:.3f}）{note}"


if __name__ == "__main__":
    raise SystemExit(main())
