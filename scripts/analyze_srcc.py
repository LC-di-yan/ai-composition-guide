#!/usr/bin/env python
"""真实评测集 SRCC 深度分析（含共线性诊断与稳健性检验）。

对应需求：AC-03
对应文档：《测试与验收.md》§4.1

为什么需要这个脚本：单看一个 SRCC = +0.9113 无法判断结论可信度。
本脚本把"这个数字虚不虚"的检验过程本身固化下来，可复跑：

1. **共线性诊断** —— 若两个分项高度共线，则评分器与原则分
   "同向"可能只是都在测同一个物理量，验证意义被削弱；
2. **分层 SRCC** —— 在主体占比近似固定的子集内重算，
   剔除"主体大小"这一主导维度后还剩多少一致性；
3. **偏相关** —— 对两边的秩同时回归掉 area_ratio，看残差相关；
4. **退化检查** —— 确认没有任何分项在所有样本上取常数
   （这正是旧标注集 subject_scale 全 1.00 的失败模式）。

用法::

    python scripts/analyze_srcc.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from aicg.settings import PROJECT_ROOT  # noqa: E402

ANN = PROJECT_ROOT / "configs" / "eval" / "annotations_real.json"
SRCC = PROJECT_ROOT / "outputs" / "reports" / "srcc_real.json"
OUT = PROJECT_ROOT / "outputs" / "reports" / "srcc_real_analysis.json"


def rank(v: list[float]) -> np.ndarray:
    """平均秩（与 evaluate_composition.py 的并列处理一致）。"""
    a = np.asarray(v, dtype=float)
    order = np.argsort(a, kind="mergesort")
    r = np.empty(len(a), dtype=float)
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and a[order[j + 1]] == a[order[i]]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return r


def srcc(x, y) -> float:
    rx, ry = rank(list(x)), rank(list(y))
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    den = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return 0.0 if den == 0 else float((rx * ry).sum() / den)


def pearson(x, y) -> float:
    a = np.asarray(x, float) - np.mean(x)
    b = np.asarray(y, float) - np.mean(y)
    den = np.sqrt((a ** 2).sum() * (b ** 2).sum())
    return 0.0 if den == 0 else float((a * b).sum() / den)


def partial_srcc(x, y, z) -> float:
    """控制 z 之后 x 与 y 的秩偏相关。"""
    rz = rank(list(z))
    A = np.vstack([np.ones_like(rz), rz]).T

    def resid(v):
        rv = rank(list(v))
        beta, *_ = np.linalg.lstsq(A, rv, rcond=None)
        return rv - A @ beta

    return srcc(resid(x), resid(y))


def main() -> int:
    ann = json.loads(ANN.read_text(encoding="utf-8"))
    ev = json.loads(SRCC.read_text(encoding="utf-8"))
    per = {Path(p["image"]).name: p for p in ev["per_item"]}
    by_name = {Path(i["image"]).name: i for i in ann["items"]}

    names = [n for n in per if n in by_name]
    auto = [per[n]["auto_score"] for n in names]
    human = [per[n]["human_score"] for n in names]
    area = [by_name[n]["area_ratio"] for n in names]
    hgt = [by_name[n]["subject_bbox"][3] - by_name[n]["subject_bbox"][1] for n in names]
    top = [by_name[n]["subject_bbox"][1] for n in names]
    det = {k: [by_name[n]["principle_detail"][k] for n in names]
           for k in ("thirds", "headroom", "scale", "balance")}

    R: dict = {"n": len(names)}

    R["headline"] = {
        "srcc_full": round(srcc(human, auto), 4),
        "note": "全样本 SRCC（原则分 vs 自动分）",
    }

    # 1. 退化检查
    R["degeneracy_check"] = {
        "subjects": {
            k: {"min": round(float(min(v)), 4), "max": round(float(max(v)), 4),
                "stdev": round(float(np.std(v)), 4),
                "degenerate": bool(np.std(v) < 1e-6)}
            for k, v in det.items()
        },
        "auto_score": {"min": round(min(auto), 2), "max": round(max(auto), 2),
                       "stdev": round(float(np.std(auto)), 3)},
        "human_score": {"min": round(min(human), 2), "max": round(max(human), 2),
                        "stdev": round(float(np.std(human)), 3)},
        "subject_height": {"min": round(min(hgt), 4), "max": round(max(hgt), 4),
                           "degenerate": bool(np.std(hgt) < 1e-6)},
    }

    # 2. 逐分项与自动分的相关性
    R["dimensionwise"] = {
        k: round(srcc(v, auto), 4) for k, v in det.items()
    }
    R["dimensionwise"]["area_ratio"] = round(srcc(area, auto), 4)

    # 3. 共线性
    R["collinearity"] = {
        "pearson_subject_height_vs_top_margin": round(pearson(hgt, top), 4),
        "pearson_subject_height_vs_headroom_sub": round(pearson(hgt, det["headroom"]), 4),
        "pearson_area_ratio_vs_subject_height": round(pearson(area, hgt), 4),
        "pearson_headroom_sub_vs_scale_sub": round(pearson(det["headroom"], det["scale"]), 4),
    }

    # 4. 分层
    strata = {
        "主体极小 area<0.05": [i for i, a in enumerate(area) if a < 0.05],
        "主体中等 0.05<=area<=0.35": [i for i, a in enumerate(area) if 0.05 <= a <= 0.35],
        "主体很大 area>0.35": [i for i, a in enumerate(area) if a > 0.35],
    }
    R["stratified"] = {}
    for label, idx in strata.items():
        if len(idx) >= 5:
            R["stratified"][label] = {
                "n": len(idx),
                "srcc": round(srcc([human[i] for i in idx], [auto[i] for i in idx]), 4),
            }
        else:
            R["stratified"][label] = {"n": len(idx), "srcc": None, "note": "样本不足"}

    # 5. 偏相关
    R["partial"] = {
        "srcc_controlling_area_ratio": round(partial_srcc(human, auto, area), 4),
        "srcc_controlling_subject_height": round(partial_srcc(human, auto, hgt), 4),
        "note": "控制主体大小后残留的排序一致性；这是剔除'都在测主体大小'后的下界估计",
    }

    # 6. 剔除非人像主体后的 SRCC（需外部标记，缺失则跳过）
    R["scope_note"] = {
        "portrait_ratio": "见 screen_report.json 与 PROVENANCE.md；"
                          "本脚本不重新跑感知，避免重复推理"
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(R, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 70)
    print("  真实评测集 SRCC 深度分析")
    print("=" * 70)
    print(f"  样本量             : {R['n']}")
    print(f"  全样本 SRCC        : {R['headline']['srcc_full']:+.4f}")
    print()
    print("  --- 退化检查（是否出现常数分项）---")
    for k, v in R["degeneracy_check"]["subjects"].items():
        flag = "  <== 退化为常数!" if v["degenerate"] else ""
        print(f"    {k:10s} 范围 [{v['min']:.3f}, {v['max']:.3f}] std={v['stdev']:.3f}{flag}")
    print(f"    自动分     范围 [{R['degeneracy_check']['auto_score']['min']:.2f}, "
          f"{R['degeneracy_check']['auto_score']['max']:.2f}] "
          f"std={R['degeneracy_check']['auto_score']['stdev']:.3f}")
    print(f"    原则分     范围 [{R['degeneracy_check']['human_score']['min']:.2f}, "
          f"{R['degeneracy_check']['human_score']['max']:.2f}] "
          f"std={R['degeneracy_check']['human_score']['stdev']:.3f}")
    print()
    print("  --- 各维度 vs 自动分 ---")
    for k, v in R["dimensionwise"].items():
        print(f"    auto vs {k:12s} SRCC = {v:+.4f}")
    print()
    print("  --- 共线性诊断 ---")
    for k, v in R["collinearity"].items():
        print(f"    {k:46s} = {v:+.4f}")
    print()
    print("  --- 分层 ---")
    for k, v in R["stratified"].items():
        s = f"{v['srcc']:+.4f}" if v["srcc"] is not None else v.get("note", "-")
        print(f"    {k:28s} n={v['n']:2d}  SRCC={s}")
    print()
    print("  --- 偏相关（剔除主体大小的主导效应）---")
    for k, v in R["partial"].items():
        if k != "note":
            print(f"    {k:40s} = {v:+.4f}")
    print("=" * 70)
    print(f"  报告 -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
