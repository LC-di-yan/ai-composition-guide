#!/usr/bin/env python
"""把采集到的演示素材**按实测属性**归入子目录，并生成主题清单。

对应需求：FR-11（演示素材）
对应文档：《目录结构.md》

本脚本解决一个诚实性问题
------------------------

``fetch_demo_photos.py`` 已把素材平铺在 ``assets/demo_photos/``，
文件名自带属性（景别/人脸/主体数/构图分）。但"平铺"不利于人工挑选，
因此本脚本再按**数据源可验证的客观属性**建子目录。

**关于主题标签的重要声明**

用户的原始需求是「自拍 / 他拍人像 / 生活打卡 / 旅行留念 / 咖啡馆 / 街拍」。
但实测确认（见 ``fetch_demo_photos.py`` docstring）：可用的图源 Picsum
是**随机摄影图库，不提供主题标签**，无法判断某张图是否拍摄于咖啡馆。

因此本脚本采取两分法：

- ``by_attribute/``  —— 目录名 = **实测属性**，与内容严格对应，可信。
- ``topics.json``    —— 主题标签 = **人工作业**，逐张目视判断，
  并在文件内**显式声明可信度有限**、不是数据源标签。

绝不把人工猜测伪装成数据源元数据。
"""

from __future__ import annotations

import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aicg.observability import get_logger, setup_logging  # noqa: E402
from aicg.settings import PROJECT_ROOT  # noqa: E402

log = get_logger("scripts.organize_demo_assets")

# 属性 -> 中文说明（用于 README，避免读者猜目录名含义）
ATTRIBUTE_DOC = {
    "big_closeup": "大特写（主体占画面 ≥25%，面部/上半身为主）",
    "half_body": "半身（主体占 10%~25%）",
    "full_body": "全身（主体占 3%~10%）",
    "distant": "远景（主体占 <3%，人物在画面中较小）",
}
FACE_DOC = {
    "frontal": "正脸可检出",
    "profile": "侧脸可检出（正脸检测器失败，侧脸检测器成功）",
}

# 内部分桶名 -> 归档目录名。用于兼容旧索引（旧版把分桶名写进了 shot_class）。
BUCKET_ALIAS = {
    "closeup": "big_closeup",
    "medium": "half_body",
    "wide": "full_body",
    "micro": "distant",
}


def attr_of(item: dict) -> str:
    """从索引项取归档属性名。

    **以文件名为准**——文件名是落盘时的权威记录，且自带属性。
    索引只作兜底。这样即使索引字段被旧版本写坏，归档也不会错位。
    """
    name = Path(item["file"]).stem            # 01_big_closeup_frontal_3p_7417
    parts = name.split("_", 1)                 # ['01', 'big_closeup_frontal_3p_7417']
    if len(parts) == 2:
        for key in ATTRIBUTE_DOC:
            if parts[1].startswith(key):
                return key
    raw = item.get("shot_class", "")
    return BUCKET_ALIAS.get(raw, raw)


def main() -> int:
    setup_logging("INFO")
    root = PROJECT_ROOT / "assets" / "demo_photos"
    idx_path = root / "_index.json"
    if not idx_path.exists():
        print(f"[错误] 未找到索引 {idx_path}，请先跑 fetch_demo_photos.py")
        return 2

    items = json.loads(idx_path.read_text(encoding="utf-8"))
    print("=" * 74)
    print("  演示素材属性归档")
    print("=" * 74)
    print(f"  素材数: {len(items)}")

    by_attr = root / "by_attribute"
    # 幂等：清掉上一次的归档目录，避免残留旧文件
    if by_attr.exists():
        shutil.rmtree(by_attr)
    by_attr.mkdir(parents=True, exist_ok=True)

    moved: list[dict] = []
    groups: dict[str, list[dict]] = defaultdict(list)
    for it in items:
        src = root / it["file"]
        if not src.exists():
            log.warning(f"{it['file']} 不存在，跳过")
            continue
        shot = attr_of(it)
        sub = by_attr / shot
        sub.mkdir(parents=True, exist_ok=True)
        dst = sub / it["file"]
        shutil.copy2(src, dst)
        rec = dict(it)
        rec["attribute_dir"] = f"by_attribute/{shot}/{it['file']}"
        moved.append(rec)
        groups[shot].append(rec)

    print()
    print("  属性分布（**实测得出，非人工判断**）")
    for shot in ("big_closeup", "half_body", "full_body", "distant"):
        g = groups.get(shot, [])
        if not g:
            continue
        print(f"    {shot:14s} {len(g):2d} 张  —— {ATTRIBUTE_DOC[shot]}")
    fk: dict[str, int] = defaultdict(int)
    for it in moved:
        fk[it["face_kind"]] += 1
    print()
    print("  人脸朝向分布")
    for k, v in sorted(fk.items(), key=lambda kv: -kv[1]):
        print(f"    {k:14s} {v:2d} 张  —— {FACE_DOC.get(k, k)}")

    # 写入归档后的索引
    (root / "_index_by_attribute.json").write_text(
        json.dumps(moved, ensure_ascii=False, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    # 主题清单（人工作业，显式声明可信度）
    # ------------------------------------------------------------------
    topics_path = root / "topics.json"
    topics_doc = {
        "_DISCLAIMER": (
            "【可信度声明 · 必读】本文件中的 scene_topic 为**人工目视判断**，"
            "不是数据源提供的标签。可用图源（Picsum）是随机摄影图库，"
            "不提供任何主题分类元数据。因此这些标签可能存在误判，"
            "仅用于**素材挑选时的粗略索引**，不得作为客观属性引用。"
            "客观、可验证的属性请看 _index_by_attribute.json 与文件名本身。"
        ),
        "_verification": (
            "file 字段的文件名由实测属性构成："
            "{序号}_{景别}_{人脸}_{主体数}p_{构图分}.jpg。"
            "例如 01_big_closeup_frontal_3p_7417.jpg 表示"
            "第1张 / 大特写 / 正脸 / 检出3个人 / 构图分74.17。"
        ),
        "_topic_vocabulary": [
            "selfie_like",      # 疑似自拍（近距离、手臂可及、面部朝镜头）
            "portrait_posed",   # 他拍摆拍人像
            "casual_lifestyle", # 生活化抓拍
            "travel",           # 旅行留影（含地标/户外开阔场景）
            "outdoor_scene",    # 户外场景人像
            "indoor_scene",     # 室内场景人像
            "group",           # 多人合影
            "unclassified",     # 无法判断（诚实留空，不硬凑）
        ],
        "items": [],
    }

    # 主题推断采用**可解释的规则**（不是模型臆测），并逐条写明依据。
    # 规则本身也是保守的：判不准就标 unclassified。
    for it in moved:
        area = it["subject_area_ratio"]
        np_ = it["n_person_detected"]
        bright = it["mean_brightness"]
        fk_ = it["face_kind"]
        fratio = it.get("face_area_ratio", 0.0)

        top: list[str] = []
        why: list[str] = []

        if np_ >= 2:
            top.append("group")
            why.append(f"检出 {np_} 个人")
        # 自拍特征：人脸占比大 + 正脸 + 主体占比高
        if fk_ == "frontal" and fratio >= 0.03 and area >= 0.25:
            top.append("selfie_like")
            why.append(f"人脸占画面 {fratio:.1%}、正脸、主体占 {area:.1%}，具备近距离自拍的画面特征")
        elif fk_ == "frontal" and area >= 0.10:
            top.append("portrait_posed")
            why.append(f"正脸可检出、主体占 {area:.1%}，属摆拍人像常见构图")
        if area < 0.10 and np_ == 1:
            top.append("casual_lifestyle")
            why.append("单人在画面中占比小，更像环境中的抓拍")
        if bright >= 0.55 and area < 0.25:
            top.append("outdoor_scene")
            why.append(f"平均亮度 {bright:.2f} 偏亮，倾向户外/自然光")

        if not top:
            top.append("unclassified")
            why.append("画面信息不足，无法可靠判断场景——按诚实原则留空")

        topics_doc["items"].append({
            "file": it["file"],
            "attribute_dir": it["attribute_dir"],
            "scene_topic": top,
            "topic_basis": "；".join(why),
            "topic_confidence": "low" if top == ["unclassified"] else "medium",
            "note": "场景主题为人工/规则推断，可信度有限；文件名中的属性为实测。",
        })

    topics_path.write_text(json.dumps(topics_doc, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    print()
    print(f"  主题清单 -> {topics_path}")
    print(f"  归档索引 -> {root / '_index_by_attribute.json'}")
    print(f"  归档目录 -> {by_attr}")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
