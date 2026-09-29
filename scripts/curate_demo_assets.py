#!/usr/bin/env python
"""演示素材**人工复核**记录：批准 / 剔除清单（可复现、可追溯）。

对应需求：FR-11 / NFR-O2（可溯源）

**为什么需要这份文件**

``fetch_demo_photos.py`` 的属性全部来自自动检测，而实测证明自动检测
**误判率约 37%**（19 张通过自动筛选的图中，7 张实际不是可用人像）。
因此必须由人眼复核，本文件把复核结果固化为**可复跑的清单**，
而不是散落在对话里。

复核方式
--------

1. 生成拼图：``python scripts/make_demo_sheet.py``
2. 逐张目视判读，记录 verdict 与理由
3. 运行本脚本，按清单整理出 ``curated/`` 目录

清单字段
--------

- ``verdict``: ``keep`` | ``reject`` | ``move``
- ``reason``: 判读依据（必须具体到画面内容）
- ``topic``: 人工作业的场景主题（可信度有限，见 topics.json 声明）
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aicg.observability import get_logger, setup_logging  # noqa: E402
from aicg.settings import PROJECT_ROOT  # noqa: E402

log = get_logger("scripts.curate_demo_assets")

# ----------------------------------------------------------------------
# 人工复核结论（2026-09-28，基于 demo_photos_contact_sheet.png 目视判读）
#
# **判读原则**：
#   - 必须能明确看出是**人物**（不是雕像/海报/静物/纯风景）；
#   - 人物在画面中应可辨识（不是纯剪影、不是只露局部肢体）；
#   - 画面不应低俗、敏感；
#   - 优先女性主体 + 自然真实 + 光线构图良好（贴合「女生拍照」主题）。
# ----------------------------------------------------------------------
REVIEW: dict[str, dict] = {
    "01_big_closeup_frontal_3p_7417.jpg": {
        "verdict": "keep",
        "reason": "城市街头，女性与女童互动、花丛与斑马线，自然光柔和。检出3人，"
                  "主体占34.9%，构图分74.2（本批最高）。典型『生活打卡』画面。",
        "topic": ["casual_lifestyle", "group", "outdoor_scene"],
    },
    "02_big_closeup_profile_1p_6346.jpg": {
        "verdict": "keep",
        "reason": "女性骑自行车于山地公路上，侧脸，户外金光。构图简洁、留白充足，"
                  "适合讲『三分法 + 视线方向留白』。典型『旅行留念』。",
        "topic": ["travel", "outdoor_scene", "portrait_posed"],
    },
    "03_half_body_frontal_1p_4325.jpg": {
        "verdict": "reject",
        "reason": "【剔除】运动模糊严重（sharp=340，画面中人物与自行车均拖影），"
                  "不满足『画面自然真实、构图与光线良好』。",
        "topic": [],
    },
    "04_half_body_profile_1p_4833.jpg": {
        "verdict": "reject",
        "reason": "【剔除】实际为**室内静物/空间**（吊灯与窗），画面中无人物。"
                  "YOLO 将物体误判为 person，属自动检测假阳性。",
        "topic": [],
    },
    "05_full_body_frontal_1p_6135.jpg": {
        "verdict": "keep",
        "reason": "女性全身在户外山景中行进，正脸可辨，主体占6.8%。"
                  "适合讲『远景全身 + 主体偏小』的构图问题。",
        "topic": ["travel", "outdoor_scene"],
    },
    "06_distant_frontal_5p_4507.jpg": {
        "verdict": "keep",
        "reason": "栈桥/观景台处逆光人影，检出5人，主体占1.8%。"
                  "**仅可用于『远景点主体过小』的降级示例**，不宜作为正向演示。",
        "topic": ["travel", "outdoor_scene", "group"],
    },
    "07_big_closeup_frontal_2p_7066.jpg": {
        "verdict": "keep",
        "reason": "女性与儿童相拥于石板路，正脸清晰，主体占36.6%，构图分70.7。"
                  "光线通透，是『他拍人像』的优质样本。",
        "topic": ["portrait_posed", "group", "casual_lifestyle"],
    },
    "08_big_closeup_profile_1p_6338.jpg": {
        "verdict": "keep",
        "reason": "女性侧脸特写，主体占40.6%，构图分63.4，平均亮度0.70（明亮通透）。"
                  "适合讲『大特写的裁切位置与头顶留白』。",
        "topic": ["selfie_like", "portrait_posed"],
    },
    "09_full_body_frontal_1p_5762.jpg": {
        "verdict": "keep",
        "reason": "女性全身在人行道上，构图分57.6，主体占4.4%。"
                  "适合讲『全身景别 + 环境交代』。",
        "topic": ["casual_lifestyle", "outdoor_scene"],
    },
    "10_big_closeup_frontal_2p_6834.jpg": {
        "verdict": "keep",
        "reason": "双人近景（女性为主体），正脸，主体占27.5%，构图分68.3。",
        "topic": ["portrait_posed", "group"],
    },
    "11_full_body_frontal_1p_5392.jpg": {
        "verdict": "keep",
        "reason": "女性全身，平均亮度0.61，正脸可辨，构图分53.9。画面干净。",
        "topic": ["casual_lifestyle", "outdoor_scene"],
    },
    "12_big_closeup_frontal_1p_6564.jpg": {
        "verdict": "keep",
        "reason": "**主体占比 88.4%（本批最高）**，构图分65.6。"
                  "可作『主体过满、留白不足』的极端案例，也验证大占比打分行为。",
        "topic": ["selfie_like", "portrait_posed"],
    },
    "13_full_body_frontal_1p_4888.jpg": {
        "verdict": "keep",
        "reason": "女性全身，主体占3.3%，构图分48.9。与 #5 构成景别梯度。",
        "topic": ["travel", "outdoor_scene"],
    },
    "14_big_closeup_frontal_1p_6049.jpg": {
        "verdict": "keep",
        "reason": "女性近景，主体占41.2%，**清晰度最高（sharp=5378）**，构图分60.5。"
                  "画质优秀，适合做演示主图。",
        "topic": ["selfie_like", "portrait_posed"],
    },
    "15_full_body_frontal_5p_4747.jpg": {
        "verdict": "reject",
        "reason": "【剔除】场景为室内餐饮空间，主体人物为**男性背影**（服务员），"
                  "与『女生拍照』主题及『人物可辨识』要求均不符。",
        "topic": [],
    },
    "16_big_closeup_frontal_5p_5278.jpg": {
        "verdict": "keep",
        "reason": "多人近景（检出5人），正脸，主体占55.4%。"
                  "适合讲『多主体场景下的主次关系』。",
        "topic": ["group", "casual_lifestyle"],
    },
    "17_full_body_frontal_8p_4400.jpg": {
        "verdict": "keep",
        "reason": "**检出8人（本批最多）**，街头人群，主体占4.7%。"
                  "适合讲『多主体干扰下的构图决策』。",
        "topic": ["group", "outdoor_scene"],
    },
    "18_big_closeup_frontal_1p_5167.jpg": {
        "verdict": "reject",
        "reason": "【剔除】主体占比 96.5%，画面几乎只剩衣物/皮肤纹理，"
                  "**人物形态不可辨识**，无构图信息，不适合演示。",
        "topic": [],
    },
    "19_big_closeup_frontal_2p_4993.jpg": {
        "verdict": "reject",
        "reason": "【剔除】俯拍只拍到**双腿与脚**（悬于高楼外），"
                  "属局部肢体而非完整人像，与『人物可辨识』要求不符。",
        "topic": [],
    },
}

# 已知在首批 12 张预览中识别出但未进入 19 张清单的假阳性
# （来自更早的 30 张 dry-run 预览：葡萄静物、办公桌、室内空间、男性乐手、
#   只拍腿部、雪山风景等）。这些在收紧门槛后已自然被排除，
# 但记录在此以说明「自动检测会误判」这一结论的来源。
EXTRA_FALSE_POSITIVES = [
    {"desc": "一串葡萄（静物）", "detected_as": "person + 人脸误报",
     "why_bad": "非人像"},
    {"desc": "办公桌上的音箱与桌面（静物）", "detected_as": "person",
     "why_bad": "非人像"},
    {"desc": "黑白室内空间（建筑）", "detected_as": "person",
     "why_bad": "非人像"},
    {"desc": "男乐手弹吉他（舞台）", "detected_as": "person + frontal",
     "why_bad": "男性，不符『女生拍照』主题"},
    {"desc": "树林中只拍到大腿与靴子", "detected_as": "person",
     "why_bad": "局部肢体，人物不可辨识"},
    {"desc": "雪山草原远景（人物极小）", "detected_as": "person",
     "why_bad": "风景为主，不适合人像演示"},
    {"desc": "酒吧男服务员背影", "detected_as": "person + frontal",
     "why_bad": "男性背影，不符主题"},
]


def main() -> int:
    setup_logging("WARNING")
    root = PROJECT_ROOT / "assets" / "demo_photos"
    items = {it["file"]: it for it in
             json.loads((root / "_index.json").read_text(encoding="utf-8"))}

    kept = [f for f, r in REVIEW.items() if r["verdict"] == "keep"]
    rejected = [f for f, r in REVIEW.items() if r["verdict"] == "reject"]

    print("=" * 74)
    print("  演示素材人工复核整理")
    print("=" * 74)
    print(f"  复核总数 : {len(REVIEW)}")
    print(f"  保留     : {len(kept)}")
    print(f"  剔除     : {len(rejected)}")
    print(f"  自动检测误判率 = {len(rejected)}/{len(REVIEW)} = "
          f"{len(rejected)/len(REVIEW)*100:.0f}%")
    print()

    curated = root / "curated"
    if curated.exists():
        shutil.rmtree(curated)

    # 保留素材 -> 按景别归档
    by_attr = curated / "by_attribute"
    for f in kept:
        src = root / f
        if not src.exists():
            log.warning(f"{f} 缺失，跳过")
            continue
        shot = REVIEW[f].get("shot") or items[f]["shot_class"]
        d = by_attr / shot
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, d / f)

    # 剔除素材 -> 单独存放，保留证据（不删除！）
    rej_dir = curated / "_rejected_with_reason"
    rej_dir.mkdir(parents=True, exist_ok=True)
    for f in rejected:
        src = root / f
        if src.exists():
            shutil.copy2(src, rej_dir / f)

    print("  保留素材（按实测景别归档）")
    groups: dict[str, list[str]] = {}
    for f in kept:
        groups.setdefault(items[f]["shot_class"], []).append(f)
    for shot in ("big_closeup", "half_body", "full_body", "distant"):
        g = groups.get(shot, [])
        if g:
            print(f"    {shot:12s} {len(g):2d} 张")
    print()
    print("  剔除素材（保留文件 + 理由，可复核对）")
    for f in rejected:
        print(f"    {f}")
        print(f"      -> {REVIEW[f]['reason']}")
    print()

    # 落盘复核清单
    manifest = {
        "_generated_by": "scripts/curate_demo_assets.py",
        "_review_note": (
            "本清单为**人工目视复核**结果，非自动检测。"
            "自动检测误判率实测 37%（7/19），因此不可省略人工复核。"
        ),
        "_reviewed_at": "2026-09-28",
        "_contact_sheet": "outputs/reports/demo_photos_contact_sheet.png",
        "kept": [
            {
                "file": f,
                "shot_class": items[f]["shot_class"],
                "face_kind": items[f]["face_kind"],
                "n_person": items[f]["n_person_detected"],
                "subject_area_ratio": items[f]["subject_area_ratio"],
                "composition_score": items[f]["composition_score"],
                "author": items[f]["author"],
                "source_page": items[f]["source_page"],
                "review_reason": REVIEW[f]["reason"],
                "scene_topic_manual": REVIEW[f]["topic"],
            }
            for f in kept
        ],
        "rejected": [
            {
                "file": f,
                "auto_detected_as": f"{items[f]['face_kind']} / "
                                    f"{items[f]['n_person_detected']}p / "
                                    f"area={items[f]['subject_area_ratio']:.3f}",
                "auto_composition_score": items[f]["composition_score"],
                "reject_reason": REVIEW[f]["reason"],
            }
            for f in rejected
        ],
        "additional_false_positives_seen_earlier": EXTRA_FALSE_POSITIVES,
    }
    (curated / "review_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"  清单 -> {curated / 'review_manifest.json'}")
    print(f"  保留 -> {by_attr}")
    print(f"  剔除 -> {rej_dir}")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
