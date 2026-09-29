#!/usr/bin/env python
"""主题素材人工复核清单（固化为可复跑脚本）。

对应需求：FR-11 / NFR-O3（可复现）

**为什么把复核结果写成脚本而不是手工删文件**

人工复核的结论也是**项目资产**，必须可溯源、可复跑、可被质疑。如果只在
文件管理器里删几个文件，下一个人（或是三个月后的我）看到 ``topic_photos``
里少了图，无法知道是"漏抓"还是"人筛掉了"，也无法复现判断。

因此本脚本把逐张裁定写进 ``REVIEW`` 表：**文件 → 保留/剔除 + 具体理由**。
剔除的文件**移动**到 ``_rejected_with_reason/``（不删除，保留证据）。

复核结论的**边界**（诚实标注）
------------------------------

本轮是**单人一次目检**，判据是：

- 画面里是否有**清晰可辨的女生人物**（不是空街景 / 橱窗模特 / 男性背影）；
- 是否**不低俗、不敏感**（合规红线）；
- 构图上是否**足够当演示素材**（主体可辨、不太糊、光线可用）。

这是**主观判断，不是标注实验**，不应被当作客观基准。若用于正式评测，
需要多人标注 + 一致性统计（Krippendorff's α）。
"""

from __future__ import annotations

import json
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aicg.settings import PROJECT_ROOT  # noqa: E402

SRC = PROJECT_ROOT / "assets" / "topic_photos"
REJ = SRC / "_rejected_with_reason"

# ----------------------------------------------------------------------
# 逐张裁定：file -> {verdict, reason, topic}
#
# 编号对照拼图 outputs/reports/topic_photos_contact_sheet.png
# ----------------------------------------------------------------------
REVIEW: dict[str, dict] = {
    # ---- cafe：整组质量最好的主题 ----
    "01_cafe_853x1280_cdn.jpg": {
        "verdict": "keep", "topic": "cafe",
        "reason": "咖啡馆窗边自然光，半身景别，人物清晰、表情自然，桌面道具层次干净"},
    "02_cafe_853x1280_cdn.jpg": {
        "verdict": "keep", "topic": "cafe",
        "reason": "托腮侧视，半身；室内暖光，背景虚化到位，构图接近三分法"},
    "03_cafe_853x1280_cdn.jpg": {
        "verdict": "keep", "topic": "cafe",
        "reason": "低头端杯，动作自然；与 01/02 同系列但姿态不同，可作同组多样性"},
    "04_cafe_853x1280_cdn.jpg": {
        "verdict": "keep", "topic": "cafe",
        "reason": "深色背景 + 手持白杯，明暗对比强，主体突出，适合演示主体分割"},
    "05_cafe_1280x960_cdn.jpg": {
        "verdict": "keep", "topic": "cafe",
        "reason": "红衣、侧脸、环境道具丰富，色彩饱和度好，横构图补充景别多样性"},
    "06_cafe_960x480_cdn.jpg": {
        "verdict": "reject", "topic": "cafe",
        "reason": "隔着窗玻璃拍摄，人物被玻璃反光与框架切割，主体面积过小；且 960x480 长边仅 960，作为演示素材分辨率不足"},
    "07_cafe_1023x1280_cdn.jpg": {
        "verdict": "keep", "topic": "cafe",
        "reason": "街边咖啡座，人物居中偏左，纵深街道背景；光线与清晰度都好，构图示范性强"},

    # ---- lifestyle ----
    "08_lifestyle_853x1280_cdn.jpg": {
        "verdict": "keep", "topic": "lifestyle",
        "reason": "行走中的侧身影，动作感强，街拍气质；主体轮廓清晰可分割"},

    # ---- outdoor：两张都偏"剪影/氛围"，主体信息量不足 ----
    "09_outdoor_853x1280_cdn.jpg": {
        "verdict": "reject", "topic": "outdoor",
        "reason": "日落礁石剪影，人物几乎不可辨（sharp=105 也偏低），主体检测会失败；不符合『画面中有清晰可辨的女生人物』这一前提"},
    "10_outdoor_853x1280_cdn.jpg": {
        "verdict": "keep", "topic": "outdoor",
        "reason": "金色时刻草地剪影，人物轮廓完整可辨（比 09 明确得多），逆光氛围好，可作为『逆光/剪影』难度样本保留"},

    # ---- selfie：一组里混入了明显不合适的 ----
    "11_selfie_1280x868_cdn.jpg": {
        "verdict": "reject", "topic": "selfie",
        "reason": "后期合成感极强（蓝紫烟雾、光影不自然），与『自然真实』的要求不符，且清晰度虚高（sharp=2781）属锐化伪影"},
    "12_selfie_1280x853_cdn.jpg": {
        "verdict": "keep", "topic": "selfie",
        "reason": "红底自拍，明快活泼，构图居中；与 13 同为一组连拍，保留其中一张"},
    "13_selfie_1280x853_cdn.jpg": {
        "verdict": "reject", "topic": "selfie",
        "reason": "与 12 同一次拍摄（Pixabay girl-6920632 / woman-6930085 两图，像素指纹不同故未被自动去重），画面几乎一致；为免素材冗余只保留 12"},
    "14_selfie_960x1280_cdn.jpg": {
        "verdict": "keep", "topic": "selfie",
        "reason": "镜面自拍，圆形画中画构图，几何感强，是很好的『主体尺度/留白』对照样本"},
    "15_selfie_849x1280_cdn.jpg": {
        "verdict": "keep", "topic": "selfie",
        "reason": "面部大特写，五官清晰；长边 849 略低于 900 门槛，但特写题材本身不需要大分辨率，保留作大特写样本"},

    # ---- street：量最大，也是误报最多的一组 ----
    "16_street_853x1280_cdn.jpg": {
        "verdict": "keep", "topic": "street",
        "reason": "红衣背包行走背影，人群背景，色彩突出；背影主体对『主体检测』是有效难度样本"},
    "17_street_853x1280_cdn.jpg": {
        "verdict": "keep", "topic": "street",
        "reason": "红色大衣居中行走，人群环绕；主体明确、色彩抓眼，构图示范性好"},
    "18_street_1280x1239_cdn.jpg": {
        "verdict": "keep", "topic": "street",
        "reason": "多人并排行走，主体站在画面三分之一处；人群场景可用于演示『多主体时的主角判定』"},
    "19_street_853x1280_cdn.jpg": {
        "verdict": "keep", "topic": "street",
        "reason": "卡其色风衣背影，在绿色草地背景上分离度好；纵向构图留白充足"},
    "20_street_960x657_cdn.jpg": {
        "verdict": "keep", "topic": "street",
        "reason": "街道纵深，人物位于中景；sharp=1066 清晰度高，适合演示远景主体尺度估计"},
    "21_street_986x1280_cdn.jpg": {
        "verdict": "keep", "topic": "street",
        "reason": "两位女生并排行走，光线充足；可用于演示『多主体平衡』权重维度"},
    "22_street_853x1280_cdn.jpg": {
        "verdict": "reject", "topic": "street",
        "reason": "短裤露肤度较高，与『避免低俗』的红线要求有冲突风险；素材充足时从严处理，剔除"},
    "23_street_1280x1024_cdn.jpg": {
        "verdict": "keep", "topic": "street",
        "reason": "街头行走、棕色调统一，人物居中；与 24 同组连拍，保留其一"},
    "24_street_900x720_cdn.jpg": {
        "verdict": "keep", "topic": "street",
        "reason": "与 23 同场景但景别略远、背景层次不同；保留作同场景不同景别对照"},
    "25_street_1280x976_cdn.jpg": {
        "verdict": "keep", "topic": "street",
        "reason": "粉色上衣、蓝色长裤，服装色彩差异化明显；构图居中，背景广场干净"},
    "26_street_1000x667_images.jpg": {
        "verdict": "reject", "topic": "street",
        "reason": "主体是戴眼镜的**男性**——检索词 woman 的典型误报，与『女生拍照』主题不符，必须剔除"},

    # ---- travel ----
    "27_travel_1280x857_cdn.jpg": {
        "verdict": "keep", "topic": "travel",
        "reason": "雪山背景、女生背影侧脸，人物与风景比例得当；sharp=897 清晰"},
    "28_travel_960x638_cdn.jpg": {
        "verdict": "keep", "topic": "travel",
        "reason": "高原湖泊远景，人物极小（面积比接近 0.02）；是**主体过小**的极端样本，对评测很有价值"},
    "29_travel_1280x1280_cdn.jpg": {
        "verdict": "keep", "topic": "travel",
        "reason": "背包女生正面微笑，方构图，主体占比高；与 27/28 形成远—中—近完整梯度"},
    "30_travel_853x1280_cdn.jpg": {
        "verdict": "reject", "topic": "travel",
        "reason": "长边 853 低于 900 门槛，且人物被树干遮挡、清晰度一般；同组已有 27/28/29 三张质量更好的，取舍时剔除"},
}

# 自动去重漏掉、但仍属"同源连拍"的组合（像素指纹不同，视觉近重复）
NEAR_DUPLICATE_GROUPS: list[list[str]] = [
    ["12_selfie_1280x853_cdn.jpg", "13_selfie_1280x853_cdn.jpg"],
    ["23_street_1280x1024_cdn.jpg", "24_street_1280x976_cdn.jpg"],
]


def main() -> int:
    index_path = SRC / "_index.json"
    if not index_path.exists():
        print(f"[错误] 找不到索引 {index_path}")
        return 1
    records = json.loads(index_path.read_text(encoding="utf-8"))
    known = {r["file"] for r in records}

    missing = [f for f in REVIEW if f not in known]
    unlisted = sorted(known - set(REVIEW))
    if missing:
        print(f"[警告] REVIEW 里有 {len(missing)} 个文件不在索引中: {missing}")
    if unlisted:
        print(f"[错误] 有 {len(unlisted)} 个文件未被复核，拒绝继续（不允许漏筛）:")
        for f in unlisted:
            print(f"    {f}")
        return 1

    REJ.mkdir(parents=True, exist_ok=True)
    kept, rejected = [], []
    for r in records:
        v = REVIEW[r["file"]]
        r["review_verdict"] = v["verdict"]
        r["review_reason"] = v["reason"]
        r["review_method"] = "单人一次目检（主观，非标注实验）"
        if v["verdict"] == "keep":
            kept.append(r)
        else:
            rejected.append(r)
            src = SRC / r["file"]
            if src.exists():
                shutil.move(str(src), str(REJ / r["file"]))

    (SRC / "_index_curated.json").write_text(
        json.dumps(kept, ensure_ascii=False, indent=2), encoding="utf-8")
    (REJ / "_manifest.json").write_text(
        json.dumps(rejected, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 72)
    print("  主题素材人工复核结果")
    print("=" * 72)
    print(f"  复核总数 : {len(records)}")
    print(f"  保留     : {len(kept)}")
    print(f"  剔除     : {len(rejected)}")
    print(f"  剔除率   : {len(rejected)/len(records):.1%}")
    print()
    print("  保留主题分布:")
    for t, c in Counter(r["topic_requested"] for r in kept).most_common():
        print(f"    {t:12s} {c:3d}")
    print()
    print("  剔除明细:")
    for r in rejected:
        print(f"    {r['file']:34s} {r['review_reason'][:52]}…")
    print()
    print(f"  保留索引 -> {SRC / '_index_curated.json'}")
    print(f"  剔除证据 -> {REJ}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
