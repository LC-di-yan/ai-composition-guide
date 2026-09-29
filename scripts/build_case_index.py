"""构建案例检索库（FR-09）：素材图 → 8 维构图特征 → Milvus。

用法::

    # 试运行：只做"解码 → 感知 → 特征提取"，不连 Milvus（用于核对特征质量）
    python scripts/build_case_index.py --dry-run

    # 建库（幂等，case_id 相同即覆盖）
    python scripts/build_case_index.py

    # 推倒重建（collection 先 drop 再建——向量契约演进后必须用它）
    python scripts/build_case_index.py --rebuild

素材来源：``assets/demo_photos/``（27 张，景别/朝向/人数标注命名）+
``assets/topic_photos/``（27 张，场景标注命名）。

**诚实原则**：未检出主体的图**跳过并记录**（几何维没有真实来源），
绝不伪造向量入库；产出统计里单独列出，让"库里有 54 条"与
"素材有 54 张"的差距可解释。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402

from aicg.perception import perception_from_settings  # noqa: E402
from aicg.retrieval import (  # noqa: E402
    DIM,
    DIM_NAMES,
    CaseRecord,
    MilvusConfig,
    MilvusStore,
    StoreError,
    features_from_snapshot,
)
from aicg.settings import load_settings  # noqa: E402
from aicg.utils.image import ImageLoadError, imread_unicode  # noqa: E402

# 目录 → scene_tags 的映射（素材命名即元数据，避免人工标注成本）
DIRECTORY_TAGS: dict[str, list[str]] = {
    "demo_photos": ["demo"],
    "topic_photos": ["topic"],
}

# 文件名片段 → scene_tags（补充语义标签，如 cafe / selfie）
FILENAME_TAG_KEYWORDS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("cafe",), "cafe"),
    (("selfie",), "selfie"),
    (("outdoor",), "outdoor"),
    (("lifestyle",), "lifestyle"),
    (("big_closeup",), "big_closeup"),
    (("half_body",), "half_body"),
    (("full_body",), "full_body"),
    (("distant",), "distant"),
    (("frontal",), "frontal"),
    (("profile",), "profile"),
)


def scene_tags_for(path: Path) -> list[str]:
    """从目录名 + 文件名推断 scene_tags。"""
    tags: list[str] = list(DIRECTORY_TAGS.get(path.parent.name, []))
    name = path.stem.lower()
    for keywords, tag in FILENAME_TAG_KEYWORDS:
        if any(k in name for k in keywords):
            tags.append(tag)
    return tags


def collect_images() -> list[Path]:
    """收集案例素材：仅 demo_photos / topic_photos 的**根层** jpg。

    刻意不递归：
    - ``by_attribute/`` 是根层的归类**副本**（同名文件），递归会重复入库；
    - ``_rejected_with_reason/`` 是质量淘汰件，不该出现在案例库；
    - ``faces/`` 是感知层测试素材（AI 生成人脸），混入会让案例库
      失去"真实拍摄场景"语义。
    """
    assets = PROJECT_ROOT / "assets"
    images = sorted(
        list((assets / "demo_photos").glob("*.jpg"))
        + list((assets / "topic_photos").glob("*.jpg"))
    )
    if not images:
        raise SystemExit(f"assets 下未发现素材: {assets}")
    return images


def extract_feature(
    img: np.ndarray, perception, settings
) -> tuple[list[float] | None, str, str, str]:
    """单图 → (向量 | None, pattern, pattern_label, shot_size_label)。

    复用 FrameProcessor 的感知 + 评分链路（与运行时查询**同一份代码**，
    避免建库/查询两侧特征口径分叉——分叉会让相似度全部失真）。
    """
    from aicg.camera.base import Frame
    from aicg.pipeline.frame_processor import FrameContext, FrameProcessor

    processor = FrameProcessor(perception, settings)
    snapshot = processor.process(
        Frame(image=img, frame_id=0, timestamp_ms=0), FrameContext()
    )
    vec = features_from_snapshot(snapshot)
    comp = snapshot.composition
    return (
        vec,
        comp.pattern.value,
        comp.pattern_label,
        comp.shot_size_label,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="构建案例检索库（FR-09）")
    ap.add_argument("--rebuild", action="store_true", help="先删除既有 collection 再重建")
    ap.add_argument("--dry-run", action="store_true", help="不连 Milvus，只输出特征核对表")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 张（调试用）")
    args = ap.parse_args()

    settings = load_settings()
    perception = perception_from_settings(settings)

    images = collect_images()
    if args.limit > 0:
        images = images[: args.limit]

    print(f"素材总数: {len(images)}")
    print(f"向量契约: {DIM} 维 = {', '.join(DIM_NAMES)}")
    print("-" * 72)

    # **先握手，再跑图像链路**：实测 Windows + Milvus Lite 下，若 Milvus
    # 客户端的「首次初始化」发生在帧处理之后，进程会在 Lite 的
    # pa.RecordBatch 写入路径 SIGSEGV（详见 retrieval.search.warmup 注释）。
    # 建库脚本因此把 ensure_collection 提到特征提取之前，代价是"库不通时就
    # 不必浪费 10 秒跑图"（反而更早失败），收益是消除崩溃窗口。
    store: MilvusStore | None = None
    if not args.dry_run:
        store = MilvusStore(
            MilvusConfig(
                uri=settings.retrieval.resolved_uri,
                collection=settings.retrieval.collection,
                hnsw_m=settings.retrieval.hnsw_m,
                ef_construction=settings.retrieval.ef_construction,
                ef=settings.retrieval.ef,
                timeout_s=settings.retrieval.timeout_s,
            )
        )
        try:
            if args.rebuild:
                store.drop()
                print("已删除既有 collection（--rebuild）")
            store.ensure_collection()
            print("已就绪 collection（首次接触先于图像链路）")
        except StoreError as exc:
            print(f"\n准备 collection 失败: [{exc.reason}] {exc}")
            print("提示：Milvus 未启动时请先执行 docker/docker-compose.yml 的 milvus 服务，")
            print("或用 Milvus Lite 本地文件（AICG_RETRIEVAL_URI=<路径>.db）。")
            return 1

    records: list[CaseRecord] = []
    skipped: list[tuple[str, str]] = []
    t0 = time.perf_counter()

    for p in images:
        image_ref = p.relative_to(PROJECT_ROOT).as_posix()
        try:
            img = imread_unicode(p)
        except ImageLoadError as exc:
            skipped.append((image_ref, f"解码失败: {exc}"))
            continue
        try:
            vec, pattern, pattern_label, shot_label = extract_feature(
                img, perception, settings
            )
        except Exception as exc:  # noqa: BLE001 — 单张失败不拖垮整个建库
            skipped.append((image_ref, f"特征提取异常: {exc}"))
            continue
        if vec is None:
            # 无主体：诚实跳过。评分器兜底框不是真实信息，入库=伪造。
            skipped.append((image_ref, "未检出主体"))
            continue

        tags = scene_tags_for(p)
        description = f"{shot_label}｜{pattern_label}"
        records.append(
            CaseRecord(
                case_id=p.stem,
                vector=vec,
                image_ref=image_ref,
                pattern=pattern,
                scene_tags=tags,
                description=description,
            )
        )
        preview = " ".join(f"{v:.3f}" for v in vec)
        print(f"  {image_ref:52s} pattern={pattern:14s} vec=[{preview}]")

    elapsed = time.perf_counter() - t0
    print("-" * 72)
    print(f"成功提取: {len(records)} 张｜跳过: {len(skipped)} 张｜耗时 {elapsed:.1f}s")
    for name, why in skipped:
        print(f"  [跳过] {name}: {why}")

    if args.dry_run:
        print("\n[dry-run] 未连接 Milvus。核对特征后去掉 --dry-run 正式建库。")
        return 0
    if not records:
        print("\n无可用案例（全部跳过），不建库。")
        return 1

    try:
        assert store is not None  # 入口已保证非 dry-run 时先握手
        inserted = store.upsert(records)
        total = store.count()
    except StoreError as exc:
        print(f"\n建库失败: [{exc.reason}] {exc}")
        print("提示：Milvus 未启动时请先执行 docker/docker-compose.yml 的 milvus 服务，")
        print("或本机 docker run（配方见 docs/research/Milvus本地方案调研.md §3）。")
        return 1
    print(f"\n入库: {inserted} 条｜库内总数: {total}｜collection={settings.retrieval.collection}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
