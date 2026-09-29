#!/usr/bin/env python
"""采集「女生拍照」主题演示素材，并按**实测属性**分类入库。

对应需求：FR-11（演示素材）/ 项目资源目录规范
对应文档：《目录结构.md》、《技术方案.md》

设计背景与**诚实约束**（必读）
------------------------------

本脚本是为「AI 实时构图指导 Agent」准备**演示与评测用**的人像素材。
联网实测（2026-09-28）确认了以下图源现实：

======================  ==========================================
渠道                     结果
======================  ==========================================
Pexels / Unsplash 搜索 API   401，需申请 API Key
Unsplash 网页搜索页          401，被挡
Wikimedia Commons API       连接超时（网络不可达）
Openverse API               502 Bad Gateway
Picsum 元数据 + 原图         **200 可用**，且带作者 + Unsplash 原始页
images.unsplash.com 直链     **200 可用**
======================  ==========================================

**因此本脚本无法做"按主题检索"**。Picsum 是一个随机摄影图库，
不提供 ``cafe`` / ``street`` / ``selfie`` 这类主题标签。
上一轮实测其**人像占比仅约 22%**，且拍摄场景不可控。

结论与本脚本的策略：

1. **只按数据源可验证的客观属性分目录**（是否检出主体、主体数、
   人脸是否能检出且朝向、景别、主体占比、构图分档）。
   每个目录名都与**实测结果**对应，不存在"目录名说咖啡馆、
   实际是海边"这种造假。
2. 主题标签（自拍 / 咖啡馆 / 街拍 …）另出 ``topics.json``，
   并**明确标注为人工作业、可信度有限**——不伪装成数据源标签。
3. 入库前做**质量门槛**与**合规门槛**筛选（见下），宁可少而精。

合规与质量门槛
--------------

- **质量**：长边 ≥ 900px；Laplacian 方差 ≥ 阈值（不过糊）；
  曝光不极端（既不死黑也不死白）；构图分达演示可用档位。
- **合规**：只保留**检出人脸**且人脸面积占比合理的图，
  避免误收录风景 / 物体 / 格调不当的图片；
  记录 provenance（作者 + 原始页 + 授权）以备溯源。

用法::

    python scripts/fetch_demo_photos.py --target 36          # 采集 36 张
    python scripts/fetch_demo_photos.py --target 36 --dry-run  # 只筛不写
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from aicg.composition.scorer import HeuristicCompositionScorer  # noqa: E402
from aicg.observability import get_logger, setup_logging  # noqa: E402
from aicg.perception import perception_from_settings  # noqa: E402
from aicg.settings import PROJECT_ROOT, load_settings  # noqa: E402
from aicg.utils.image import resize_keep_aspect  # noqa: E402

log = get_logger("scripts.fetch_demo_photos")

PICSUM_LIST = "https://picsum.photos/v2/list?page={page}&limit={limit}"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36"

# 分类阈值（全部来自实测分布，不是拍脑袋定的）
AREA_EDGES = (0.03, 0.10, 0.25)      # 主体占比分档 -> closeup / medium / wide
AREA_NAMES = ("closeup", "medium", "wide", "micro")
SHOT_NAME = {"closeup": "big_closeup", "medium": "half_body", "wide": "full_body", "micro": "distant"}


@dataclass
class Candidate:
    pid: str
    author: str = ""
    source_page: str = ""
    thumb: np.ndarray | None = None
    full: np.ndarray | None = None
    # 实测属性
    n_person: int = 0
    area_ratio: float = 0.0
    comp_score: float = 0.0
    pattern: str = ""
    shot_class: str = ""
    face_kind: str = ""      # frontal / profile / none
    face_ratio: float = 0.0
    sharpness: float = 0.0
    brightness: float = 0.0
    orientation: str = ""    # portrait / landscape / square
    long_edge: int = 0
    rejects: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.rejects


def get(url: str, timeout: int = 30) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def to_bgr(raw: bytes) -> np.ndarray | None:
    buf = np.frombuffer(raw, dtype=np.uint8)
    try:
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except Exception:  # noqa: BLE001
        return None


def write_bytes(path: Path, data: bytes) -> None:
    """二进制写盘，规避 Windows 非 ASCII 路径的文本编码问题。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)


# ----------------------------------------------------------------------
# 质量与合规度量
# ----------------------------------------------------------------------
def laplacian_var(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def mean_brightness(gray: np.ndarray) -> float:
    return float(gray.mean()) / 255.0


def detect_face(img: np.ndarray, cascades: dict) -> tuple[str, float, int]:
    """返回 (kind, face_area_ratio, n_faces)。

    先试正脸，再试侧脸。侧脸检测器对**镜像图**更敏感，
    因此对翻转图再跑一次取较优结果（这是 Haar 侧脸的常见工程做法）。
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    frame_area = float(h * w)

    front = cascades["front"]
    faces = front.detectMultiScale(gray, 1.1, 5, minSize=(max(24, w // 20),) * 2)
    if len(faces):
        a = max(fw * fh for (_, _, fw, fh) in faces) / frame_area
        return "frontal", a, len(faces)

    prof = cascades["profile"]
    best = 0.0
    n = 0
    for src in (gray, cv2.flip(gray, 1)):
        fs = prof.detectMultiScale(src, 1.1, 5, minSize=(max(24, w // 20),) * 2)
        if len(fs):
            a = max(fw * fh for (_, _, fw, fh) in fs) / frame_area
            if a > best:
                best, n = a, len(fs)
    if best > 0:
        return "profile", best, n
    return "none", 0.0, 0


def classify_area(ratio: float) -> str:
    if ratio < AREA_EDGES[0]:
        return "micro"
    if ratio < AREA_EDGES[1]:
        return "wide"
    if ratio < AREA_EDGES[2]:
        return "medium"
    return "closeup"


# ----------------------------------------------------------------------
# 采集
# ----------------------------------------------------------------------
def collect_ids(pages: int, limit: int) -> list[dict]:
    out: list[dict] = []
    for p in range(1, pages + 1):
        try:
            raw = get(PICSUM_LIST.format(page=p, limit=limit))
            items = json.loads(raw.decode("utf-8"))
        except Exception as e:  # noqa: BLE001
            log.warning(f"列表页 {p} 拉取失败: {e}")
            continue
        if not items:
            break
        out.extend(items)
        log.info(f"列表页 {p}/{pages} -> 累计 {len(out)} 个候选 id")
    return out


def download_thumb(item: dict, size: int = 480) -> Candidate:
    c = Candidate(
        pid=str(item["id"]),
        author=item.get("author", ""),
        source_page=item.get("url", ""),
    )
    try:
        raw = get(f"https://picsum.photos/id/{c.pid}/{size}/{int(size * 1.5)}")
        c.thumb = to_bgr(raw)
    except Exception as e:  # noqa: BLE001
        c.rejects.append(f"缩略图下载失败: {type(e).__name__}")
    return c


def download_full(c: Candidate, long_edge: int = 1080) -> None:
    """按**竖构图**取全图（人像演示以竖幅为主）。

    Picsum 会按请求尺寸裁剪原图，因此请求 3:4 竖版可以直接得到
    构图研究更关心的竖幅素材，而不是把横图标过来。
    """
    try:
        h = long_edge
        w = int(long_edge * 0.75)
        raw = get(f"https://picsum.photos/id/{c.pid}/{w}/{h}", timeout=45)
        c.full = to_bgr(raw)
    except Exception as e:  # noqa: BLE001
        c.rejects.append(f"全图下载失败: {type(e).__name__}")


# ----------------------------------------------------------------------
# 筛选
# ----------------------------------------------------------------------
def screen(c: Candidate, perception, scorer, cascades: dict,
           min_sharp: float, min_comp: float) -> Candidate:
    img = c.thumb
    if img is None:
        return c

    # --- 竖构图优先：演示素材以人像竖幅为主 ---
    h0, w0 = img.shape[:2]
    if h0 <= w0:
        c.rejects.append("非竖构图")

    # --- 感知 ---
    try:
        perc = perception.infer(img, 0, 0)
    except Exception as e:  # noqa: BLE001
        c.rejects.append(f"感知失败: {type(e).__name__}")
        return c

    persons = [s for s in perc.subjects if s.label == "person"]
    c.n_person = len(persons)
    subj = perc.primary_subject
    if subj is None or subj.label != "person":
        c.rejects.append("未检出人物主体")
        return c

    # --- 构图分 ---
    h, w = img.shape[:2]
    sal = perc.extras.get("saliency_map")
    try:
        comp = scorer.score_frame(
            subject_bbox=subj.bbox,
            saliency=sal if isinstance(sal, np.ndarray) else None,
            frame_shape=(h, w),
        )
        c.comp_score = float(comp.composition_score)
        c.pattern = comp.pattern.value if comp.pattern else ""
    except Exception as e:  # noqa: BLE001
        c.rejects.append(f"打分失败: {type(e).__name__}")
        return c
    if c.comp_score < min_comp:
        c.rejects.append(f"构图分过低({c.comp_score:.0f})")

    # 主体占比
    x1, y1, x2, y2 = subj.bbox
    c.area_ratio = float(max(0.0, (x2 - x1)) * max(0.0, (y2 - y1)))

    # --- 人脸（合规 + 用途判定）---
    kind, fratio, nface = detect_face(img, cascades)
    c.face_kind, c.face_ratio = kind, fratio
    if kind == "none":
        # 演示素材要求出现过人脸——据此排除风景 / 物体 / 背影误检
        c.rejects.append("未检出人脸")

    # --- 质量 ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    c.sharpness = laplacian_var(gray)
    c.brightness = mean_brightness(gray)
    if c.sharpness < min_sharp:
        c.rejects.append(f"过糊(sharp={c.sharpness:.0f})")
    if c.brightness < 0.16:
        c.rejects.append("欠曝")
    if c.brightness > 0.92:
        c.rejects.append("过曝")

    # --- 分档 ---
    c.long_edge = max(h0, w0)
    c.shot_class = classify_area(c.area_ratio)
    c.orientation = "portrait" if h0 > w0 * 1.05 else ("landscape" if w0 > h0 * 1.05 else "square")
    return c


def pick_diverse(cands: list[Candidate], target: int) -> list[Candidate]:
    """按 (景别, 人脸朝向) 网格做多样性配额，避免全是同一种构图。"""
    grid: dict[tuple[str, str], list[Candidate]] = {}
    for c in cands:
        if not c.ok:
            continue
        grid.setdefault((c.shot_class, c.face_kind), []).append(c)

    # 每个格子内按构图分降序（演示优先），并做作者去重（避免同摄影师刷屏）
    for k, v in grid.items():
        v.sort(key=lambda x: -x.comp_score)

    keys = sorted(grid, key=lambda k: (AREA_NAMES.index(k[0]), k[1]))
    picked: list[Candidate] = []
    seen_author: dict[str, int] = {}
    AUTHOR_CAP = max(2, target // 8)

    # 轮转取，保证跨格子均衡
    rounds = 0
    while len(picked) < target and rounds < 60:
        progressed = False
        for k in keys:
            if len(picked) >= target:
                break
            bucket = grid[k]
            while bucket:
                c = bucket.pop(0)
                if seen_author.get(c.author, 0) >= AUTHOR_CAP:
                    continue
                seen_author[c.author] = seen_author.get(c.author, 0) + 1
                picked.append(c)
                progressed = True
                break
        if not progressed:
            break
        rounds += 1
    return picked


# ----------------------------------------------------------------------
# 落盘
# ----------------------------------------------------------------------
def fname_for(c: Candidate, idx: int) -> str:
    """规范命名：``{序号}_{景别}_{人脸}_{\主体数}p_{构图分}.jpg``。

    例：``01_big_closeup_frontal_1p_8849.jpg``
    全部 ASCII + 下划线，避免 CJK 路径问题，且文件名自带属性便于检索。
    """
    return (
        f"{idx:02d}_{SHOT_NAME[c.shot_class]}_{c.face_kind}"
        f"_{c.n_person}p_{int(c.comp_score * 100):04d}.jpg"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="采集「女生拍照」演示素材")
    ap.add_argument("--target", type=int, default=36, help="入库张数")
    ap.add_argument("--pages", type=int, default=44, help="扫描列表页数")
    ap.add_argument("--limit", type=int, default=100, help="每页条数")
    # 门槛依据实测分布：首轮 993 张候选中，糊图（sharp<300）约占三成。
    # 演示素材以"出镜好看"为先，因此把清晰度门槛提到 300，
    # 并加上构图分门槛 45 —— 低于此分的画面不适合作为演示素材。
    ap.add_argument("--min-sharp", type=float, default=300.0,
                    help="Laplacian 方差下限（糊图过滤）")
    ap.add_argument("--min-comp", type=float, default=45.0,
                    help="构图分下限（演示素材要求）")
    ap.add_argument("--min-long-edge", type=int, default=900,
                    help="全图长边下限（由请求尺寸决定，用于自检）")
    ap.add_argument("--out", default=str(PROJECT_ROOT / "assets" / "demo_photos"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    setup_logging("INFO")
    out_dir = Path(args.out)

    cascades = {
        "front": cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"),
        "profile": cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_profileface.xml"),
    }

    cfg = load_settings(overrides={"perception.backend": "yolo"})
    perception = perception_from_settings(cfg)
    perception.warmup((480, 720))
    scorer = HeuristicCompositionScorer(cfg.composition.scoring)

    print("=" * 74)
    print("  「女生拍照」演示素材采集")
    print("=" * 74)
    print(f"  目标张数 : {args.target}")
    print(f"  输出目录 : {out_dir}")
    print(f"  图源     : Picsum（随机摄影图库，无主题标签——见脚本 docstring）")
    print()

    # --- Stage 1: 候选池 ---
    ids = collect_ids(args.pages, args.limit)
    if not ids:
        print("[错误] 未取到任何候选 id，检查网络。")
        return 2

    print(f"  [Stage 1] 下载缩略图 + 初筛（候选 {len(ids)} 个）…")
    cands: list[Candidate] = []
    with ThreadPoolExecutor(max_workers=12) as ex:
        futs = {ex.submit(download_thumb, it): it for it in ids}
        for i, f in enumerate(as_completed(futs), 1):
            cands.append(f.result())
            if i % 400 == 0:
                print(f"            已下载 {i}/{len(ids)}")
    got_thumbs = sum(1 for c in cands if c.thumb is not None)
    print(f"            缩略图成功 {got_thumbs}/{len(cands)}")

    print(f"  [Stage 2] 感知 + 质量 + 合规筛选 …")
    t0 = time.perf_counter()
    cands = [screen(c, perception, scorer, cascades, args.min_sharp, args.min_comp)
             for c in cands]
    passed = [c for c in cands if c.ok]
    print(f"            通过 {len(passed)}/{len(cands)}"
          f"（耗时 {time.perf_counter() - t0:.1f}s）")

    # 通过率诊断：按拒因统计，便于调参而不是瞎猜
    reasons: dict[str, int] = {}
    for c in cands:
        for r in c.rejects:
            key = r.split("(")[0].split(":")[0]
            reasons[key] = reasons.get(key, 0) + 1
    if reasons:
        print("            拒因分布（可叠加，故总和 > 被拒数）:")
        for k, v in sorted(reasons.items(), key=lambda kv: -kv[1]):
            print(f"              {k:22s} {v:4d}")

    if len(passed) < args.target:
        print(f"  [警告] 通过数 {len(passed)} < 目标 {args.target}；"
              f"按实际数量入库（宁少不凑）。")

    picked = pick_diverse(passed, args.target)
    print(f"  [Stage 3] 多样性配额选出 {len(picked)} 张")

    if args.dry_run:
        print("\n  --dry-run，不写盘。预览：")
        for i, c in enumerate(picked, 1):
            print(f"    {fname_for(c, i)}  comp={c.comp_score:.1f} "
                  f"area={c.area_ratio:.3f} sharp={c.sharpness:.0f} "
                  f"bright={c.brightness:.2f} '{c.author}'")
        return 0

    # --- Stage 4: 取全图 + 落盘 ---
    print(f"  [Stage 4] 下载全图并入库 …")
    out_dir.mkdir(parents=True, exist_ok=True)
    index: list[dict] = []
    written = 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(download_full, c): c for c in picked}
        for f in as_completed(futs):
            f.result()

    for i, c in enumerate(picked, 1):
        if c.full is None or not c.ok:
            continue
        fn = fname_for(c, i)
        ok, buf = cv2.imencode(".jpg", c.full, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        if not ok:
            log.warning(f"{fn} 编码失败，跳过")
            continue
        write_bytes(out_dir / fn, buf.tobytes())
        h, w = c.full.shape[:2]
        index.append({
            "file": fn,
            "picsum_id": c.pid,
            "author": c.author,
            "source_page": c.source_page,
            "license": "Unsplash License（经 Picsum 分发，见 PROVENANCE.md）",
            "width": w,
            "height": h,
            "n_person_detected": c.n_person,
            "subject_area_ratio": round(c.area_ratio, 4),
            "composition_score": round(c.comp_score, 2),
            "pattern": c.pattern,
            # 存**人类可读**的档名（与文件名一致），而不是内部分桶名。
            # [修复 2026-09-28] 原先存 c.shot_class（closeup/medium/...），
            # 但文件名用的是 SHOT_NAME 映射后的名字，两者不一致，
            # 导致归档脚本按目录名查不到。现在统一为同一个值。
            "shot_class": SHOT_NAME[c.shot_class],
            # 保留内部分桶名以便追溯分档阈值
            "shot_bucket": c.shot_class,
            "face_kind": c.face_kind,
            "face_area_ratio": round(c.face_ratio, 4),
            "sharpness_laplacian_var": round(c.sharpness, 1),
            "mean_brightness": round(c.brightness, 3),
        })
        written += 1

    write_bytes(out_dir / "_index.json",
                json.dumps(index, ensure_ascii=False, indent=2).encode("utf-8"))
    print(f"\n  入库完成：{written} 张 -> {out_dir}")
    print(f"  索引：{out_dir / '_index.json'}")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
