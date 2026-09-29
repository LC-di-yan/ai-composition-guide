# Milvus Windows 本地方案调研（M6-2 前置）

> 日期：2026-09-29　目的：为 FR-09 案例检索选定本机可**真实运行**的 Milvus 形态。
> 纪律约束：绝不重演「Docker 标已备未实测」——**检索模块必须在真实 Milvus 上验证后才允许标已实现**。

## 1. 结论

**选 Docker Standalone 单容器（内嵌 etcd + 本地存储），镜像 `milvusdb/milvus:v2.5.11`。**

## 2. 候选方案对比

| 方案 | Windows 支持 | 版本 | 与生产一致性 | 判定 |
| --- | --- | --- | --- | --- |
| **A. Milvus Lite**（`MilvusClient("./x.db")` 内嵌） | ❌ **不支持**。官方：仅 Ubuntu ≥ 20.04 / macOS ≥ 11.0（milvus-io/milvus discussion #36759 维护者确认；官方 Operational FAQ 同样答复） | 2.4+ | 高 | 排除（Windows 无法装 `milvus-lite` 包，`import milvus_lite` 直接 ModuleNotFoundError） |
| **B. 旧版 pip 包**（`milvus==2.2.16` + `milvus-server` 命令） | ✅ 可跑 | **2.2.x（2023 年）** | 低——需锁死 `pymilvus==2.2.13` + `marshmallow<4`（CSDN 实录），无 `MilvusClient` 新 API | 排除——为本地省一个 Docker 换来一套锁死的旧依赖，作品集价值为负 |
| **C. Docker Standalone 单容器**（内嵌 etcd + local storage） | ✅ 官方文档明确支持（milvus.io「Run Milvus in Docker (Windows)」，Docker Desktop + WSL2） | 2.5.11 | **高**——与 CI service container / 未来部署同版本 | **选定** |

## 3. 单容器启动配方（官方 scale-standalone.md 原样，经 Docker Hub 文档交叉核对）

```bash
docker run -d --name milvus-standalone \
  --security-opt seccomp:unconfined \
  -e ETCD_USE_EMBED=true \
  -e ETCD_DATA_DIR=/var/lib/milvus/etcd \
  -e ETCD_CONFIG_PATH=/milvus/configs/embedEtcd.yaml \
  -e COMMON_STORAGETYPE=local \
  -v ./volumes/milvus:/var/lib/milvus \
  -v ./embedEtcd.yaml:/milvus/configs/embedEtcd.yaml:ro \
  -v ./user.yaml:/milvus/configs/user.yaml:ro \
  -p 19530:19530 -p 9091:9091 \
  --health-cmd="curl -f http://localhost:9091/healthz" \
  --health-interval=30s --health-start-period=90s --health-timeout=20s --health-retries=3 \
  milvusdb/milvus:v2.5.11 \
  milvus run standalone
```

- `embedEtcd.yaml`：`listen-client-urls: http://0.0.0.0:2379`、`advertise-client-urls: http://0.0.0.0:2379`、`quota-backend-bytes: 4294967296`、`auto-compaction-mode: revision`、`auto-compaction-retention: "1000"`
- `user.yaml`：空覆盖文件（占位，后续加认证时改）
- 端口：**19530**（gRPC，pymilvus 连这个）、9091（healthz/metrics）；2379 不发布也可
- 资源：官方示例 `--memory 4g`；实测空闲约 200-400MB 起步
- 就绪判据：`curl http://localhost:9091/healthz` OK，**start-period 90s**（首启要等）

## 4. 对检索设计的影响（FR-09 决策）

1. **向量来源 = 可解释构图特征（8 维），不是深度 embedding**：
   客户端契约要求「以图搜图」+「以文搜图」。以文搜图需要文本编码器（CLIP text tower ≈ 数百 MB 重依赖），
   违背本项目「镜像最小化 + 无重依赖也能跑」的既定纪律。
   **本期决策**：向量 = 感知 + 评分器已产出的构图特征（rule_of_thirds / subject_scale / balance /
   headroom / lead_room / saliency_center / subject_height / center_offset_x，全部 [0,1] 有界）——
   复用既有链路零新增模型；`query_text` 仅做 pattern/scene_tags 标量过滤，**纯文本检索 → 200 + degraded
   （text_embedding_unavailable）**，诚实降级而非假装能搜。CLIP 双塔列为后续增强项。
2. **度量 COSINE**：特征全部有界非负，余弦即"构图风格相似度"；Milvus 返回 distance，similarity = 1 − distance 后裁剪到 [0,1]（契约要求 Ratio）。
3. **索引 HNSW**（M=16, efConstruction=200, ef=64）：案例库规模 ≤ 数百条，HNSW 查询延迟最低且 standalone 全支持；IVF_FLAT 在小数据上无优势。
4. **降级链路**：pymilvus 未安装 / 连接失败 / collection 缺失 → `/v1/cases/search` 返回 200 + `degraded=true` + 原因枚举，
   **绝不 5xx**（NFR-R1：实时服务"崩掉"比"给不出案例"糟糕得多）。

## 5. 风险与边界（诚实标注）

- [x] **pymilvus 与 2.5.11 服务端版本匹配**：已装 **2.5.18**（`pip install "pymilvus>=2.5,<2.6"`
      刻意封顶——先装出的是 2.6.9，与服务端次版本不一致，官方建议客户端与
      服务端次版本对齐，故降级到 2.5.x）
- [x] **特征提取链路**：`scripts/build_case_index.py --dry-run` **实测跑通**
      ——42 张素材（demo 19 + topic 23）→ 40 张成功提取 8 维向量，
      2 张"未检出主体"**诚实跳过**（不伪造向量入库），耗时 8.1s
- [ ] **Windows 跑 Milvus 本身**：官方支持，但**本机 daemon 至今未起来**——
      非交互会话无法拉起 Docker Desktop 的 GUI 进程树（PowerShell
      `Start-Process`、`cmd start` 均无任何日志/进程痕迹），需人工双击启动
- [ ] **真实端到端**（建库 + 检索 + 相似度排序）：等 daemon 就绪后立即补
- [ ] 本机 `docker compose up api milvus` 全链路互通：待实测
- [ ] CI 侧真实 Milvus service container：设计可行，本轮先本地验证

### 5.1 dry-run 实测暴露的**真实质量边界**（必须写进简历话术）

rule 后端下 `lead_room` 与 `saliency_center` 两维**恒为 0.500**：前者依赖人脸
yaw（rule 后端无人脸估计）、后者依赖显著性图（rule 后端为 None 时 scorer
取中性值 0.5，见 `scorer.py:210`）。即 **8 维中 2 维在 rule 后端退化成常数**，
不携带区分信息；yolo 后端下 8 维全活。

这不是 bug，是"复用既有链路零新增模型"这一决策的**代价**，必须诚实标注：
- 代价表现：rule 后端检索区分度靠其余 6 维；
- 生产环境做法：换成 yolo 后端（有显著图 + 人脸），或直接上 CLIP 双塔
  embedding（维度 512/768，语义更强但要接受重依赖 + 以文搜图能力）。
