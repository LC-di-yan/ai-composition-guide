# Milvus Windows 本地方案调研（M6-2 前置）

> 日期：2026-09-29　目的：为 FR-09 案例检索选定本机可**真实运行**的 Milvus 形态。
> 纪律约束：绝不重演「Docker 标已备未实测」——**检索模块必须在真实 Milvus 上验证后才允许标已实现**。

## 1. 结论（2026-09-29 修订）

**生产/容器路径选 Docker Standalone 单容器，镜像 `milvusdb/milvus:v2.6.24`，客户端 `pymilvus==2.6.17`。**
**本机开发/演示路径启用 Milvus Lite（`milvus-lite 3.x` 本地 `.db` 文件）——真跑已验证可用。**

> **修订说明（重要）**：初版调研判定"Milvus Lite 不支持 Windows"，依据是
> milvus-lite **2.4.x 时代**（内嵌 ~200MB C++ 二进制、仅 Linux/macOS  wheel）。
> 2026-09-29 复测发现 PyPI 上已只有 **3.0~3.2.1** 版本：该版本改为
> **纯 Python 实现**（依赖 `faiss-cpu` + `grpcio` + `pyarrow`，wheel 是
> `py3-none-any`），而 `faiss-cpu` 提供官方 `win_amd64` 轮子 —— 于是
> **Windows 上真能跑起来**（证据见 §6）。§2 表 A 行据此更正。
> 旧结论在当时是正确的，现在过期了；按 honesty 纪律，此处明确记录**结论变更**。

## 2. 候选方案对比

| 方案 | Windows 支持 | 版本 | 与生产一致性 | 判定 |
| --- | --- | --- | --- | --- |
| **A. Milvus Lite**（`MilvusClient("./x.db")` 内嵌） | ⚠️ **结论已变更**（见下方注）。2.4.x 时代不支持；**3.x 改成纯 Python 后在 Windows 实测可用**，但官方支持矩阵仍只写 Linux/macOS——属"能用但不受支持" | **3.0~3.2.1**（PyPI 上已无 2.4.x） | 中——同一套 pymilvus API，但引擎是单进程 Lite | **本机开发/演示/CI 选定**（详细证据见 §6）；生产仍走 C |
| **B. 旧版 pip 包**（`milvus==2.2.16` + `milvus-server` 命令） | ✅ 可跑 | **2.2.x（2023 年）** | 低——需锁死 `pymilvus==2.2.13` + `marshmallow<4`（CSDN 实录），无 `MilvusClient` 新 API | 排除——为本地省一个 Docker 换来一套锁死的旧依赖，作品集价值为负 |
| **C. Docker Standalone 单容器**（内嵌 etcd + local storage） | ✅ 官方文档明确支持（milvus.io「Run Milvus in Docker (Windows)」，Docker Desktop + WSL2）——**但本机 daemon 起不来**（环境策略拦截 `wsl.exe`，见 §6.1） | **2.6.24** | **高**——与未来部署同版本 | **生产部署选定**；本机不可用于实测 |

> A 行注释：初版调研写"❌ 不支持、Windows 装不了 `milvus-lite`"，依据是
> **2.4.x 的内嵌 C++ 二进制**。2026-09-29 复测：PyPI 上只剩 3.0~3.2.1，
> wheel 是 `py3-none-any`（纯 Python），依赖 `faiss-cpu`（有官方 `win_amd64`
> 轮子）+ `grpcio` + `pyarrow`，且 `server_manager.py` 里明写"Unlike v1
> (which spawns a ~200MB C++ subprocess), this runs entirely in-process as
> pure Python threads"。**旧结论在当时正确、现已过期**，此处记录变更。

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
  milvusdb/milvus:v2.6.24 \
  milvus run standalone
# 版本与 docker-compose.yml 保持一致；本机需先把 -v 源目录建好（Windows 上 Docker
# 不可用，见 §6.1，本配方的执行情况因此在本地**未实测**）
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

- [x] **pymilvus 与服务端次版本对齐**：现锁 **2.6.17**，对应服务端 **v2.6.24**
      ——依据 Milvus GitHub Releases 页：v2.6.24 的 Release Note 标注其
      Python SDK = 2.6.17。（上一轮的 2.5.18 + v2.5.11 组合已被替换，原因：
      Lite 3.x 的 search 协议要求 `function_score` 字段，pymilvus 2.5.x 不发
      该字段，握手能过、一 search 就报错；为保持 client/server 次版本一致，
      两端一起升到 2.6 线）
- [x] **特征提取链路**：`scripts/build_case_index.py` **实测跑通**
      ——42 张素材（demo 19 + topic 23）→ 40 张成功提取 8 维向量，
      2 张"未检出主体"**诚实跳过**（不伪造向量入库），耗时 6.7~9.7s
- [x] **真实端到端**（建库 + 检索 + 相似度排序 + 标量过滤 + 跨进程读取）：
      **已在真实 Milvus 引擎上验证**（本机 Milvus Lite；CI 侧同款 Lite 作业
      亦接入，见 §6.3）。证据：`tests/test_retrieval_integration.py` 6 项
- [x] **CI 侧真实 Milvus**：不是 service container，而是 **Milvus Lite
      单进程**（Linux 官方支持），CI job `retrieval` 真建库 + 真检索
- [ ] **Windows 跑 Docker 桌面版**：**已确认本机不可行**——底层 `wsl.exe`
      被执行环境的程序黑名单拦截，属环境策略问题而非配置问题（§6.1）
- [ ] 本机 `docker compose up api milvus` 全链路互通：受上一条阻塞，仍未实测
- [ ] **镜像 v2.6.24 本身未真跑**：本机无 Docker daemon，该标签存在性经
      GitHub Releases 页核实，但 compose 起来的行为未经实测，换 tag 需重验

### 5.1 dry-run 实测暴露的**真实质量边界**（必须写进简历话术）

rule 后端下 `lead_room` 与 `saliency_center` 两维**恒为 0.500**：前者依赖人脸
yaw（rule 后端无人脸估计）、后者依赖显著性图（rule 后端为 None 时 scorer
取中性值 0.5，见 `scorer.py:210`）。即 **8 维中 2 维在 rule 后端退化成常数**，
不携带区分信息；yolo 后端下 8 维全活。

这不是 bug，是"复用既有链路零新增模型"这一决策的**代价**，必须诚实标注：
- 代价表现：rule 后端检索区分度靠其余 6 维；
- 生产环境做法：换成 yolo 后端（有显著图 + 人脸），或直接上 CLIP 双塔
  embedding（维度 512/768，语义更强但要接受重依赖 + 以文搜图能力）。

---

## 6. 本章实测记录（2026-09-29，"真跑"取代"标已备"）

上一版停在"Docker daemon 起不来 → 端到端待补"。这一节记录**追到根因、
换路打到真实 Milvus、并由此挖出 4 个真缺陷**的全过程。

### 6.1 Docker Desktop 起不来：不是配置，是环境策略

排查路径（每一步都有证据，不是猜）：

1. 无 `com.docker.backend.exe` 进程，日志停在 4 天前 → 说明根本没起来过；
2. 直接拉 `com.docker.backend.exe` 才第一次看到真正报错：

   ```
   starting services: initializing Inference manager:
     remove C:\Users\86134\AppData\Local\Docker\run\dockerInference: Access is denied
   c:\windows\system32\wsl.exe --version failed: fork/exec …: Access is denied
   monitor exited: exit status 150
   ```

3. `AppData\Roaming\Docker\settings-store.json` 里 `"EnableDockerAI": true`
   → 启动必须初始化 Inference manager（Docker AI），第一步就是清那个
   9-25 崩溃残留的 0 字节 socket 文件。已将该项改为 `false`（改动前备份为
   `settings-store.json.bak-20260929`）以绕开这条路径；
4. 改完仍退出，最终命中真正的拦截：**执行环境把 `wsl.exe` 列入程序黑名单**，
   而 Windows 版 Docker 的底座就是 WSL2 —— 该拦截无法从当前会话批准或绕过：

   ```
   PROGRAM BLOCKED BY SECURITY POLICY - The sandbox prevented a program on the
   configured Program Blacklist from starting: wsl.exe
   ```

**结论**：本机 Docker 不可用是**环境策略限制**，不是配置错误、也不是我能
从会话内部解决的事（需用户在「安全中心 → 命令安全 → 程序黑名单」移除）。
因此停止重试，改为寻找**不依赖守护进程**的真跑路径。

### 6.2 换路：Milvus Lite 3.x 在 Windows 上真能跑

关键发现：milvus-lite **3.x 是纯 Python 实现**（wheel 为 `py3-none-any`，
依赖 `faiss-cpu` + `grpcio` + `pyarrow`；`faiss-cpu` 有官方 `win_amd64`
轮子），与 2.4.x 的内嵌 C++ 二进制是两回事。本机实测结果：

| 项 | 结果 |
| --- | --- |
| 建 collection（HNSW / COSINE / dim=8） | ✅ 2.7~3.8s |
| upsert 50 条 → `row_count` | ✅ 50 |
| search top3 相似度 | ✅ 0.9739 / 0.9534 / 0.9402（降序） |
| `pattern == "center"` 标量过滤 | ✅ 结果全为 center |
| 4 维向量写入 8 维 collection | ✅ 被引擎拒绝（维度契约生效） |
| 项目素材真建库（42 张 → 40 条） | ✅ 入库 40 / 库内 40 |

代价有三条裂缝，都已沉淀到代码或文档，**必须讲清楚**：

1. **初始化顺序敏感 → SIGSEGV**。若 Milvus 客户端的**首次初始化**发生在
   帧处理链路（OpenCV + 感知/评分）之后，进程会在 Lite 的
   `pa.RecordBatch.from_pydict` 写入路径上访问违规崩溃。faulthandler 抓到
   的现场：`milvus_lite/engine/collection.py:1923 _build_wal_data_batch`。
   对照实验：

   | 顺序 | 结果 |
   | --- | --- |
   | 图像链路 → 首次接触 Lite（5 张图即可） | ❌ exit 139 SIGSEGV |
   | 仅 cv2 解码 42 张 → Lite | ✅ 正常（排除"图像解码"猜测） |
   | 仅分配 288MB numpy → Lite | ✅ 正常（排除"内存占用"猜测） |
   | **先接触 Lite → 图像链路 → 再 Lite** | ✅ 正常 |

   → 工程对策：`CaseSearchService.warmup()` + 应用 lifespan 在**跑任何帧之前**
   完成一次握手；建库脚本 likewise 把 `ensure_collection()` 提到提取之前。
   注意这只是**崩溃规避**，不是功能保证：预热失败照样走降级链。
2. **跨进程读取必须 load**。写入进程退出后 collection 处于 `released`，
   下一个进程只读时 search 报 `code=101: call load() before search`——表现为
   "库里 40 条却永远搜不到"。已修：`ensure_collection()` 增加 `_ensure_loaded()`
   （由本轮全量套件暴露）。
3. **`drop()` 在 Windows 上会炸**。Lite 落盘用 POSIX 语义的
   `os.rename(tmp, manifest.json)`，目标存在时抛 `WinError 183`；集成测试用例
   因此改为"每个用例一个独立 `.db` 文件"而不是靠 drop 清理。
   → 重建请用新路径，别依赖 `--rebuild` 的 drop。
4. **`distance` 字段的语义会被小版本翻转**（由 CI job `retrieval` 首跑抓出）。
   同一套 COSINE 检索：

   | milvus-lite | 相同向量 | 正交向量 | 口径 |
   | --- | --- | --- | --- |
   | **3.0** | `distance = 0.0000` | `distance = 1.0000` | `distance = 1 − 余弦` |
   | **3.2.1** | `distance = 1.0000` | `distance = 0.0000` | `distance = 余弦本身` |

   于是"自身向量"在 3.2.1 上按 `1 - distance` 解读会得到 **0.0** —— 同一份代码
   换个小版本，相似度**方向翻转**。对策：相似度改为**取回入库向量在本地按余弦
   重算**（`MilvusStore._cosine_similarity`），排序也由我们自己做，`distance`
   只留作观测；口径与本地余弦明显不符时记 WARNING。
   **教训**：交付契约里的数值，不要交给第三方字段的"约定语义"去定义。

**版本配套的连带结论**：Lite 3.x 的 search 请求需要 `function_score` 字段，
pymilvus 2.5.x 不发该字段（握手能过、一检索就报错）。为不破坏自己立的
"client/server 次版本对齐"规矩，客户端升到 **2.6.17**、compose 镜像同步到
**v2.6.24**（后者存在性经 GitHub Releases 页核实，但**未真跑**）。

### 6.3 由此固化成 CI 能力

新增 CI job `retrieval`（Ubuntu）：装 `pymilvus==2.6.17` +
`milvus-lite>=3.0,<3.3` → 用仓库内真实素材跑 `build_case_index.py` 建库 →
跑 `tests/test_retrieval_integration.py`。意义是把"真·Milvus 引擎验证"从
"本机一次性手工验证"升级为**任何人都能复现的自动化事实**；同时因为跑在
Linux 上，用的是 Lite 的**官方支持范围**，不依赖 §6.2 那些 Windows 裂缝。
（本机 Windows 上的同款验证属"能用但不受官方支持"，刻意不进 CI 承诺。）
