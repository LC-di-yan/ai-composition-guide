# AI 实时构图指导 Agent —— 运行时镜像
#
# 设计决策（诚实标注）：
#
# 1. **基于 CPU-only torch**：本镜像默认不含 CUDA，体积从 ~6GB 降到 ~1.2GB。
#    理由：容器化的目的是"交付一个可复现的服务"，而推理性能取决于宿主
#    GPU。生产环境若需 GPU，应使用 nvidia/cuda 基础镜像并挂载 GPU
#    （见 docker-compose.yml 的 gpu profile）。
#
# 2. **只装运行必需依赖**：不装 ultralytics（含 torch），改用规则后端兜底。
#    这样镜像小、构建快，且能验证"感知降级链路"在纯净环境下确实可用。
#    需要真实模型时：build --build-arg WITH_MODEL=1
#
# 3. **非 root 运行**：容器以 uid 1000 运行，符合最小权限原则（NFR-S2）。
#
# 4. **多阶段构建**：builder 阶段装编译依赖，runtime 阶段只带运行产物。

# ===========================================================================
# Stage 1: builder
# ===========================================================================
FROM python:3.11-slim AS builder

ARG WITH_MODEL=0
ARG WITH_TEST=0

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 编译期依赖（opencv 的部分轮子需要）
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libglib2.0-0 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

# 先只装依赖，利用 Docker 层缓存：源码变动不会导致依赖重装
WORKDIR /build
COPY requirements.txt pyproject.toml README.md ./
COPY src/aicg/__init__.py src/aicg/__init__.py

# 基础依赖 + 接口层（不含 torch）
RUN pip install --prefix=/install \
        "numpy>=1.26,<3.0" \
        "opencv-python-headless>=4.9" \
        "pydantic>=2.6,<3.0" \
        "PyYAML>=6.0" \
        "loguru>=0.7" \
        "fastapi>=0.110" \
        "uvicorn[standard]>=0.29" \
        "python-multipart>=0.0.9" \
        "openai>=1.30" \
        "httpx>=0.27"

# 可选：真实模型后端（CPU 版 torch，体积约 +800MB）
RUN if [ "$WITH_MODEL" = "1" ]; then \
        pip install --prefix=/install \
            --index-url https://download.pytorch.org/whl/cpu \
            torch torchvision && \
        pip install --prefix=/install "ultralytics>=8.2" ; \
    fi

# 可选：测试依赖（供 compose 的 test profile 使用）
# 不加这个参数时镜像里没有 pytest，`python -m pytest` 会直接
# "No module named pytest" —— 而 compose 的 test profile 正是这么用的。
# 实测测试只额外依赖 pytest（其余 fastapi/numpy/pydantic 已在基础层）。
RUN if [ "$WITH_TEST" = "1" ]; then \
        pip install --prefix=/install "pytest>=8.0" "pytest-cov>=5.0" ; \
    fi

# ===========================================================================
# Stage 2: runtime
# ===========================================================================
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="AI 实时构图指导 Agent"
LABEL org.opencontainers.image.description="边拍边教的多模态实时构图引导服务"
LABEL org.opencontainers.image.version="0.1.0"

# 运行期系统库（opencv-headless 仍需要 libglib）
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1 \
        curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 --shell /bin/bash appuser

COPY --from=builder /install /usr/local

WORKDIR /app
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser configs/ ./configs/
COPY --chown=appuser:appuser scripts/ ./scripts/
COPY --chown=appuser:appuser tests/ ./tests/
COPY --chown=appuser:appuser pyproject.toml README.md ./

# 运行时目录（产物不落在镜像层里，通过 volume 持久化）
RUN mkdir -p /app/models /app/outputs/demo /app/outputs/reports \
    && chown -R appuser:appuser /app/models /app/outputs

ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    AICG_ENV=prod \
    AICG_LOG_LEVEL=INFO

USER appuser
EXPOSE 8000

# 健康检查：/healthz 必须返回 200（NFR-R3 可用性）
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/healthz || exit 1

# 默认启动 API 服务；跑 Demo 时覆盖 command 即可
#
# ⚠️ 必须带 `--factory`：`aicg/api/app.py` 暴露的是**工厂函数** `create_app()`，
# 模块级并没有 `app` 对象。写 `aicg.api.app:app` 会直接报
# `Attribute "app" not found in module`，容器启动即失败。
# 与 `src/aicg/cli/__main__.py` 的 serve 子命令保持同一种入口
# （那里用 uvicorn.run("aicg.api.app:create_app", factory=True)）。
CMD ["uvicorn", "aicg.api.app:create_app", "--factory", \
     "--host", "0.0.0.0", "--port", "8000"]
