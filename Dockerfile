# Dockerfile — mini-infer HTTP serving 镜像
#
# 构建（需要 CUDA 12.1 宿主机）：
#   docker build -t mini-infer .
#
# 运行（dry-run，无需模型权重）：
#   docker run --rm -p 8000:8000 mini-infer
#
# 运行（真实模型，挂载本地权重目录）：
#   docker run --rm --gpus all \
#     -v /path/to/model:/model:ro \
#     -e MINI_INFER_MODEL=/model \
#     -p 8000:8000 mini-infer
#
# 可选环境变量：
#   MINI_INFER_MODEL          模型目录路径（默认 dry-run）
#   MINI_INFER_USE_CUDA_GRAPH 1/true 开启 CUDA Graph
#   MINI_INFER_QUANT_MODE     w8a8 开启 W8A8 量化
#   MINI_INFER_CHUNK_PREFILL_SIZE 256 开启 Chunked Prefill

FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# 先复制依赖定义，利用 layer 缓存
COPY pyproject.toml ./
COPY mini_infer/ ./mini_infer/

# 安装 Python 依赖（serve + 核心，不含 flash-attn，需要额外编译）
RUN pip install --no-cache-dir -e ".[serve]"

# 复制其余文件
COPY . .

EXPOSE 8000

# 默认 dry-run 模式；使用真实模型时通过环境变量 MINI_INFER_MODEL 指定
ENV MINI_INFER_MODEL=dry

CMD ["uvicorn", "mini_infer.serving.server:app", \
     "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
