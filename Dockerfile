# 使用 Python 3.11 作为基础镜像
FROM python:3.11-slim

# 在容器内创建并切换到/app目录，后续所有命令（如COPY、RUN）都会在这个目录下执行，相当于 “项目根目录”
WORKDIR /app

# 设置环境变量
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# 安装系统依赖（可选，用于某些 PDF 处理）
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 配置 pip 使用多个镜像源（阿里云、清华、官方）
RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/ && \
    pip config set global.extra-index-url "https://mirrors.aliyun.com/pypi/simple/ https://pypi.tuna.tsinghua.edu.cn/simple/"

# 安装 Python 依赖（添加超时设置和重试）
RUN pip install --no-cache-dir \
    --default-timeout=300 \
    --retries=3 \
    -r requirements.txt

# 复制项目代码
COPY main.py chain.py fileload.py ./

# 暴露 Gradio 默认端口
EXPOSE 7860

# 启动命令
CMD ["python", "main.py"]

