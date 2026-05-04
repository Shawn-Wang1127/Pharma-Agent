# 使用官方 Python 3.10 轻量级镜像
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 安装底层 C++ 编译环境（防止 ChromaDB 等生信库编译失败）
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

# 复制依赖文件并安装
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目所有文件到工作目录
COPY . .

# 预先创建系统运行必须的输出目录；向量库和 PDF 数据应通过受控挂载提供
RUN mkdir -p clinical_data

# 暴露 FastAPI 运行端口
EXPOSE 8000

# 启动 Uvicorn 服务器
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
