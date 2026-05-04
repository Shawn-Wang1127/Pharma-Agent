# Pharma-Agent (原 Pharma-RAG Assistant)

本项目是一个 biomedical AI agent prototype，围绕本地 RAG、PubMed 检索、CSV 数据分析和 FastAPI 服务层组织，当前具备以下四类主要能力：

1. **本地 RAG 引擎 (Local RAG)**
   * **功能描述**：基于本地部署的 BAAI/bge-m3 向量模型与 ChromaDB，对医学文献或临床试验材料提供离线语义检索与来源追踪。
   * **应用场景**：解答特定靶点机制、提炼临床试验核心结论。

2. **PubMed 文献检索 (PubMed Retrieval)**
   * **功能描述**：集成 Biopython 并调度 NCBI Entrez API，支持按查询抓取公开的生物医学文献并生成摘要式结果。
   * **应用场景**：横向对比本地结论与全球前沿真实世界数据，交叉验证预后因素（如 TP53 共突变）的一致性。

3. **CSV 数据分析工具链 (Analytics Sandbox)**
   * **功能描述**：提供受控的 Python 执行工具和 CSV 预览能力，方便读取结构化临床数据后编写 Pandas/SciPy 脚本完成统计计算，并用 Matplotlib 生成图表。
   * **工程亮点**：工具链支持基于真实列名的分析流程，并在出错时把 traceback 返回给智能体继续修正；生成的图表可通过 FastAPI 静态路由访问，便于前后端协同查看。

4. **FastAPI + LangGraph 服务层**
   * **功能描述**：底层基于 LangGraph 状态机构建，暴露标准化 RESTful API (`POST /agent/chat`)。通过 `session_id` 区分会话上下文，适合单进程内的多轮交互演示。
   * **部署方案**：提供 Docker 容器封装，尽量减少 Python 依赖和本地编译环境差异带来的部署摩擦。
---

## 🛠️ 全景技术栈 (Technology Stack)

系统采用较为清晰的模块化组合，底层组件均为开源库：

* 🧠 **AI 与智能体编排**: `LangGraph`, `LangChain`, `DeepSeek-V3` (Tool Calling)
* 🗄️ **向量检索 (RAG)**: `ChromaDB`, `BAAI/bge-m3` (本地私有化 Embedding)
* 📊 **数据科学与计算**: `Pandas`, `NumPy`, `SciPy`, `Matplotlib`
* 🌐 **外部 API 集成**: `Biopython` (NCBI PubMed Entrez API)
* ⚙️ **后端与服务化**: `FastAPI`, `Uvicorn`, `Pydantic`
* 🐳 **基础设施**: `Docker` (全栈容器化)

---

## 🚀 快速开始 (Quick Start with Docker)

本项目提供 Docker 化运行方式，便于在一致的环境中启动服务和演示原型能力。

### 1. 前置准备
在首次运行前，请确保生成本地临床测试数据（Agent 数据分析沙盒的运行底座）：
```bash
python generate_mock_data.py
```

### 2. 构建与运行
请确保本地已安装运行 [Docker Desktop](https://www.docker.com/products/docker-desktop/)，在终端执行：
```bash
# 构建 Docker 镜像
docker build -t pharma-agent:v3.1 .

# 启动服务并映射至本地 8000 端口
docker run -p 8000:8000 pharma-agent:v3.1
```

构建镜像时，`.dockerignore` 会排除本地 `.env`、向量库、PDF 数据、缓存与生成图片；如需 API 密钥或私有数据，请在运行容器时通过环境变量或受控挂载方式提供，不要打包进镜像。本地向量库需预构建后挂载，或在容器内按需另行构建。

### 3. API 接口调试
容器启动成功后，访问交互式 API 文档 (Swagger UI)：
👉 **`http://127.0.0.1:8000/docs`**

Agent 生成数据分析图表后可通过静态路由直接访问，例如：
👉 **`http://127.0.0.1:8000/clinical_data/api_test_chart.png`**

*(注：以下为 V2.0 阶段 FastAPI 基础接口运行演示视频)*

[https://github.com/user-attachments/assets/211631dc-3af6-4e0e-b46e-2273bac580b8](https://github.com/user-attachments/assets/211631dc-3af6-4e0e-b46e-2273bac580b8)

---

## 🔌 API 调用规范 (API Reference)

系统提供标准的 RESTful API 接口，引入了 `session_id` 以支持多用户并发隔离。

**Endpoint:** `POST /agent/chat`

**请求示例 (触发三大引擎的终极测试指令):**

```json
{
  "question": "1. 查阅本地知识库，简述MARIPOSA试验中TP53的影响。2. 读取 clinical_data/lung_cancer_mock_data.csv，计算携带TP53共突变与未携带者的平均OS差异，用 matplotlib 画出图表并保存为 clinical_data/api_test_chart.png (请使用英文标注)。3. 去 PubMed 检索最新全网文献，与本地计算结果对比总结。",
  "session_id": "api_integration_test_01"
}
```

**响应示例:**

```json
{
  "answer": "## 综合研究报告...\n\n### 1. 本地文献分析\n根据 MARIPOSA 试验...\n\n### 2. 数据验证与可视化\n计算得出 TP53 共突变患者平均 OS 劣势为 3.1 个月。可视化图表已保存至 clinical_data/api_test_chart.png...\n\n### 3. 全球最新进展 (PubMed)\n根据检索到的最新文献 (PMID: 41510380)...",
  "session_id": "api_integration_test_01"
}
```

---

## 🖥️ 附加测试工具：Gradio UI (Legacy V1.0)

本项目保留了早期 V1.0 版本的交互式 Web UI，提供基础的 RAG 流式输出体验，不包含 Agent 路由与代码执行工具。该模式可脱离 Docker 运行，主要用于早期验证和算法测试。

[https://github.com/user-attachments/assets/50fdceb4-10a0-49ac-9d7a-a1b6283ac31d](https://github.com/user-attachments/assets/50fdceb4-10a0-49ac-9d7a-a1b6283ac31d)

### 运行方式
```bash
python app.py
```
启动后访问 `http://127.0.0.1:7860` 即可进入图形化问答界面。
