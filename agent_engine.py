import os
import logging
import json
import multiprocessing as mp
from queue import Empty
from typing import Annotated, TypedDict
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

from core import BioAssistant

load_dotenv()

EXECUTION_TIMEOUT_SECONDS = 10
MAX_STDOUT_CHARS = 4000
MAX_TRACEBACK_CHARS = 6000


def _truncate_text(text: str, limit: int, label: str) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n...[{label} truncated to {limit} chars]"


def _execute_python_code_worker(code: str, result_queue) -> None:
    import io
    import sys
    import traceback
    import pandas as pd
    import numpy as np

    old_stdout = sys.stdout
    redirected_output = io.StringIO()
    sys.stdout = redirected_output
    try:
        local_vars = {"pd": pd, "np": np}
        exec(code, {}, local_vars)
        output = redirected_output.getvalue()
        result_queue.put(("ok", output))
    except Exception:
        result_queue.put(("error", traceback.format_exc()))
    finally:
        sys.stdout = old_stdout

# ==========================================
# 1. Industrial Logging Configuration
# ==========================================
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

logger = logging.getLogger("PharmaAgent")
logger.setLevel(logging.INFO)
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter('\033[92m[%(asctime)s] [%(levelname)s] %(message)s\033[0m', datefmt='%H:%M:%S')
    )
    logger.addHandler(console_handler)

# ==========================================
# 2. State Definitions & System Prompts
# ==========================================
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

SYSTEM_PROMPT = """你是一个严谨的生物医药 AI 智能体。你具备调度多个专业检索工具与代码沙盒的能力。
请严格遵守以下纪律：
1. 双引擎协同：本地医学知识库 (search_medical_literature) 与 PubMed (search_pubmed_literature) 视需求调度。
2. 数据分析双步法则（核心纪律）：当需要分析 CSV 数据时，你必须遵守两步走战略：
   - 第一步：必须先调用 preview_csv_data 获取真实的列名和数据样本。
   - 第二步：根据上一步获取的真实列名，调用 execute_python_code 编写 pandas 脚本进行计算。
   绝不允许在没有探查真实列名的情况下凭空猜测并编写数据分析代码。

3. 【高危-致命乱码防御】可视化与字体铁律：
   若需画图，必须遵守以下全栈工程纪律，彻底杜绝在 Docker(Linux) 封闭环境中因缺少字体导致的中文乱码（方块）：
   - **强制全英文标注**：图表标题 (title)、坐标轴标签 (ylabel, xlabel)、图例 (legend)、统计文本框 (bbox) 等所有视觉文本**必须全部使用英文**。严禁出现任何中文或非标准 Unicode 字符。
   - **显式字体配置范式**：在绘图代码初始化部分，必须显式设置标准字体族。强制采用以下配置范式（该字体默认包含在 python-slim 基础镜像中）：
     ```python
     import pandas as pd
     import matplotlib.pyplot as plt
     import matplotlib
     matplotlib.use('Agg')  # Docker 必需
     
     # Explicitly configure standard English font family to prevent square character corruption
     matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
     matplotlib.rcParams['axes.unicode_minus'] = False  
     ```

4. 强制溯源：不捏造数据，引用文献须附上来源或 PMID。
5. 可视化排版纪律(防重叠)：若需生成带有统计信息文本框(bbox)的柱状图,代码必须具备动态布局意识：
   - 第一选择：严禁使用绝对坐标定位文本框。应使用 `fig.subplots()` 创建双子图排版，将统计文本专门放置在左侧或右侧的子图中，将图表放置在另一侧。
   - 第二选择（若必须在单图中显示）：代码必须动态计算数据最大值（如 `max(means) * 1.5`），并将 `plt.ylim` 设置得足够高，为顶部的文本框预留出至少 30% 的空白缓冲区，避免文字标签与 bbox 重叠。"""

# ==========================================
# 3. Core Agent Engine Encapsulation
# ==========================================
class PharmaAgentEngine:
    def __init__(self):
        logger.info("[SYSTEM] Initializing PharmaAgentEngine and underlying RAG instances...")
        self.rag_engine = BioAssistant()
        self.app = self._build_graph()

    def _build_graph(self):
        clinical_data_root = os.path.abspath("clinical_data")

        def _resolve_clinical_csv_path(filepath: str):
            if not isinstance(filepath, str) or not filepath.strip():
                return None, "Only CSV files under clinical_data are allowed."

            normalized_input = os.path.normpath(filepath.strip())
            candidate_path = normalized_input
            if not os.path.isabs(candidate_path):
                candidate_path = os.path.abspath(candidate_path)

            try:
                common_root = os.path.commonpath([candidate_path, clinical_data_root])
            except ValueError:
                return None, "Only CSV files under clinical_data are allowed."

            if common_root != clinical_data_root:
                return None, "Only CSV files under clinical_data are allowed."

            if not candidate_path.lower().endswith(".csv"):
                return None, "Only CSV files under clinical_data are allowed."

            relative_path = os.path.relpath(candidate_path, clinical_data_root)
            path_parts = relative_path.split(os.sep)
            if any(part.startswith(".") for part in path_parts):
                return None, "Hidden files are not allowed."

            if os.path.basename(candidate_path).startswith("."):
                return None, "Hidden files are not allowed."

            if not os.path.isfile(candidate_path):
                return None, "CSV file not found under clinical_data."

            return candidate_path, None

        @tool
        def search_medical_literature(query: str) -> str:
            """Retrieves context from the local, high-security medical knowledge base (ChromaDB)."""
            logger.info(f"[TOOL] Executing [search_medical_literature] for: {query}")
            final_answer = ""
            final_sources = []
            for partial_answer, sources in self.rag_engine.rag_chat_stream(query):
                final_answer = partial_answer
                final_sources = sources
            seen = set()
            source_list = []
            for doc in final_sources:
                src = doc.metadata.get('source', 'Unknown')
                if src not in seen:
                    source_list.append(src)
                    seen.add(src)
            formatted_sources = "\n".join([f"- {s}" for s in source_list])
            return f"Local Retrieval Conclusion:\n{final_answer}\n\nSources:\n{formatted_sources}"

        @tool
        def search_pubmed_literature(query: str, max_results: int = 3) -> str:
            """Interacts with the NCBI Entrez API to fetch real-time global biomedical literature."""
            from Bio import Entrez, Medline
            import io
            import ssl
            Entrez.email = "pharma_agent_developer@example.com"
            ssl._create_default_https_context = ssl._create_unverified_context
            logger.info(f"[TOOL] Executing [search_pubmed_literature] for query: {query}")
            try:
                handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results, sort="pub date")
                record = Entrez.read(handle)
                handle.close()
                id_list = record["IdList"]
                if not id_list:
                    return f"No PubMed records found for '{query}'. Please refine your boolean query."
                fetch_handle = Entrez.efetch(db="pubmed", id=id_list, rettype="medline", retmode="text")
                medline_data = fetch_handle.read()
                fetch_handle.close()
                records = Medline.parse(io.StringIO(medline_data))
                results = []
                for rec in records:
                    title = rec.get("TI", "No Title")
                    abstract = rec.get("AB", "No Abstract available.")
                    pmid = rec.get("PMID", "Unknown")
                    results.append(f"PMID: {pmid}\nTitle: {title}\nAbstract: {abstract}\n---")
                return "\n".join(results)
            except Exception as e:
                logger.error(f"[TOOL] PubMed API failure: {str(e)}")
                return f"PubMed API invocation failed. Check network or query syntax: {str(e)}"
        
        @tool
        def preview_csv_data(filepath: str) -> str:
            """Extracts structural metadata (schema, dtypes) and data samples from a CSV file."""
            import pandas as pd
            safe_name = os.path.basename(filepath) if isinstance(filepath, str) and filepath.strip() else "unknown"
            logger.info(f"[TOOL] Executing [preview_csv_data] for: {safe_name}")
            try:
                resolved_path, error_message = _resolve_clinical_csv_path(filepath)
                if error_message:
                    return error_message

                df = pd.read_csv(resolved_path)
                info = f"[Dataset Overview] {os.path.basename(resolved_path)}\nRows: {len(df)} | Columns: {len(df.columns)}\n\n"
                info += f"[Schema & Data Types]\n{df.dtypes.to_string()}\n\n[Data Sample (Top 3 rows)]\n{df.head(3).to_string()}"
                return info
            except Exception as e:
                logger.error("[TOOL] preview_csv_data failed during CSV inspection.")
                return "Failed to probe CSV file under clinical_data."
            
        @tool
        def execute_python_code(code: str) -> str:
            """Executes dynamically generated Python scripts in a controlled REPL-style runner."""
            logger.info("[TOOL] Executing [execute_python_code] sandbox instance...")
            ctx = mp.get_context("spawn")
            result_queue = ctx.Queue()
            process = ctx.Process(
                target=_execute_python_code_worker,
                args=(code, result_queue),
            )
            process.start()
            process.join(EXECUTION_TIMEOUT_SECONDS)

            if process.is_alive():
                logger.error("[TOOL] Python execution timed out.")
                process.terminate()
                process.join()
                return f"Execution timed out after {EXECUTION_TIMEOUT_SECONDS} seconds. Please simplify the code and try again."

            try:
                status, payload = result_queue.get(timeout=1)
            except Empty:
                logger.error("[TOOL] Python execution finished without returning a result.")
                return "Execution failed before producing a result. Please revise the code and try again."
            finally:
                result_queue.close()
                result_queue.join_thread()

            if status == "ok":
                output = payload
                if not output.strip():
                    return "Execution completed successfully, but no output was intercepted. Ensure print() statements are used."
                truncated_output = _truncate_text(output, MAX_STDOUT_CHARS, "stdout")
                if len(truncated_output) != len(output):
                    return f"Execution successful. Stdout:\n{truncated_output}"
                return f"Execution successful. Stdout:\n{output}"

            logger.error("[TOOL] Sandbox exception raised.")
            truncated_error = _truncate_text(payload, MAX_TRACEBACK_CHARS, "traceback")
            return (
                "Execution encountered an exception. Analyze the traceback and revise the code:\n"
                f"{truncated_error}"
            )

        tools = [search_medical_literature, search_pubmed_literature, preview_csv_data, execute_python_code]
        
        # Synchronous LLM initialization
        llm = ChatOpenAI(
            model='deepseek-chat', 
            openai_api_key=os.getenv("DEEPSEEK_API_KEY"), 
            openai_api_base=os.getenv("DEEPSEEK_BASE_URL")
        )
        llm_with_tools = llm.bind_tools(tools)

        def agent_node(state: AgentState):
            """Core routing node for synchronous LLM inference and tool planning."""
            logger.info("[AGENT] Evaluating state context and generating next actions...")
            messages = state["messages"]
            if not messages or getattr(messages[0], "type", "") != "system":
                messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
            response = llm_with_tools.invoke(messages)
            return {"messages": [response]}

        workflow = StateGraph(AgentState)
        workflow.add_node("agent", agent_node)
        workflow.add_node("tools", ToolNode(tools))
        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges("agent", tools_condition)
        workflow.add_edge("tools", "agent")

        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)

    def chat(self, question: str, thread_id: str = "default_session") -> str:
        """
        Synchronous execution method. Blocks until the LangGraph state machine completes.
        """
        logger.info(f"[API] Processing synchronous request for thread: {thread_id}")
        initial_state = {"messages": [("user", question)]}
        config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": 15 
        }
        
        final_state = self.app.invoke(initial_state, config=config)
        if final_state and "messages" in final_state:
            return final_state["messages"][-1].content
        return "System encountered an error generating a response."
