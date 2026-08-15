from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from config import MEMORY_BACKEND, MULTI_AGENT, PG_CONNECTION, RETRIEVAL_SCORE_THRESHOLD, WARMUP
from llm_providers import get_chat_llm
from vector_stores import get_vector_store

TOP_K = 5


def _build_checkpointer():
    if MEMORY_BACKEND == "postgres":
        import psycopg
        from langgraph.checkpoint.postgres import PostgresSaver

        # 直连 psycopg 连接（autocommit 是 checkpointer 的要求）；
        # 不用 PostgresSaver.from_conn_string：那个上下文管理器被 GC 后会关掉连接
        conn = psycopg.connect(PG_CONNECTION.replace("+psycopg", ""), autocommit=True)
        saver = PostgresSaver(conn)
        saver.setup()  # 首次运行自动建表
        return saver
    return InMemorySaver()

SYSTEM_PROMPT = """你是 LangChain 开发助手，回答基于 LangChain 官方文档。
1. 技术问题必须先调用 search_docs 工具查证；可多次调用、换关键词重查。
2. 严格依据工具返回的文档片段回答；文档未涵盖的，明确回答"当前文档中未找到"，不要凭记忆补全。
3. 涉及代码时，给出一个可直接运行的最小示例。
4. 先给结论，再给依据，正文控制在 300 字以内（代码除外）。
5. 闲聊、澄清性问题直接回答，不要调用工具。"""

# 审校员 subagent：在隔离上下文中核对主 Agent 的回答草稿，可自行再检索查证
REVIEWER_PROMPT = """你是事实审校员，核对一份基于 LangChain 官方文档的回答草稿。
1. 逐条检查草稿中的技术性论断（API 名称、参数、用法、行为描述）是否有文档依据。
2. 对有疑问的论断，调用 search_docs 工具用英文关键词查证。
3. 只输出审校结论，不要重写回答。第一行只写结论词本身（`通过` 或 `存疑`，不带任何 markdown 标记），从第二行起再写简短说明：
   - 全部有据可依：通过，可附一句简短说明。
   - 发现问题：存疑，并列出具体存疑点及文档中的正确说法。
4. 草稿明确声明"文档中未找到"的部分不属于错误，无需审校。"""

# 主 Agent 开启多 Agent 时的附加指令
MULTI_AGENT_SUFFIX = """
6. 正式输出回答前，先调用 review_answer 工具，把用户问题和你的回答草稿交给审校员核对；
   审校员指出问题时，修正后输出最终回答。闲聊类问题跳过审校。"""


def build_agent():
    store = get_vector_store()
    if WARMUP:
        # 主线程预热：初始化 Chroma 客户端并完成首次嵌入推理。
        # 否则首次嵌入发生在 langgraph 工具线程里，会触发 chromadb SharedSystemClient
        # 竞态 KeyError，或 torch 在 macOS 子线程首次推理时段错误
        store.get_stats()
        store.similarity_search("warmup", k=1)

    @tool(response_format="content_and_artifact")
    def search_docs(query: str) -> tuple[str, list[dict]]:
        """搜索 LangChain 官方文档。输入英文技术关键词查询效果最好。返回相关文档片段及其来源。"""
        results = store.similarity_search(
            query, k=TOP_K, score_threshold=RETRIEVAL_SCORE_THRESHOLD
        )
        if not results:
            return "未检索到相关文档。", []
        content = "\n\n---\n\n".join(
            f"[来源: {r['source']} | 章节: {r['metadata'].get('header_path', '')}]\n{r['content']}"
            for r in results
        )
        return content, results

    llm, chat_provider = get_chat_llm(temperature=0)

    tools = [search_docs]
    system_prompt = SYSTEM_PROMPT

    if MULTI_AGENT:
        # 多 Agent 协作（tool-per-agent 模式）：审校员是独立 Agent，
        # 包装成工具交给主 Agent 编排；子 Agent 无 checkpointer，上下文隔离
        reviewer = create_agent(
            model=llm,
            tools=[search_docs],
            system_prompt=REVIEWER_PROMPT,
        )

        @tool("review_answer", description="把回答草稿交给审校员做事实核查。参数为原始问题与草稿全文，返回审校结论（通过/存疑点）。")
        def review_answer(question: str, draft: str) -> str:
            result = reviewer.invoke({
                "messages": [{"role": "user", "content": f"用户问题：{question}\n\n待审校的回答草稿：\n{draft}"}]
            })
            return str(result["messages"][-1].content)

        tools.append(review_answer)
        system_prompt += MULTI_AGENT_SUFFIX

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
        checkpointer=_build_checkpointer(),
    )
    return agent, chat_provider, store
