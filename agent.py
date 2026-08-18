from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from config import MEMORY_BACKEND, PG_CONNECTION, RETRIEVAL_SCORE_THRESHOLD, WARMUP
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

# 查询规划员 subagent：把复杂问题拆成多个英文子查询，扩大召回覆盖面
PLANNER_PROMPT = """你是查询规划员，为检索 LangChain 英文技术文档设计子查询。
1. 把用户问题分解为 2~4 个英文子查询，每个聚焦一个方面（概念、API 用法、配置、错误处理等）。
2. 保留问题中的专有名词（如 checkpointer、middleware、create_agent），子查询用英文关键词短语，不要完整句子。
3. 简单的单一问题不需要分解，原样输出一个英文查询即可。
4. 只输出子查询列表，每行一个，不编号、不解释、不输出其他内容。"""

# 审校员 subagent：在隔离上下文中核对主 Agent 的回答草稿，可自行再检索查证
REVIEWER_PROMPT = """你是事实审校员，核对一份基于 LangChain 官方文档的回答草稿。
1. 逐条检查草稿中的技术性论断（API 名称、参数、用法、行为描述）是否有文档依据。
2. 对有疑问的论断，调用 search_docs 工具用英文关键词查证。
3. 只输出审校结论，不要重写回答。第一行只写结论词本身（`通过` 或 `存疑`，不带任何 markdown 标记），从第二行起再写简短说明：
   - 全部有据可依：通过，可附一句简短说明。
   - 发现问题：存疑，并列出具体存疑点及文档中的正确说法。
4. 草稿明确声明"文档中未找到"的部分不属于错误，无需审校。
5. 代码示例同样要核对：用到的符号都有对应 import（如用了 Command 就要有导入）、参数名与文档中的签名一致、
   调用方式与文档示例不矛盾。代码错误同样属于实质性错误，判存疑并指出正确写法。
6. 只有实质性错误才判存疑：API 名/参数/行为描述错误、与文档相悖的建议（如推荐文档已标注弃用的方案）、
   把无文档依据的内容当作"文档建议"的事实性断言。措辞不精确、对文档的合理引申、
   以及明确是建议性而非文档断言的表述（如"社区有第三方实现"），不算错误——可在说明中提及，但结论仍为通过。"""

# 主 Agent 开启多 Agent 时的附加指令
MULTI_AGENT_SUFFIX = """
6. 复杂或涉及多个方面的问题，先调用 plan_queries 拆成英文子查询，再对每个子查询调用 search_docs 检索，
   综合所有结果作答；简单问题直接 search_docs 即可。
7. 正式输出回答前，先调用 review_answer 工具，把用户问题和你的回答草稿交给审校员核对；
   审校员指出问题时，修正后输出最终回答。闲聊类问题跳过规划与审校。"""


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
    checkpointer = _build_checkpointer()  # 两个 Agent 共享：开关切换时会话历史无缝衔接

    agent = create_agent(
        model=llm,
        tools=[search_docs],
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
    )

    # 多 Agent 协作（tool-per-agent 模式）：规划员/审校员是独立 Agent，
    # 包装成工具交给主 Agent 编排；子 Agent 无 checkpointer，上下文隔离。
    # 始终构建，由请求级开关决定是否使用（构图无网络开销）
    planner = create_agent(
        model=llm,
        tools=[],
        system_prompt=PLANNER_PROMPT,
    )

    @tool("plan_queries", description="把复杂问题分解为多个英文检索子查询，每行一个。适合涉及多个方面或检索覆盖不全的问题。")
    def plan_queries(question: str) -> str:
        result = planner.invoke({
            "messages": [{"role": "user", "content": question}]
        })
        return str(result["messages"][-1].content)

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

    agent_reviewed = create_agent(
        model=llm,
        tools=[search_docs, plan_queries, review_answer],
        system_prompt=SYSTEM_PROMPT + MULTI_AGENT_SUFFIX,
        checkpointer=checkpointer,
    )
    return agent, agent_reviewed, chat_provider, store
