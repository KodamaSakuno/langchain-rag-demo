# 裸 LLM 直答 vs RAG 对照

- 生成时间: 2026-08-15T16:14:04+00:00
- Chat 模型: deepseek-v4-flash | Embedding: BAAI/bge-m3
- 检索: top-5, 阈值 0.3, 查询改写 开

## Q1: 如何给 agent 添加短期记忆，让它记住当前对话的历史？

**裸模型直答：**

要给你的 LangChain Agent 添加“短期记忆”（即记住当前对话历史），最推荐的方式是使用 **LangGraph 的检查点（checkpoint）机制**。  
它会为每个会话保存一份状态，在后续请求中自动把历史消息喂回给模型，从而实现记忆。

下面是一个**可直接运行的最小示例**（需要安装 `langgraph`、`langchain-openai`，并设置 `OPENAI_API_KEY`）：

```python
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_agent

# 1. 定义一个简单工具
@tool
def get_weather(location: str) -> str:
    """返回指定地点的天气。"""
    return f"{location} 的天气：晴天，25°C。"

# 2. 创建模型、检查点（内存版短期记忆）和 Agent
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
checkpointer = MemorySaver()  # 内存型检查点，重启后丢失

agent = create_agent(
    model=llm,
    tools=[get_weather],
    checkpointer=checkpointer,
)

# 3. 为当前会话设置一个唯一的 thread_id
config = {"configurable": {"thread_id": "my-session-1"}}

# 第一轮对话：让 Agent 记住一个事实
response1 = agent.invoke(
    {"messages": [{"role": "user", "content": "你好，我叫小明，请记住我的名字。"}]},
    config,
)
print("第一轮回答:", response1["messages"][-1].content)

# 第二轮对话：同一会话中，Agent 能回忆起上一轮内容
response2 = agent.invoke(
    {"messages": [{"role": "user", "content": "我叫什么名字？"}]},
    config,
)
print("第二轮回答:", response2["messages"][-1].content)
```

输出类似：

```
第一轮回答: 你好，小明！我已经记住你的名字了。
第二轮回答: 你叫小明。
```

### 关键点说明
- `MemorySaver()` 将状态保存在内存中，适合“短期记忆”。如果程序重启，记忆会消失；若需要持久化，可换用 `langgraph-checkpoint-sqlite`、`PostgresSaver` 等。
- 通过 `config = {"configurable": {"thread_id": ...}}` 来区分不同会话，同一个 `thread_id` 共享记忆，不同 `thread_id` 之间相互隔离。
- `create_agent` 是 LangGraph 提供的高层 Agent 接口，内部自动处理消息历史和工具调用，无需手动拼接 prompt。

如果你目前使用的是较旧的 `AgentExecutor` 写法，也可以通过 `ConversationBufferMemory` 实现，但官方已逐渐推荐 LangGraph 方案。以上示例即为当前 LangChain 生态中最简洁的“短期记忆”解法。

**RAG 回答**（来源: short-term-memory.mdx, streaming.mdx, messages.mdx, context-engineering.mdx）：

结论：为 agent 添加短期记忆（线程级持久化）需在创建时指定 `checkpointer`，并在调用时通过 `configurable` 传入 `thread_id`。

依据：
- 文档明确说：“To add short-term memory (thread-level persistence) to an agent, you need to specify a `checkpointer` when creating an agent.”
- 本地环境可显式传入检查点：“Locally, pass one explicitly, for example `create_agent(..., checkpointer=InMemorySaver())`.”
- 通过 `thread_id` 恢复对话：“Pass a `thread_id` via `configurable` so the conversation is checkpointed and follow-up turns can resume the same history.”

最小示例（Python）：

```python
from langchain import create_agent
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(
    model="gpt-5.5",
    tools=[],
    checkpointer=InMemorySaver(),
)

# 第一轮对话
agent.invoke(
    {"messages": [{"role": "user", "content": "你好！"}]},
    config={"configurable": {"thread_id": "thread-1"}}
)

# 后续轮次，同一 thread_id 可继续历史
agent.invoke(
    {"messages": [{"role": "user", "content": "我刚才说了什么？"}]},
    config={"configurable": {"thread_id": "thread-1"}}
)
```

生产环境建议使用数据库支持的 checkpointer（如 `langgraph-checkpoint-postgres`）。

## Q2: 怎样跨会话记住用户的偏好信息？

**裸模型直答：**

要跨会话记住用户偏好，不能用只存在内存里的 `ChatMessageHistory`，需要把聊天记录/记忆持久化到外部存储（比如 SQLite、Redis）。LangChain 提供了：

- `SQLChatMessageHistory`：SQLite 保存会话消息
- `RunnableWithMessageHistory`：按 `session_id` 自动加载/保存历史消息

下面给出一个可直接运行的最小示例：用户第一次会话说“我喜欢简洁的回答”，第二次会话用同一个 `session_id` 提问，模型可以结合历史消息回忆起该偏好。

## 1. 安装依赖

```bash
pip install langchain langchain-community langchain-openai
```

如果使用其他模型，替换 `langchain-openai` 即可。

## 2. 完整代码

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_openai import ChatOpenAI

# 1. 定义模型和提示模板
llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key="YOUR_OPENAI_API_KEY",  # 也可以使用环境变量 OPENAI_API_KEY
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个乐于助人的助手。"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

chain = prompt | llm

# 2. 定义按 session_id 获取历史记录的函数
def get_session_history(session_id: str) -> SQLChatMessageHistory:
    return SQLChatMessageHistory(
        session_id=session_id,
        connection_string="sqlite:///memory.db",  # SQLite 数据库文件
    )

# 3. 包装为带消息历史的链
chat_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

# 4. 第一次“会话”：用户告诉模型自己的偏好
response1 = chat_with_history.invoke(
    {"input": "我喜欢简洁的回答，不要啰嗦。"},
    config={"configurable": {"session_id": "user-001"}},
)
print("第一次回答：", response1.content)

# 5. 模拟跨会话：第二次提问，使用同一个 session_id
response2 = chat_with_history.invoke(
    {"input": "你记得我的偏好是什么吗？"},
    config={"configurable": {"session_id": "user-001"}},
)
print("第二次回答：", response2.content)
```

## 3. 运行说明

- 第 4 步会创建 SQLite 文件 `memory.db`，把该会话的消息持久化到本地。
- 第 5 步尽管不是同一个 `chat_with_history` 调用，但由于指定了相同的 `session_id`，LangChain 会从 SQLite 中加载之前的消息，因此模型能“记住”偏好。
- 如果不传 `session_id` 或传不同 `session_id`，则无法互相访问历史记录，也就不会记住偏好。

## 4. 如果偏好需要长期独立使用

如果用户偏好信息不只在对话历史中，而是希望每次生成回复前都显式注入系统提示，可以通过以下方式：

1. 使用一个持久化存储（SQLite/Redis/数据库）保存用户偏好。
2. 在每次构建提示时读取用户偏好，插入 `SystemMessage` 或提示模板里。
3. 同时配置 `RunnableWithMessageHistory` 保存对话历史。

示例片段：

```python
from langchain_core.messages import SystemMessage

# 从数据库读取偏好（伪代码）
user_prefs = get_user_prefs_from_db("user-001")  # 例如 "简洁回答"
pref_message = SystemMessage(content=f"用户偏好：{user_prefs}")

history = get_session_history("user-001")
history.add_message(pref_message)  # 注入系统消息
```

但直接使用历史消息通常已经足够，因为大模型可以从历史内容中推断用户偏好。选择哪种方式取决于你的业务场景和对成本/隐私的要求。

**RAG 回答**（来源: agents.mdx, context-engineering.mdx, quickstart.mdx, short-term-memory.mdx, multi-agent/handoffs-customer-support.mdx）：

结论：跨会话记住用户偏好应使用长时记忆（Store），在创建 agent 时传入 `store`，并在 middleware 中通过 `runtime.store` 按用户 ID 读取。文档只展示了读取偏好，未展示写入偏好的代码，因此写入方式当前文档未找到。

依据：文档示例使用 `InMemoryStore` 和 `@dynamic_prompt`，从 `request.runtime.store` 读取用户偏好，并通过 `context_schema` 传入 `user_id` 作为上下文。

```python
from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from langgraph.store.memory import InMemoryStore

@dataclass
class Context:
    user_id: str

@dynamic_prompt
def store_aware_prompt(request: ModelRequest) -> str:
    user_id = request.runtime.context.user_id
    store = request.runtime.store
    user_prefs = store.get(("preferences",), user_id)
    base = "You are a helpful assistant."
    if user_prefs:
        style = user_prefs.value.get("communication_style", "balanced")
        base += f"\nUser prefers {style} responses."
    return base

agent = create_agent(
    model="gpt-5.5",
    tools=[],
    middleware=[store_aware_prompt],
    context_schema=Context,
    store=InMemoryStore(),
)
```

运行调用方式未在文档中给出；上述代码依据文档片段可直接组装 agent。

## Q3: 如何定义一个自定义工具交给 agent 调用？

**裸模型直答：**

在 LangChain 中，自定义工具最推荐的方式是使用 `@tool` 装饰器或者继承 `BaseTool`。核心思路是：**定义一个函数，加上工具装饰器，然后把工具实例传给 Agent**。

下面是两种可运行的最小示例。

---

## 方式一：使用 `@tool` 装饰器（最简洁）

```python
# pip install langchain langchain-openai
import os

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# 1. 定义自定义工具
@tool
def add(a: int, b: int) -> int:
    """将两个整数相加并返回结果。"""
    return a + b

# 2. 构建 agent
tools = [add]
llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个善于使用工具的助手。"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 3. 调用
result = executor.invoke({"input": "请计算 2 + 3 等于多少？"})
print(result["output"])
```

说明：
- 函数名 `add` 会成为工具名。
- 函数的 docstring 会作为工具的说明（LLM 依赖它来决定何时调用工具）。
- 参数类型注解 `a: int, b: int` 会自动生成工具的 JSON Schema。

---

## 方式二：继承 `BaseTool`（适合复杂工具）

当工具需要内部状态、更多配置或自定义参数校验时，可以继承 `BaseTool`。

```python
from typing import Type

from langchain.tools import BaseTool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# 1. 定义参数 Schema
class AddInput(BaseModel):
    a: int = Field(description="第一个加数")
    b: int = Field(description="第二个加数")

# 2. 定义工具类
class AddTool(BaseTool):
    name: str = "add"
    description: str = "将两个整数相加并返回结果。"
    args_schema: Type[BaseModel] = AddInput

    def _run(self, a: int, b: int) -> int:
        return a + b

# 3. 交给 agent
tools = [AddTool()]
llm = ChatOpenAI(model="gpt-4o", temperature=0)
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个善于使用工具的助手。"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools)
print(executor.invoke({"input": "5 + 7 = ?"}))
```

注意：
- `BaseTool` 是 Pydantic 模型，所以 `name`、`description` 是字段而不是普通类属性。
- `_run` 是同步执行方法，如果需要异步可覆盖 `_arun`。
- 如果工具不复杂，优先使用 `@tool` 装饰器。

---

## 历史兼容写法（可选）

较早的 LangChain 版本常用 `initialize_agent`，写法更短：

```python
from langchain.agents import initialize_agent, AgentType
from langchain.tools import tool
from langchain_openai import ChatOpenAI

@tool
def add(a: int, b: int) -> int:
    """将两个整数相加并返回结果。"""
    return a + b

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
agent = initialize_agent([add], llm, agent=AgentType.OPENAI_FUNCTIONS)
print(agent.run("3 + 4 = ?"))
```

但该 API 在新版中已不推荐，建议使用 `create_tool_calling_agent` 或 `create_openai_functions_agent`。

---

按这个思路，你就能任意定义自定义工具，并让它被 Agent 自动识别和调用了。

**RAG 回答**（来源: models.mdx, multi-agent/custom-workflow.mdx, overview.mdx, multi-agent/subagents-personal-assistant.mdx）：

结论：当前文档片段中未找到定义自定义工具并交给 agent 调用的具体 API 与写法。

依据：文档片段仅提到“This example demonstrates how to create a simple LangChain agent with a custom tool:”，但对应的 python 代码块为空；同时提及 `create_agent` 支持自定义工具，但未展示任何工具定义代码（如 `@tool` 或 `bind_tools`）。因此无法依据当前片段给出可运行示例。

## Q4: 工具执行报错时怎么处理错误？

**裸模型直答：**

在 LangChain 中处理工具执行错误，推荐使用工具自带的 `handle_tool_error` 参数。它可以让工具在抛出异常时**不中断调用**，而是返回一段错误信息，尤其在 Agent 中能被 LLM 看到并继续推理。

## 最小可运行示例

```python
from langchain.tools import Tool

def bad_tool(query: str) -> str:
    raise ValueError("boom")

tool = Tool(
    name="bad_tool",
    func=bad_tool,
    description="总是失败的示例工具",
    handle_tool_error="Tool error: {error}",  # 支持字符串模板，{error} 为原始异常
)

# 直接调用工具：不会抛出异常，而是返回错误描述
result = tool.run("test")
print(result)
```

输出：

```text
Tool error: boom
```

## 其他方式

- `handle_tool_error=True`：使用默认的错误信息。
- `handle_tool_error=callable`：自定义函数，接收异常参数并返回字符串。
- 在工具函数内部用 `try/except` 自己捕获并返回错误字符串，也可行，但不推荐与 Agent 配合时使用。

```python
def my_handler(error: Exception) -> str:
    return f"自定义错误：{error}"

tool = Tool(..., handle_tool_error=my_handler)
```

当工具被 AgentExecutor 调用时，`handle_tool_error` 同样生效——工具异常不会让 Agent 崩溃，错误信息会成为 `Observation` 反馈给模型继续处理。

**RAG 回答**（来源: structured-output.mdx, tools.mdx, agents.mdx）：

结论：LangChain 默认会通过 `ToolMessage` 将错误反馈给模型并自动重试；若要自定义处理，可在结构化输出工具的 `response_format` 中配置 `handle_errors`（Python）或 `handleError`（TS）传入错误处理函数。若指一般工具调用错误，文档仅提及可用 middleware，未给出具体代码，当前文档中未找到。

依据：文档中的 `custom_error_handler` 示例及 `ToolStrategy(..., handle_errors=...)`；TS 示例使用 `toolStrategy(ProductRating, { handleError: ... })` 捕获 `ToolInputParsingException`。

Python 最小示例（基于文档写法，需按官方文档补齐导入）：

```python
from langchain.agents.structured_output import StructuredOutputValidationError, MultipleStructuredOutputsError

def custom_error_handler(error: Exception) -> str:
    if isinstance(error, StructuredOutputValidationError):
        return "There was an issue with the format. Try again."
    elif isinstance(error, MultipleStructuredOutputsError):
        return "Multiple structured outputs were returned. Pick the most relevant one."
    else:
        return f"Error: {str(error)}"

agent = create_agent(
    model="gpt-5.5",
    tools=[],
    response_format=ToolStrategy(
        schema=Union[ContactInfo, EventDetails],
        handle_errors=custom_error_handler
    )
)
```

## Q5: 如何让模型输出符合指定 schema 的结构化结果？

**裸模型直答：**

要让模型输出符合指定 schema 的结构化结果，最简单的方式是使用 LangChain 的 `with_structured_output()` 方法，配合 Pydantic 定义输出结构。它会自动调用模型的功能调用（function calling）或 JSON 模式，并把输出解析为 Pydantic 对象。

下面是一个可运行的最小示例（需要安装 `langchain-openai` 并配置 `OPENAI_API_KEY`）：

```python
import os
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# 设置 OpenAI API Key（也可以使用环境变量）
os.environ["OPENAI_API_KEY"] = "你的-API-Key"

# 定义期望的输出 Schema
class Person(BaseModel):
    name: str = Field(description="姓名")
    age: int = Field(description="年龄")

# 初始化模型并绑定结构化输出
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
structured_llm = llm.with_structured_output(Person)

# 调用模型，返回的是 Person 实例
result = structured_llm.invoke("张三今年25岁")
print(result)
print(result.name, result.age)
```

运行后会输出类似：
```
name='张三' age=25
张三 25
```

如果 Prompt 中没有提供足够信息，模型可能会根据已知知识填补或要求澄清，但 `with_structured_output` 会确保输出结构严格符合 `Person` 定义。你还可以通过 `method` 参数指定使用的机制，例如 `method="json_mode"` 或 `method="function_calling"`。

**RAG 回答**（来源: structured-output.mdx）：

结论：在 `create_agent` 中设置 `response_format` 为你的 schema 类型，模型就会输出符合该 schema 的结构化结果，数据会保存在返回状态的 `'structured_response'` 键中。

依据：文档指出 `response_format` 用于控制 agent 返回结构化数据；直接传 schema 类型时，LangChain 会自动选择 `ProviderStrategy`（模型支持原生结构化输出时）或回退到 `ToolStrategy`。支持 Pydantic、dataclass、TypedDict、JSON Schema、Union 类型。

最小示例（需先初始化 `model`，文档未提供 `model` 的创建方式，故此处为占位）：

```python
from langchain import create_agent
from pydantic import BaseModel

class MySchema(BaseModel):
    summary: str
    score: float

# model = ...  # 你的聊天模型实例（需自行配置）
agent = create_agent(
    model=model,
    response_format=MySchema,
)

# 调用 agent 后，result["structured_response"] 即为符合 MySchema 的对象
```

注意：当前文档未找到 `model` 的初始化方法，因此上述代码需你自行提供模型实例才能直接运行。

## Q6: ProviderStrategy 和 ToolStrategy 有什么区别？

**裸模型直答：**

在 LangChain 中，`ProviderStrategy` 和 `ToolStrategy` 通常不是两个公开的“类”，而是指获取结构化输出（`with_structured_output`）时两种不同的实现策略：

| 维度 | ProviderStrategy | ToolStrategy |
|------|------------------|--------------|
| 思路 | 使用模型提供方自己的“原生结构化输出”能力 | 使用“工具/函数调用”能力，强制模型把结构化输出放到工具参数里 |
| 例子 | OpenAI 的 `response_format={"type": "json_schema", ...}`；Ollama 的 `format=json`；Gemini 的 `responseSchema` | 把输出 schema 定义成一个 `tool`，让模型调用这个工具，再解析 tool call 的 `args` |
| 可移植性 | 低：每个 Provider 的参数和实现都不同 | 高：只要模型支持 tool calling，就可以用统一的 `bind_tools` 接口 |
| 输出解析 | 模型直接输出 JSON，通常只需要一个 JSON Parser | 模型返回的是“工具调用”，需要提取 tool call 里的 arguments 参数 |
| 适用场景 | 使用 OpenAI 新版 `json_schema`、Gemini 原生 structured output 等 | 大多数支持 function calling 的模型，例如 OpenAI、Anthropic、Llama 3、Mistral 等 |

简单说：

- **ProviderStrategy** = “我信任 Provider 原生约束，让它直接吐 JSON 匹配 schema。”
- **ToolStrategy** = “我不依赖 Provider 的特殊 JSON 模式，而是把 schema 包装成一个工具，让模型调用工具来返回数据。”

---

## 最小可运行示例

假设有 OpenAI API Key，并且安装：

```bash
pip install -U langchain-openai langchain-core pydantic
```

```python
from langchain_openai import ChatOpenAI

# 定义输出结构
class Person(BaseModel):
    name: str
    age: int

llm = ChatOpenAI(model="gpt-4o", temperature=0)

# ToolStrategy 写法：把 schema 当工具
# 方法等价于 llm.with_structured_output(Person, method="function_calling")
tool_strategy_chain = llm.with_structured_output(Person, method="function_calling")

# ProviderStrategy 写法：使用 OpenAI 原生 JSON Schema 能力
provider_strategy_chain = llm.with_structured_output(Person, method="json_schema")

print(tool_strategy_chain.invoke("Alice is 30 years old"))
print(provider_strategy_chain.invoke("Alice is 30 years old"))
```

输出类似：

```text
name='Alice' age=30
name='Alice' age=30
```

两种方案最终都能得到 `Person` 对象，但内部原理不同：

- `method="function_calling"` 是 **ToolStrategy**：LangChain 会调用 `bind_tools([Person])`，然后从模型返回的 tool call 参数中解析出 `Person`。
- `method="json_schema"` 是 **ProviderStrategy**：LangChain 会调用 OpenAI 的 `response_format={"type": "json_schema", ...}`，让模型直接输出符合 schema 的 JSON。

如果模型或 Provider 不支持 `json_schema`，用 `ToolStrategy` 通常是最通用的选择。

**RAG 回答**（来源: structured-output.mdx, middleware/built-in.mdx）：

**结论：**  
`ProviderStrategy` 使用模型提供方的原生结构化输出能力；`ToolStrategy` 通过工具调用（tool calling）实现结构化输出。

**依据：**  
文档指出：  
- `ProviderStrategy[StructuredResponseT]`：Uses provider-native structured output  
- `ToolStrategy[StructuredResponseT]`：Uses tool calling for structured output  

当直接传 schema 类型给 `create_agent(response_format)` 时，LangChain 会自动选择：  
- 若模型支持原生结构化输出（如 OpenAI、Anthropic、xAI），使用 `ProviderStrategy`；  
- 其他模型使用 `ToolStrategy`；  
- 若结构化输出不支持，则回退到工具调用策略。  

简言之，`ProviderStrategy` 依赖模型原生能力，`ToolStrategy` 是通用兜底方案。

## Q7: 创建 agent 时需要指定 checkpointer 是为什么？

**裸模型直答：**

创建 **LangGraph Agent**（例如 `create_react_agent`）时指定 `checkpointer`，核心目的是：**让 Agent 具备“状态持久化”能力**，从而支持多轮对话、断点续跑、人工审批、错误恢复等场景。

如果没有 checkpointer，Agent 每次 `invoke` 都是独立的，它不会记得上一轮聊了什么，也无法在长时间运行的任务中暂停/恢复。

### checkpointer 主要作用

- **会话记忆**：通过 `thread_id` 保存消息历史，让同一会话上下文连续。
- **断点续跑**：Agent 执行过程中如果中断（例如等待人工确认），可以从保存的 checkpoint 恢复。
- **容错重试**：某一步失败后，可以回到之前的 checkpoint 重试。
- **时间旅行**：调试时可回放历史状态。

---

### 最小可运行示例

```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

# 1. 定义一个工具
@tool
def get_weather(city: str) -> str:
    """返回指定城市的天气"""
    return f"{city}：晴，25℃"

# 2. 创建 checkpointer（MemorySaver 是内存版，生产环境可换成 SqliteSaver 等）
checkpointer = MemorySaver()

# 3. 创建 Agent 时传入 checkpointer
agent = create_react_agent(
    model=ChatOpenAI(model="gpt-4o", temperature=0),
    tools=[get_weather],
    checkpointer=checkpointer,  # 关键参数
)

# 4. 用 thread_id 标识同一个会话
config = {"configurable": {"thread_id": "user-123"}}

# 第一次提问
result1 = agent.invoke(
    {"messages": [("user", "北京天气？")]},
    config=config,
)
print(result1["messages"][-1].content)

# 第二次提问，Agent 会记得这是同一个会话
result2 = agent.invoke(
    {"messages": [("user", "那上海呢？")]},
    config=config,
)
print(result2["messages"][-1].content)
```

运行前需要设置环境变量：

```bash
export OPENAI_API_KEY="sk-..."
```

并安装依赖：

```bash
pip install langgraph langchain-openai langchain-core
```

---

### 总结

指定 checkpointer 不是“语法上的必须”，而是**构建真正有记忆、可恢复、可交互的 Agent 的工程需要**。如果只是做一次性无状态调用，不传也能跑；但要做多轮对话、人工介入、任务续跑，就必须指定 checkpointer。

**RAG 回答**（来源: quickstart.mdx, short-term-memory.mdx, multi-agent/skills-sql-assistant.mdx, overview.mdx, multi-agent/handoffs-customer-support.mdx）：

结论：指定 `checkpointer` 是为了给 agent 添加短期记忆（线程级持久化），使对话状态在多次调用间保留，并支持随时恢复线程。

依据：文档中明确说明“To add short-term memory (thread-level persistence) to an agent, you need to specify a `checkpointer` when creating an agent.” 状态通过 checkpointer 持久化到数据库或内存中，agent 在每次步骤开始时读取状态，并在调用或工具完成后更新状态。

最小示例（TypeScript）：

```ts
import { createAgent } from "langchain";
import { MemorySaver } from "@langchain/langgraph";

const agent = createAgent({
  model,
  systemPrompt: "You are a helpful assistant.",
  checkpointer: new MemorySaver(),
});
```

（Python 对应使用 `InMemorySaver`，代码略）

## Q8: 如何中断 agent 的执行，等待人工审批后再继续？

**裸模型直答：**

要中断 Agent 执行并等待人工审批后再继续，推荐使用 **LangGraph 的 `interrupt`** 机制。它可以在图的某个节点暂停执行，保存状态，待人工确认后通过 `Command(resume=...)` 恢复。

下面是一个最小可运行示例：

```python
from typing import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command


class State(TypedDict):
    question: str
    approved: bool
    result: str


def require_approval(state: State) -> State:
    # 在这里中断，等待人工输入 yes/no
    decision = interrupt({"question": state["question"]})
    return {"approved": decision == "yes"}


def after_approval(state: State) -> State:
    if state["approved"]:
        return {"result": "已获批，继续执行任务"}
    return {"result": "被拒绝，流程终止"}


# 构建图
graph = StateGraph(State)
graph.add_node("require_approval", require_approval)
graph.add_node("after_approval", after_approval)
graph.add_edge(START, "require_approval")
graph.add_edge("require_approval", "after_approval")
graph.add_edge("after_approval", END)

# 必须有 checkpointer 才能恢复
app = graph.compile(checkpointer=MemorySaver())

# 第一次执行：会暂停在 require_approval 节点
config = {"configurable": {"thread_id": "test-1"}}
result = app.invoke({"question": "请批准这次操作"}, config)
print("暂停时返回:", result)

# 人工审批后，恢复执行
result = app.invoke(Command(resume="yes"), config)
print("恢复后返回:", result)
```

### 关键点

- `interrupt(...)` 会将控制权交回外部调用者，此时图处于挂起状态。
- 恢复时使用同一个 `thread_id`，并通过 `Command(resume="yes")` 传入人工审批结果。
- `interrupt` 的返回值就是 `Command(resume=...)` 中传入的值。
- 必须使用 `checkpointer` 保存状态，否则无法恢复。

**RAG 回答**（来源: sql-agent.mdx, middleware/built-in.mdx, human-in-the-loop.mdx）：

结论：通过 `HumanInTheLoopMiddleware` 中间件并配置 checkpointer，在 `interrupt_on` 中指定需要审批的工具，即可在工具执行前中断 agent，等待人工审批后恢复执行。

依据：文档片段说明，Human-in-the-loop middleware 需要 checkpointer 维持中断状态，并展示了 `interrupt_on` 配置方式。例如：
```python
HumanInTheLoopMiddleware(
    interrupt_on={"execute_sql": {"allowed_decisions": ["approve", "reject"]}},
    description_prefix="Tool execution pending approval",
)
```
恢复执行时使用 `Command`（详见文档）。

最小示例（基于文档 Python 片段补充工具定义）：
```python
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver

def execute_sql(query: str) -> str:
    return f"Executed: {query}"

agent = create_agent(
    model="gpt-5.5",
    tools=[execute_sql],
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={"execute_sql": True},
            description_prefix="Tool execution pending approval",
        ),
    ],
    checkpointer=InMemorySaver(),
)
```
运行 agent 后，执行 `execute_sql` 前会暂停，等待人工审批。
