'''
Date: 2026-09-22
Author: parker
FilePath: \llmops-api\study\57-Agent基础\3.创建第一个agent.py
Description: Agent 基础 —— 创建第一个 Agent
'''

# ============================================================
# 【今天要理解的概念】：Agent 是一个"会自己转圈的模型"
#
# 前两个文件我们确认了两件事：
#
#   文件 1：Chain 的流程图是你画的。管道单向、无判断、不会回头。
#   文件 2：Tool Calling 只是"喊一声"——模型输出一张工单就不管了。
#
# 那缺的那两样（谁来执行、谁来转圈）交给框架之后，长什么样？
#
# ------------------------------------------------------------
# 【最重要的心智模型】
#
#   create_agent() 返回的东西，你【当成一个模型来用】就对了。
#
#   普通模型：  invoke(消息列表) → 一条 AIMessage
#   Agent：     invoke(消息列表) → 整个消息列表（含它自己转的每一圈）
#
#   接口形状几乎一样，区别只有两个：
#
#     ① 输入要包一层 {"messages": [...]}（因为它是个图，图的状态是个字典）
#     ② 它会自己转 N 圈，转完才把控制权还给你
#
#   所以你可以把它理解成：一个"内部装了 while 圈的模型"。
#
# ------------------------------------------------------------
# 【你给它三样东西，它把那三样组装成一个车间】
#
#   model          谁来思考、谁来决定下一步       ← 55 章你手写 llm_with_tools
#   tools=[...]    车间里有哪些机器可以开         ← 55 章你手写 tools 列表
#   system_prompt  工作守则                       ← 55 章你手写 SYSTEM_PROMPT
#
#   ⚠️ 注意：这三样你在 55 章全写过。create_agent 没有引入任何新概念，
#      它只是把你那 80 行代码，换成了 3 个参数。
#
# ------------------------------------------------------------
# 【那"执行工具"和"转圈"去哪了？】
#
#   去看它的返回值类型：CompiledStateGraph —— 一张编译好的图。
#
#   LangGraph 的图 = 节点 + 边。Agent 这张图只有两个节点，来回转：
#
#        ┌──────────────┐
#        │   agent 节点  │  问模型：下一步做什么？
#        └──────┬───────┘
#               │  模型说要调工具
#        ┌──────▼───────┐
#        │  tools 节点   │  执行工具，把结果塞回消息列表
#        └──────┬───────┘
#               │  回到 agent 节点再问一遍
#               └──────────→ ...直到模型说"我不调工具了"
#
#   这条边，就是 55 章你手写的 while 的"回头路"。
#   Agent 和 Chain 唯一的本质差别就在这儿：图可以成环。
#
# ============================================================


# ============================================================
# 【本文件的任务】—— 建一个 Agent，然后【数它转了几圈】
#
# 不用写逻辑。这个文件的价值全在"打印出来看"这五个字上。
#
# ------------------------------------------------------------
# 第 1 步：定义工具（从文件 2 原样抄过来）
#
#     get_weather(city) → 假天气
#     （已经写在文件 2 里了，Ctrl+C / Ctrl+V）
#
# ------------------------------------------------------------
# 第 2 步：建 Agent —— 就一行
#
#     from langchain.agents import create_agent
#
#     agent = create_agent(
#         model=llm,
#         tools=[get_weather],
#     )
#
#     注意这里【没有】system_prompt（先不要，第一次先看最干净的形状）。
#     也注意：【没有】bind_tools。bind_tools 那一步被它吞进去了。
#
# ------------------------------------------------------------
# 第 3 步：问一句需要工具的话
#
#     result = agent.invoke({
#         "messages": [{"role": "user", "content": "深圳今天天气怎么样？"}]
#     })
#
#     ⚠️ 两个坑，先跟你说，免得你卡在这：
#
#     ① 输入要包成 {"messages": [...]}，不能像文件 2 那样直接丢一个字符串。
#        字典里写 {"role": "user", "content": "..."} 就行，
#        LangChain 会自动把它转成 HumanMessage
#        —— 你在第 4 步打印的时候会亲眼看到它变成了什么。
#
#     ② 输出也是字典，最终回答在 result["messages"][-1].content，
#        【不是】 result["content"]。写错了会 KeyError。
#
# ------------------------------------------------------------
# 第 4 步：这是本文件的全部重点 —— 把整个消息列表摊开看
#
#     print("一共几条消息：", len(result["messages"]))
#
#     for i, msg in enumerate(result["messages"]):
#         print(f"--- [{i}] {type(msg).__name__} ---")
#         print("content:", repr(msg.content))
#         print("tool_calls:", msg.tool_calls)      ← 想想：每条消息都有这个属性吗？
#
#     （这个 for 只是【遍历打印】，不是业务判断，允许用。
#       文件 1 那条"不许 if/for"的规矩只针对 Chain 的"决策"，
#       不是说 Python 不能写循环。）
#
# ------------------------------------------------------------
# 【打印出来之后，你必须能回答这 4 个问题】
#
#   a. 一共几条消息？顺序是什么？
#
#   b. 把这 4 条（或几条）和你 55 章手写的那几行 append 一一对上：
#
#        你的第 1 行 messages.append(HumanMessage(...))  → 对应第几条？
#        你的第 2 行 messages.append(ai_msg)             → 对应第几条？
#        你的第 3 行 messages.append(tool_msg)           → 对应第几条？
#        你的最后一行 llm.invoke(messages) 的结果         → 对应第几条？
#
#   c. Agent 转了几圈？—— 从哪一条消息能看出来"它还想再问一次"？
#
#   d. 第 4 步打印出来的第一条消息，是 HumanMessage 还是别的？
#      说明了什么？（提示：你输入的是字典，打出来的是对象）
#
# ------------------------------------------------------------
# 【这四个问题答完，你会有一个"顿悟"】
#
#   result["messages"] 这个列表 ——
#   就是你 55 章一手 append 出来的那个 messages。
#
#   一模一样。一条不多，一条不少。
#
#   区别只在于：55 章是你一行一行 append 的，
#   现在这张图自己转，转完把列表还给你。
#
# TODO(你写)：
# ------------------------------------------------------------
from typing import Annotated

import dotenv
from langchain_core.tools import tool
from langchain_deepseek import ChatDeepSeek
from pydantic import Field
from langchain.agents import create_agent
dotenv.load_dotenv()

@tool
def get_weather(city: Annotated[str,Field(description="要查询天气的城市，例如：北京")])->str:
    """查询指定城市的天气"""
    return f"{city}今天天气晴朗，温度 25℃"

llm  = ChatDeepSeek(model="deepseek-chat",temperature=0)

agent = create_agent(
    model = llm,
    tools = [get_weather]
)

result = agent.invoke(
    {"messages":[{"role":"user","content":"你好，你是谁？"}]}
)
print("一共几条信息：",len(result["messages"]))
for i, msg in enumerate(result["messages"]):
    print(f"--- [{i}] {type(msg).__name__} ---")
    print("content:", repr(msg.content))
    print("tool_calls:", getattr(msg, "tool_calls", []))
# ============================================================
# 【写完之后的加餐实验】—— 别急着改代码，先想
#
#   这个 agent 现在只能干一件事（查天气）。但先别急着加工具。
#   先试这两句话，各跑一次，看第 4 步的输出有什么不同：
#
#     ① "深圳今天天气怎么样？"     ← 需要工具
#     ② "你好，你是谁？"           ← 不需要工具
#
#   对比两次的 len(result["messages"])。
#
#   → 差了 2 条。为什么恰好是 2 条？哪 2 条？
#
#     这就是 Agent 和 Chain 的差别最直白的一次呈现：
#     【同样的代码，问不同的话，内部走的步数不一样。】
#
#     Chain 做不到这一点 —— 它的步数在代码里就写死了。
#
# TODO(你写)：
#   条数①：      条数②：      差的是哪 2 条：
# ------------------------------------------------------------
