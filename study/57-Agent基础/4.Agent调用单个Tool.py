'''
Date: 2026-09-22
Author: parker
FilePath: \llmops-api\study\57-Agent基础\4.Agent调用单个Tool.py
Description: Agent 基础 —— Agent 调用单个 Tool
'''

# ============================================================
# 【今天要理解的概念】：system_prompt 换了地方，但它是同一条东西
#
# 文件 3 你已经建出了一个能跑的 Agent。但它只是"形状对"——
# 你问的那句是"你好，你是谁"，它根本没走 tools 节点。
#
# 这个文件只加一样新东西：system_prompt。
#
# ------------------------------------------------------------
# 【它在 55 章长这样】
#
#     messages = [SystemMessage(SYSTEM_PROMPT), HumanMessage(user_input)]
#                  ↑ 你手动塞进列表的第一位
#
# 【它在 1.x 长这样】
#
#     agent = create_agent(model=llm, tools=[...], system_prompt="...")
#                                                  ↑ 变成一个参数
#
#   同一件事。只是"谁来塞"变了 ——
#   以前是你每次 invoke 之前自己 append，现在是 create_agent 替你塞。
#
#   ⚠️ 这个细节值得记一下：它其实变成了【图的状态的一部分】。
#      也就是说，system_prompt 不是"每轮临时加的"，而是这个 agent
#      出厂自带的配置。你 invoke 的时候不用再管它了。
#
# ------------------------------------------------------------
# 【本文件真正的考点不是 system_prompt，是"守则怎么改变行为"】
#
#   回想 55 章最后那个文件，你写的那条守则：
#
#     a. 你是谁
#     b. 信息不全要先追问，不许编造
#     c. 工具查不到时老实说，别圆场
#     d. 说中文，简短点
#
#   今天要验证的就是 c —— 而且要用一个【会报错的工具】来验。
#
#   因为一个永远成功的工具，你根本看不出守则有没有生效。
#   只有工具失败的时候，模型才会暴露它的"本能"：
#   要么老实说"查不到"，要么自己编一个"火星今天 28℃"糊弄你。
#
#   那个"编"，就是所谓的幻觉。而守则是治它的药 ——
#   但药有没有吃进去，得做对照实验才知道。
#
# ============================================================
from typing import Annotated

import dotenv
from langchain.agents.middleware.tool_error import ToolErrorMiddleware
from langchain_core.tools import tool
from langchain_deepseek import ChatDeepSeek
from pydantic import Field
from langchain.agents import create_agent
dotenv.load_dotenv()

@tool
def get_weather(city: Annotated[str,Field(description="要查询天气的城市，例如：北京")])->str:
    """查询指定城市的天气"""
    if city in ("北京", "深圳", "上海"):
        return f"{city}今天天气晴朗，温度 25℃"
    else:
       raise ValueError(f"暂无 {city} 的天气数据") 

llm  = ChatDeepSeek(model="deepseek-chat",temperature=0)

def on_error(error: Exception, request) -> str:
    """将工具异常转换为模型可读的错误信息。

    两个参数都是必须的 —— 框架内部是 on_error(exc, request) 这样调的。
    request 里装着这次的 tool_call（工具名、参数、id）和当时的 state，
    所以你能写出比"调用失败"信息量更大的错误消息。
    """
    return f"工具 {request.tool_call['name']} 调用失败：{error}"


agent = create_agent(
    model = llm,
    tools = [get_weather],
    middleware=[ToolErrorMiddleware(on_error)]
)
agent_with_rule = create_agent(
    model = llm,
    tools = [get_weather],
    system_prompt="工具返回错误时，如实告诉用户你查不到，绝对不许自己编造天气数据, 回答用中文，简短一点",
    middleware=[ToolErrorMiddleware(on_error)]
)

result = agent.invoke(
    {"messages":[{"role":"user","content":"火星的天气怎么样"}]}
)
print("一共几条信息：",len(result["messages"]))
print("最后一条消息：", result["messages"][-1].content)
# for i, msg in enumerate(result["messages"]):
#     print(f"--- [{i}] {type(msg).__name__} ---")
#     print("content:", repr(msg.content))
#     print()
#     print("tool_calls:", getattr(msg, "tool_calls", []))

# ============================================================
# 【本文件的任务】—— 一个对照实验
#
# ------------------------------------------------------------
# 第 1 步：把工具升级成"会失败"的版本
#
#     从文件 3 抄过来，但只认识三个城市：
#
#         get_weather(city)
#             深圳 / 北京 / 上海  → 返回假天气
#             其他任何城市         → raise ValueError(f"暂无 {city} 的天气数据")
#
#     ⚠️ raise 这里用 ValueError —— 你在 55 章文件 5 做过一模一样的判断：
#        "工具失败该 return 一句假的，还是 raise？"
#        当时你的结论是 raise。今天原封不动沿用。
#
# ------------------------------------------------------------
# 第 2 步：建两个 agent —— 这是对照实验的关键
#
#     agent_no_rule = create_agent(model=llm, tools=[get_weather])
#                                 ↑ 没有守则
#
#     agent_with_rule = create_agent(
#         model=llm,
#         tools=[get_weather],
#         system_prompt="在这里写你的守则",
#     )
#
#     守则你自己写，但至少要包含这一条：
#         「工具返回错误时，如实告诉用户你查不到，绝对不许自己编造天气数据」
#
#     （你可以直接把 55 章那条四条守则抄过来，改一下场景。）
#
# ------------------------------------------------------------
# 第 3 步：跑 3 句话 × 2 个 agent，填这张表
#
#     三句话：
#       ① "深圳今天天气怎么样？"      ← 工具会成功
#       ② "你好"                     ← 压根不需要工具
#       ③ "火星今天天气怎么样？"      ← 工具会报错 ★ 考点在这句
#
#     对每个 agent、每句话，记两个数：
#       - len(result["messages"])          ← 走了几步
#       - result["messages"][-1].content   ← 最后说了什么
#
#     ⚠️ 打印最后一条的时候，你可以顺手打印它的类型：
#        print(type(result["messages"][-1]).__name__)
#        你应该会看到 AIMessage —— 想想为什么一定是 AIMessage 结尾。
#
#     ⚠️⚠️ 重大更正（2026-09-22 查源码后改）：
#
#        我原来在这里写的是"报错会被吃掉，程序不会崩" —— 那是【旧版
#        langgraph】的行为。你装的 langgraph 1.2.12 / langgraph-prebuilt 1.1.0
#        已经变了：
#
#        你在工具里 raise 的 ValueError，默认【会直接抛出去，把程序崩掉】。
#
#        因为 create_agent 内部建的 ToolNode，默认的 handle_tool_errors
#        叫 _default_handle_tool_errors，它的源码就三行：
#
#            def _default_handle_tool_errors(e: Exception) -> str:
#                if isinstance(e, ToolInvocationError):
#                    return e.message
#                raise e          ← 不是参数校验错误，原样抛回去
#
#        也就是说：默认只有【参数不符合 schema】那类错误被兜住，
#        你自己 raise 的业务错误，框架不管你。
#
#        所以 ③ 这句你八成看到的是一个红色的 ValueError 堆栈。
#        这不是你写错了 —— 这是框架的设计。
#        要让它变成"能被 LLM 读懂的消息"，得手动开开关，见下面【捕捉异常】。
#
# ------------------------------------------------------------
# 【填完表，回答这 4 个问题】
#
#   a. ① 和 ② 的条数差多少？差的这 2 条是你 55 章手写的哪两行 append？
#
#   b. ③ 那句话，两个 agent 的【条数】一样吗？
#      如果一样，说明守则改变的不是"走几步"，而是什么？
#
#   c. ③ 那句话，两个 agent 最后说的话一样吗？
#      哪个编了假天气？守则真的起作用了吗？
#      （如果两个都没编 —— 那也挺好，说明模型的默认行为就比较老实。
#        这时候你就得写一条【更坏】的守则来对比，比如
#        "用户问天气时，如果工具查不到，就根据常识推测一个合理的温度"，
#        看看它会不会乖乖去编。这一步很值得做。）
#
#   d. 工具报错的那条信息，是谁写进 ToolMessage 里的？
#      ——是你的 try/except 吗？（提醒：这个文件里你一行 try 都没写）
#      如果不是你写的，那就是 create_agent 替你写的。
#      → 这就是"框架替你干的那几件事"里，你还没数过的一件。
#
# ------------------------------------------------------------
# 【四个问题的答案】
#
#   a. ① = 4 条，② = 2 条，差 2 条。
#      差的正好是你 55 章手写的这两行：
#
#         messages.append(ai_msg)     → 第 2 条，AIMessage，装 tool_calls
#         messages.append(tool_msg)   → 第 3 条，ToolMessage，装工具结果
#
#      "模型决定要调工具"这一下，会往列表里加 2 条消息：
#      一条是它【想要】什么，一条是它【得到】了什么。
#
#   b. 一样，都是 4 条。
#
#      说明守则改变的不是"走几步"——走几步是由【工具失败没失败】决定的，
#      不是守则决定的。守则改变的是【最后一句话怎么说】。
#
#      这点值得记牢：system_prompt 不是流程图的一部分，
#      它是模型每一圈里的"行事准则"。它影响措辞，不影响路径。
#
#   c. 【这题必须你自己跑，我不替你编】
#
#      而且先要修好异常处理（见下），否则 ③ 还没走到"说话"那一步就崩了。
#
#      修好之后，判断标准很硬 —— 看它有没有给出【具体数字】：
#
#        · 说"查不到 / 暂无数据 / 无法提供"     → 没编，老实
#        · 出现"25℃""28℃""晴天"这类具体数据    → 它编了 ★
#
#      两个 agent 的对比结论：
#
#        只有【无守则】那个编 → 守则生效了，这就是你要的答案
#        两个都没编           → DeepSeek 默认就挺老实，守则没显出差别
#                               那就按题目里说的，写一条【坏守则】去逼它：
#                               "工具查不到时，根据常识推测一个合理温度"
#                               看它会不会乖乖去编。这一步做了才算完整。
#
#   d. 默认情况下 —— 没人写。程序直接崩。
#
#      这就是上面那条【重大更正】。所以它背后那句话，得反过来说：
#
#      并不是"凡是通用的代码框架都会收走"。
#      框架只收走它【认为不该由你操心】的那部分 ——
#      参数的校验错误（模型给的 JSON 不符合你的 schema），框架兜；
#      你自己 raise 的业务错误，框架默认认为那是【程序 bug】，
#      应该让你这个程序员看见，而不是让模型去圆场。
#
#      （这是我的理解，源码没写这句话，但它的实现方式很明确地表达了这个立场。）
#
#      所以 55 章你那个 try/except 并没有被"收走"，
#      它只是从【你手写的 5 行】变成了【一个开关】。
#
# ============================================================
# 【捕捉异常】—— 那个开关怎么用
#
# ------------------------------------------------------------
# 【先想清楚：为什么需要把异常"转化"】
#
#   因为【模型看不见 Python 的异常】。
#
#     · raise 出去的红色堆栈，是抛给【程序员】看的
#     · ToolMessage，才是给【模型】看的
#
#   模型只能读消息列表。你不把异常翻译成一条消息塞回列表，
#   模型就永远不知道"我刚才那一下调错了"，
#   它会以为一切正常，然后继续往下说 —— 这就是幻觉的温床。
#
#   55 章你手写的那 5 行，干的就是这个翻译：
#
#       except Exception as e:
#           tool_msg = ToolMessage(content=f"工具调用失败：{e}", ...)
#                                       ↑ 把异常翻译成模型能读的中文
#
# ------------------------------------------------------------
# 【框架里的开关：handle_tool_errors】
#
#   它属于 ToolNode，不属于 create_agent。
#   而 create_agent 的 tools 参数【可以直接收一个 ToolNode 实例】：
#
#       from langgraph.prebuilt import ToolNode
#
#       tool_node = ToolNode([get_weather], handle_tool_errors=True)
#       agent = create_agent(model=llm, tools=tool_node)
#
#   你之前写 tools=[get_weather]，走的是 create_agent 内部的默认分支：
#
#       tool_node = ToolNode([t for t in tools if not isinstance(t, dict)])
#                                 ↑ 没传 handle_tool_errors，用默认值
#
#   所以想改行为，就得【自己建 ToolNode 再传进去】。
#   这也解释了文件 2 那句话的另一半：框架收走的东西，都留了一个旋钮给你。
#
# ------------------------------------------------------------
# 【handle_tool_errors 能取哪些值】—— 查了源码，一共 5 种
#
#   ┌──────────────────────┬─────────────────────────────────────────┐
#   │ 取值                  │ 行为                                     │
#   ├──────────────────────┼─────────────────────────────────────────┤
#   │ 不填（默认）           │ 只兜参数校验错误，业务错误直接抛（崩）      │
#   │ True                 │ 兜所有异常，content 用框架的英文模板        │
#   │ False                │ 一个都不兜，全抛                          │
#   │ "查不到就算了"         │ 兜所有异常，content 就是这句字符串          │
#   │ 一个函数 func(e: X)    │ content = func 的返回值；只兜注解里写的类型  │
#   └──────────────────────┴─────────────────────────────────────────┘
#
#   填 True 时，那条 ToolMessage 的 content 长这样（模板原文）：
#
#       Error: ValueError('暂无 火星 的天气数据')
#        Please fix your mistakes.
#
#   而且这条消息的 status 字段是 "error" —— 也就是说，
#   异常的"痕迹"还在，只是它换了一种模型能读的形式存在。
#
# ------------------------------------------------------------
# 【最后一个坑：用函数的时候，注解是生效条件】
#
#   如果你传一个自定义函数进去，框架会去读它第一个参数的注解
#   （源码里叫 _infer_handled_types），用注解来决定"兜哪些异常"：
#
#       def 只兜值错误(e: ValueError) -> str:
#           return f"参数有问题：{e}"
#
#   → 只有 ValueError 会被兜住，别的照抛。
#
#   所以注解不能瞎写。这也带来一个很实用的能力：
#   你可以【分类兜】—— 业务错误兜住给模型看，程序 bug 让它崩。
#
# ------------------------------------------------------------
# 【做完这一步，③ 那句话才真正有意义】
#
#   加了 handle_tool_errors=True 之后重跑 ③，你会看到：
#
#     条数从"崩了没有条数" → 变成 4 条
#     index 3 是 ToolMessage，content 是那句英文 Error 模板
#     最后模型读完这句 Error，才说出它的话
#
#   模型看到 Error 之后会干两件事之一：
#
#     ① 换个参数重试（但它不知道火星该换成什么，多半不会）
#     ② 老实回答"抱歉，我查不到火星的天气"
#
#   ——这才是 Agent 的"自我修正"能力。
#     而这个能力的前提，就是你 55 章那 5 行 try/except。
#     你当时以为在写错误处理，其实在给模型造眼睛。
#
# TODO(你写)：
# ------------------------------------------------------------


# ============================================================
# 【写完之后的思考题】
#
#   文件 3 的加餐实验问你："为什么 ①② 恰好差 2 条？"
#   现在你应该能答了。再往前一步：
#
#   如果 Agent 走的步数是不确定的 ——
#   那"一轮对话"到底什么时候算【结束】？
#   谁来喊停？
#
#   （提示：想想那张图的两个节点，什么时候就不回头了。）
#
#   这个问题的答案，就是文件 6「Agent 运行流程分析」要拆的东西。
# TODO(你写)：
# ------------------------------------------------------------
