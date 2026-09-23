'''
Date: 2026-09-22
Author: parker
FilePath: \llmops-api\study\55-多个Tool调用\5.Tool调用异常处理.py
Description: 多个 Tool 调用 —— Tool 调用异常处理
'''
from typing import Annotated

from langchain_core.tools import tool
from pydantic import Field

import dotenv
import json
from langchain_core.messages import HumanMessage, ToolMessage

dotenv.load_dotenv()
from langchain_deepseek import ChatDeepSeek


@tool
def get_weather_info(city:Annotated[str, Field(description="城市名称")]) -> str:
    """用户查询天气来这里"""
    return f"{city}的天气是晴天"

@tool 
def get_city_name(code:Annotated[str, Field(description="城市代码")]) -> str:
    """用户输入代码获取城市名称"""
    if code == '0755':
        return "深圳"
    if code == '020':
            return "广州"
    else:
       raise ValueError(f"查不到城市代码 {code}")

@tool
def get_order_status(order_id: Annotated[str, Field(description="当前订单号")]) -> str:
    """用户查询订单信息"""
    return f"订单{order_id}的状态是已支付"
tools = [get_weather_info, get_order_status,get_city_name]
llm = ChatDeepSeek(model="deepseek-chat", temperature=0)
llm_with_tools = llm.bind_tools(tools)

tools_by_name = {t.name: t for t in [get_order_status, get_weather_info,get_city_name]}

question = "城市代码 999 今天天气怎么样？"
messages = [HumanMessage(content=question)]
rounds = 0  

# ============================================================
# 本文件要搞明白的一件事：
#
#   文件 4 那个圈有个致命弱点。看这一行：
#
#       tool_result = tools_by_name[tool_call["name"]].invoke(tool_call)
#
#   这行有【三处】随时会炸：
#
#     1. tools_by_name[...]      → LLM 报了个不存在的工具名？KeyError
#     2. .invoke(tool_call)      → 工具内部自己报错？异常直接抛出来
#     3. 返回值是 None 或者空    → 不炸，但 LLM 收到个空字符串
#
#   而一旦炸了，异常会穿过 for、穿过 while，把你的程序整个干掉。
#   对面那个 LLM 呢？它什么都收不到 —— 对话戛然而止。
#
#   本文件要做的就一件事：
#     把「失败」也变成一条能被 LLM 读懂的消息。
# ============================================================


# ============================================================
# 第 1 步：把文件 4 的圈搬过来，并修好两处
#
# 文件 4 你那个 while 里有两处要改：
#
#   ① 这一行必须挪到 break 【前面】：
#        if not ai_msg.tool_calls:
#            print(ai_msg.content)    ← 少了它，LLM 最后说的话被吞了
#            break
#      你自己想：现在你的代码里，最后那句话是在哪儿打印出来的？
#      那下一轮 LLM 说的那句呢？有人打印它吗？
#
#   ② 循环里统一用 llm_with_tools，别用 llm。
#      llm 是没绑工具的裸模型 —— 在圈里用它，等于把菜单收走。
#      而且想清楚：你那句 f_msg 现在【没有】被 append 进 messages，
#      这会让下一轮 LLM 看到一份什么样的历史？
#
#   ③ 顺手删掉那个 if/else。上面已经 if not ... break 过了，
#      再写一次 if ai_msg.tool_calls: / else: 是死代码
#      （那个 else 永远不会执行，因为 tool_calls 为空时上面已经 break 了）
#
# TODO(你写)：
# ------------------------------------------------------------
while True:
    rounds += 1                   # 循环里，第一行
    if rounds > 5:
        print("达到最大轮次，提前结束")
        break
    ai_msg  = llm_with_tools.invoke(messages)
    if not ai_msg.tool_calls:
        print(ai_msg.content) 
        break
    messages.append(ai_msg)
    for tool_call in ai_msg.tool_calls:
        try:
            tool_result = tools_by_name[tool_call['name']].invoke(tool_call)
            tool_msg  = ToolMessage(content=tool_result, tool_call_id = tool_call['id'])
        except Exception as e:
            tool_msg  = ToolMessage(content=f"工具 {tool_call['name']} 调用失败：{e}", tool_call_id = tool_call['id'])
            print(f"工具报错：{e}")
        messages.append(tool_msg)
    f_msg = llm.invoke(messages)
    print(f_msg.content)
   

# ============================================================
# 第 2 步：造一个「一定会失败」的工具
#
# 现在你的 get_city_name 遇到不认识的代码，是 return "未知城市"。
# 这太温柔了 —— 我们要看真实世界的样子。
#
# 改成：遇到不认识的代码，直接 raise 一个异常。
#
#     raise ValueError(f"查不到城市代码 {code}")
#
# 自己判断：用哪种异常类型？（ValueError / KeyError / RuntimeError...）
# 提示：这里的意思是「你给我的这个值，我不认识」，不是「键找不到」。
#
# TODO(你写)：
# ------------------------------------------------------------


# ============================================================
# 第 3 步：先看它怎么死 —— 这一步别跳过
#
# 加保护之前，先亲眼看一次崩溃。
#
# 用文件 4 那个 while 跑这句：
#
#     question = "城市代码 999 今天天气怎么样？"
#
# 你会看到：终端刷出一大片红色 Traceback，程序当场停住。
# 那一大片红里，有一句是真正的原因，其余都是「谁调用了谁」的路径。
# 学会一眼找到那一句 —— 这是你以后每天都在干的事。
#
# 盯住看三秒，记住这个感觉：
#    此时 LLM 那边，一条消息都没收到。
#
# TODO(你写)：
# ------------------------------------------------------------


# ============================================================
# 第 4 步：给工具调用穿上防护服 —— 本文件的核心
#
# 把那个 for 循环里的「执行工具」用 try / except 包起来：
#
#     try:
#         tool_result = 从电话本取出工具并执行
#         tool_msg = ToolMessage(content=tool_result, tool_call_id=...)
#     except Exception as e:
#         tool_msg = ToolMessage(content=____, tool_call_id=...)
#         print(f"[工具报错] {e}")
#     messages.append(tool_msg)        ← 注意它在 try 外面
#
#   三个地方要想清楚：
#
#   a. except 里 content=____ 该填什么？
#      ⚠️ 这是本文件最重要的一行。
#
#      想想这条消息是【谁】在读 —— 是 LLM，不是你自己。
#      你写给它看的东西，得让它能看懂，并且能据此决定下一步。
#
#      对比这两种写法，哪种能让 LLM 做出正确反应？
#         A. content=str(e)                     → "查不到城市代码 999"
#         B. content=traceback.format_exc()      → 一整片堆栈
#         C. content="不知道"                    → 什么都不说
#
#   b. 为什么 messages.append(tool_msg) 要放在 try / except 【外面】？
#      提示：不管成功还是失败，有件事都必须发生。是哪件？
#
#   c. except Exception 抓的是「所有异常」。
#      这是好事还是坏事？什么时候应该抓得窄一点？
#
#   d. tool_call_id 在 except 里也必须填吗？为什么？
#
# TODO(你写)：
# ------------------------------------------------------------
#a.A  b. 不管成功还是失败，都要把 tool_msg 放到 模型里，让模型知道工具调用结果  

# ============================================================
# 第 5 步：跑，然后盯着 LLM 的反应
#
# 还是那句：「城市代码 999 今天天气怎么样？」
#
# 这次不该崩了。你要观察的是：
#
#   工具抛了异常 → 你的 except 把它变成了一条 ToolMessage
#   → 这条消息被喂回给 LLM → LLM 会怎么处理？
#
#   它是老实说「查不到这个城市代码」，
#   还是假装没事、自己编了个天气？
#
#   这个答案很能说明问题：错误信息写得够不够清楚，
#   直接决定 LLM 是老实还是瞎编。
#
# TODO(你写)：
# ------------------------------------------------------------


# ============================================================
# 第 6 步：思考题 —— 哪些异常【不该】吞？
#
# 你现在给所有异常都套了一件防护服。但并不是所有异常都该被吞掉。
#
# 判断标准是：这个错误，LLM 有办法处理吗？
#
#   A. 「城市代码 999 查不到」 → LLM 能干嘛？（提示：它可以转告用户）
#   B. 「DeepSeek API key 无效」 → LLM 能干嘛？
#   C. 「数据库连接超时」 → LLM 能干嘛？
#
#   如果 LLM 拿着这个错误什么也做不了，那你吞掉它只是把问题藏起来 ——
#   线上出故障时，你连日志都看不到。这种错就该让它崩，崩给你看。
#
#   真实项目里的做法一般是：
#     - 业务错误（查不到、参数不对）  → 喂回给 LLM
#     - 系统错误（网络、鉴权、数据库）→ 记日志 + 抛出去
#
# TODO(你写)：你自己的答案
# ------------------------------------------------------------
