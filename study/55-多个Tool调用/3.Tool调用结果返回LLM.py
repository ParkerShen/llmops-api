'''
Date: 2026-09-22
Author: parker
FilePath: \llmops-api\study\55-多个Tool调用\3.Tool调用结果返回LLM.py
Description: 多个 Tool 调用 —— 把工具结果返回给 LLM
'''

# ============================================================
# 本文件要搞明白的一件事：
#
#   文件 2 里 LLM 只是「说」它想调谁，然后就没了。
#   它说了「我要调 get_order_status」，但没人去执行。
#
#   本文件把这一圈走完：
#
#     用户说话 → LLM 说要调工具 → 你真的去执行 → 把结果喂回去
#                                                    → LLM 说出人话
#
#   上一章 54/6 你其实走过一遍流程了。但那会儿只有 1 个工具，
#   所以你可以偷懒写成 get_logistics_fee.invoke(...)。
#
#   现在有 2 个工具，偷不了懒了 —— 这就是本文件的新问题。
# ============================================================
from typing import Annotated

from langchain_core.tools import tool
from pydantic import Field

import dotenv
import json
from langchain_core.messages import HumanMessage, ToolMessage

dotenv.load_dotenv()
from langchain_deepseek import ChatDeepSeek

# ============================================================
# 第 1 步：搬两个工具 + 建 llm_with_tools
#
# 这一步和文件 2 一模一样，直接复制过来。
#
# 偷懒提醒：description 就用文件 2 里那版（已经比文件 1 好了）。
#          别改，本文件不做描述实验了。
#
# TODO(你写)：
# ------------------------------------------------------------
@tool
def get_weather_info(city:Annotated[str, Field(description="城市名称")]) -> str:
    """用户查询天气来这里"""
    return f"{city}的天气是晴天"


@tool
def get_order_status(order_id: Annotated[str, Field(description="当前订单号")]) -> str:
    """用户查询订单信息"""
    return f"订单{order_id}的状态是已支付"
tools = [get_weather_info, get_order_status]
llm = ChatDeepSeek(model="deepseek-chat", temperature=0)
llm_with_tools = llm.bind_tools(tools)

# ============================================================
# 第 2 步：做一张「电话本」—— 本文件的核心
#
# 先想清楚问题出在哪：
#
#   LLM 给你的东西长这样：
#       {'name': 'get_order_status', 'args': {...}, 'id': 'call_xxx'}
#
#   注意：它给的 'name' 是一个【字符串】，不是函数本身。
#   Python 里你没法用字符串直接调用函数。
#
#   所以你得先准备一张对照表，能从「字符串名字」找回「函数对象」：
#
#       'get_weather_info'  →  get_weather_info 这个函数
#       'get_order_status'  →  get_order_status 这个函数
#
#   这种「一个键对上一个值」的东西，你学过它叫什么吗？
#   （提示：中括号 + 冒号的那个数据结构）
#
# TODO(你写)：变量名建议叫 TOOL_MAP 或 tool_map
# ------------------------------------------------------------


tools_by_name = {t.name: t for t in [get_order_status, get_weather_info]}
# ============================================================
# 第 3 步：选一句话，让 LLM 一定会调工具
#
# ⚠️ 用文件 2 的教训来选：
#
#   别再问「我的快递怎么还没到」了 —— 那句会得到 []，
#   因为订单号填不出来，整个流程根本走不到「执行工具」那一步。
#
#   要选一句【参数齐全】的，比如：
#       「帮我查一下订单 12345 付款了吗?」
#
#   先把流程走通，再玩难的。
#
# TODO(你写)：question = "..."
# ------------------------------------------------------------
question = "帮我查询订单12345状态"

# ============================================================
# 第 4 步：走完整个流程
#
# 上一章 54/6 你写过这一段，先自己回忆，回忆不起来再看下面。
messages = [HumanMessage(content=question)]
ai_msg  = llm_with_tools.invoke(messages)
messages.append(ai_msg)
if ai_msg.tool_calls:
    for tool_call in ai_msg.tool_calls:
        
        tool_result = tools_by_name[tool_call['name']].invoke(tool_call)
        tool_msg  = ToolMessage(content=tool_result, tool_call_id = tool_call['id'])
        messages.append(tool_msg)
        f_msg = llm.invoke(messages)
        print(f_msg.content)
else:
    print(ai_msg.content)
    
# 骨架是这四步，一步都不能少：
#
#   1. messages = [HumanMessage(content=question)]
#      → 先把用户的话装进消息列表
#
#   2. ai_msg = llm_with_tools.invoke(messages)
#      → LLM 回你一个带 tool_calls 的消息
#
#   3. messages.append(ai_msg)                    ← 最容易漏的一句！
#      → 把「它要调工具」这件事也记进历史
#      → 为什么必须加？你自己想：如果历史里只有 HumanMessage
#        和 ToolMessage，第二遍 invoke 时 LLM 会看到什么？
#
#   4. 遍历 ai_msg.tool_calls，对每一个：
#        a. 从第 2 步的电话本里，用 tool_call["name"] 取出函数
#        b. 执行它，参数传 tool_call（上一章你用过 .invoke(tool_call)）
#        c. 把结果包成 ToolMessage，append 进 messages
#           注意 ToolMessage 有【两个】必填东西：
#             content       → 工具返回的文字
#             tool_call_id  → 填 tool_call["id"]
#           tool_call_id 是干什么的？想想：如果 LLM 一次性要调 3 个
#           工具，你喂回 3 条 ToolMessage，它怎么知道哪条对应哪一个？
#
#   5. 最后再 invoke 一次 messages，这回 LLM 拿到工具结果，
#      就能用【人话】回答了。打印它的 .content。
#
# TODO(你写)：
# ------------------------------------------------------------


# ============================================================
# 第 5 步：盯住两个地方看
#
#   A. 工具函数的返回值（你的 f-string 拼的那句）
#   B. LLM 最后的 .content（它说的话）
#
#   问自己：这两句是同一句吗？
#   如果不是，LLM 在中间干了什么？
#
#   （这一步你会看到「Agent 是什么」的雏形 ——
#    工具负责查数据，LLM 负责把数据说成人话。）
#
# TODO(你写)：
# ------------------------------------------------------------

# 不是同一句， llm 会结合 human  tool  aimag 得出结果
# ============================================================
# 第 6 步：思考题（不用写代码）
#
#   A. 如果我把第 3 步的问题换成「你好吗」，
#      这段代码会发生什么？会在哪一步崩？还是会平平静静跑完？
#      （提示：ai_msg.tool_calls 是 []，你的 for 循环会怎样？）
#
#   B. 你现在的代码只跑了「一轮」工具调用。
#      如果 LLM 看到工具结果后，觉得还不够，想再调一次工具呢？
#      你的代码支持吗？
#      （这个问题就是文件 4 的标题：连续调用多个 Tool）
# ============================================================
#A平静跑完了
#B,不行,你继续吧