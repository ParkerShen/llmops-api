'''
Date: 2026-09-22
Author: parker
FilePath: \llmops-api\study\55-多个Tool调用\4.连续调用多个Tool.py
Description: 多个 Tool 调用 —— 连续调用多个 Tool
'''

# ============================================================
# 本文件要搞明白的一件事：
#
#   文件 3 的流程是「一条直线」：
#     问 → 调工具 → 喂回 → 说人话 → 结束
#
#   但真实的 Agent 经常不止一轮。比如用户说：
#     「用户 U001 所在的城市今天天气怎么样？」
#
#   你手里有 get_user_city(U001) → 深圳
#               get_weather_info(深圳) → 晴
#
#   LLM 得先调第一个拿到城市，再拿这个结果去调第二个。
#   一轮根本不够。
#
#   所以本文件要把那条直线，掰成一个【能转的圈】。
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
# 第 1 步：先把文件 3 那套完整搬过来，并且真的跑通
#
# 复制：两个工具、tools_by_name、llm_with_tools、第 4 步那段流程
#
# ⚠️ 文件 3 你选了「你好吗」，工具那段代码从没执行过。
#    这次搬过来第一件事就是跑一遍，看它到底出不出结果。
#    代码没跑过，就等于没写。
#
# 顺手把文件 3 里那 2 个坑修掉：
#   - f_msg / print 挪出 for 循环（为什么，文件 3 末尾讲了）
#   - 删掉多余的 if tool_call:
#
# TODO(你写)：
# ------------------------------------------------------------
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
        return "未知城市"

@tool
def get_order_status(order_id: Annotated[str, Field(description="当前订单号")]) -> str:
    """用户查询订单信息"""
    return f"订单{order_id}的状态是已支付"
tools = [get_weather_info, get_order_status,get_city_name]
llm = ChatDeepSeek(model="deepseek-chat", temperature=0)
llm_with_tools = llm.bind_tools(tools)

tools_by_name = {t.name: t for t in [get_order_status, get_weather_info,get_city_name]}



# ============================================================
# 第 2 步：加第 3 个工具 —— 这是本文件的实验器材
#
# 为什么非加不可？
#   光靠「查订单」和「查天气」这两个工具，LLM 一轮就调完了，
#   永远走不到第二轮。
#   你需要一个【中间结果】——它的返回值，是另一个工具的输入。
#
# 要加的工具：
#
#   get_user_city(user_id) → 返回这个用户所在的城市
#
#   比如 get_user_city("U001") 返回 "深圳"
#
#   写成假数据就行：用一个字典把用户和城市对起来。
#   自己判断：docstring 该怎么写，LLM 才知道什么时候找它。
#
# ⚠️ 新工具加进来之后，别忘了两件事：
#   1. tools 列表要更新   2. tools_by_name 那张电话本要更新
#   （电话本你用推导式写的，改起来很省事 —— 这正是它的好处）
#
# TODO(你写)：
# ------------------------------------------------------------
question = "帮我 查询 0755 的天气 ，然后 订单号是12345内订单状态"
messages = [HumanMessage(content=question)]
rounds = 0  
while True:
    rounds += 1                   # 循环里，第一行
    if rounds > 5:
        print("达到最大轮次，提前结束")
        break
    ai_msg  = llm_with_tools.invoke(messages)
    if not ai_msg.tool_calls:
        break
    messages.append(ai_msg)
    if ai_msg.tool_calls:
        for tool_call in ai_msg.tool_calls:
            print(f"tool_call:{tool_call}")
            tool_result = tools_by_name[tool_call['name']].invoke(tool_call)
            tool_msg  = ToolMessage(content=tool_result, tool_call_id = tool_call['id'])
            messages.append(tool_msg)
        # print(f"messages:{messages}")
        f_msg = llm.invoke(messages)
        print(f_msg.content)
    else:
        print(ai_msg.content)
    # a. 结束
    # b. 填llm 直接 break 了
    #C.需要有多轮ai回复，不然推出循环了
    #d 

# ============================================================
# 第 3 步：实验 A —— 一句话，一轮里调 2 个工具
#
# 问题：
#   「北京今天天气怎么样？顺便帮我查下订单 12345 付款了吗」
#
# 这一句里有两个需求，LLM 可能一次性返回 2 个 tool_call。
# 先跑，把 ai_msg.tool_calls 打印出来，看它给了几个。
#
# 然后重点来了 —— 回到文件 3 那个坑：
#   如果你把 llm.invoke(messages) 写在 for 循环体【里面】，
#   调 2 个工具时会发生什么？
#   它会在只追加了第 1 条 ToolMessage 之后就去问 LLM，
#   LLM 那一轮看到的是残缺的历史。
#
#   所以正确做法：for 循环只管【攒结果】，攒齐了再 invoke 一次。
#
# TODO(你写)：
# ------------------------------------------------------------


# ============================================================
# 第 4 步：实验 B —— 一句话，需要连着调两个工具（多轮）
#
# 问题：
#   「用户 U001 所在的城市今天天气怎么样？」
#
# 期望它这么走：
#   第 1 轮：调 get_user_city("U001")  → "深圳"
#   第 2 轮：调 get_weather_info("深圳") → "晴天"
#   第 3 轮：不要工具了，说人话
#
# 先用文件 3 那条「直线」跑它，把每一轮的 tool_calls 打印出来。
# 你应该会看到：直线跑完第 1 轮就结束了，第 2 轮根本没发生。
#
# 亲眼看它断在哪，你才知道 while 循环是为了解决什么。
#
# TODO(你写)：
# ------------------------------------------------------------


# ============================================================
# 第 5 步：把直线掰成圈 —— 本文件的核心
#
# 骨架长这样（这不是完整代码，缺的地方你补）：
#
#   while True:
#       ai_msg = ____.invoke(messages)
#       messages.append(ai_msg)
#
#       if not ai_msg.tool_calls:
#           print(ai_msg.content)
#           ______          ← 这里该写什么？（一个字的关键字）
#
#       for tool_call in ai_msg.tool_calls:
#           ...执行工具、append ToolMessage...
#
#   三个判断题，一个一个想清楚，别跳：
#
#   a. 为什么退出条件是 `if not tool_calls`？
#      tool_calls 为空说明 LLM 想干什么？
#
#   b. invoke 的那个 ____ 应该填 llm 还是 llm_with_tools？
#      提示：如果填 llm，第二轮 LLM 还有本事调工具吗？
#
#   c. 为什么 messages.append(ai_msg) 必须在循环【里面】？
#      （文件 3 是一轮，放哪都行；现在多轮了，放外面会怎样？）
#
#   d. 那个 for 循环跑完之后，为什么不能再 invoke 一次？
#      （对照文件 3：文件 3 是在哪 invoke 的？现在为什么不用了？）
#
# TODO(你写)：
# ------------------------------------------------------------


# ============================================================
# 第 6 步：给这个圈装个刹车
#
# 现在你的 while True 是【没有上限】的。
# 万一 LLM 抽风，一直说要调工具，你这个程序会转到天荒地老，
# 而且每转一圈都在烧钱（每次 invoke 都是一次真实的 API 调用）。
#
# 加一个计数器，转够 N 轮就强行退出（N 取 5 左右就行）。
# 退出时打印一句人话，比如「达到最大轮次，提前结束」。
#
# 这个东西在真实框架里有个正式名字，叫 max_iterations
# （LangGraph 的 create_react_agent 就有这个参数）。
# 你现在是手写出来的 —— 想明白它为什么必须有。
#
# TODO(你写)：
# ------------------------------------------------------------


# ============================================================
# 第 7 步：思考题（不用写代码）
#
#   A. 你现在是手写的这个圈。回头看看这 40 行：
#      能不能一句话说清「Agent 的循环」是什么？
#
#   B. 如果第 3 步那句（一句话两个需求）在你的 while 里跑，
#      它会走几轮？你的代码扛得住吗？
#
#   C. 你现在能想到的坑，哪个最危险？
#      （提示：想想如果某个工具函数内部报错了会怎样 ——
#        exception 一抛，while 直接崩，LLM 永远收不到结果。
#        这就是文件 5 的标题。）
# ============================================================
