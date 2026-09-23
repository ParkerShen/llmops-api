'''
Date: 2026-09-22
Author: parker
FilePath: \llmops-api\study\55-多个Tool调用\6.多个Tool综合案例.py
Description: 多个 Tool 调用 —— 综合案例：电商客服 Agent
'''

# ============================================================
# 综合案例：一个能跟你聊天的电商客服 Agent
#
# ------------------------------------------------------------
# 【这一章你学到的所有东西，这个文件全用上】
#
#   文件 1  多个工具的「菜单」（name / description / args）
#   文件 2  LLM 怎么在多个工具里选；required 参数的坑
#   文件 3  执行工具 → ToolMessage → 喂回 → 说人话
#   文件 4  while 圈：多轮调用 + 轮次上限
#   文件 5  try/except：让「失败」也变成 LLM 能读懂的消息
#
# ------------------------------------------------------------
# 【本文件新增的两样东西】
#
#   ① system prompt —— 给 Agent 一条「工作守则」
#   ② input() 对话循环 —— 让它真的能跟人一问一答
#
#   第 ② 样是「聊天」和「Agent」的分界线：
#   前面 5 个文件都是一问一答就跑完，这个文件它会一直等你说话。
# ============================================================

import math
import sys
from typing import Annotated

from langchain_core.tools import tool
from pydantic import BaseModel, Field

import dotenv
import json
from langchain_core.messages import HumanMessage, ToolMessage,SystemMessage

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
class Addr(BaseModel):
    departure_city: str = Field(description="出发城市，例如：深圳")
    arrival_city: str = Field(description="到达城市，例如：北京")

@tool 
def get_logistics_fee(cities:Addr,
                      is_urgent: Annotated[bool, Field(description="是否加急，默认False")] = False,
                      weight: float = Field(description="包裹总重量，单位 kg，例如 3.5")) -> str:
    """
    计算运费

    Args:
     cities: 传进出发城市和到达城市的字典
     is_urgent: 加急 ，额外加8元
     weight: 包裹重量，单位kg

    计算物流运费的工具。
    计价规则：
    1. 首重 1kg（含）以内：10 元
    2. 超出 1kg 的部分：每 kg 加 5 元（不足 1kg 按 1kg 算，向上取整）
    3. 加急：额外 +8 元
    4. 收件城市属于「偏远地区」（新疆、西藏、内蒙古、青海）：额外 +15 元
    
    """
     # 定义偏远地区
    outlying_areas = ["新疆", "西藏", "内蒙古", "青海"]

    #1. 基础运费计算

    base_fee = 10 
    excess_weight = 0
    if weight > 1:
        #向上取整计算续重
        excess_weight = math.ceil(weight - 1)
    total_fee = base_fee + (excess_weight*5)

    if is_urgent:
        total_fee += 8

    arrival_city = cities.arrival_city

    is_remote = any(area in arrival_city for area in outlying_areas)

    if is_remote:
        total_fee += 15

    return f"{cities.departure_city}寄往{cities.arrival_city}，{weight}kg，{'加急件' if is_urgent else '普通件'}，共 {total_fee} 元"


tools = [get_weather_info, get_order_status,get_city_name,get_logistics_fee]
llm = ChatDeepSeek(model="deepseek-chat", temperature=0)
llm_with_tools = llm.bind_tools(tools)

tools_by_name = {t.name: t for t in [get_order_status, get_weather_info,get_city_name,get_logistics_fee]}
SYSTEM_PROMPT = 'a. 你是谁（电商客服）b. 信息不全时【要先追问】，绝对不许自己编造订单号、重量、城市c. 工具查不到的时候，老实告诉用户，别圆场d. 回答用中文，简短一点'

messages = [SystemMessage(content=SYSTEM_PROMPT)]
# ============================================================
# 第 1 步：四个工具
#
# 前两个从文件 5 直接搬过来（get_city_name 记得用 ValueError）。
# 后两个要新写：
#
#   ① get_order_status(order_id)
#        查订单状态。假数据就行。
#        自己决定：订单号不认识的时候，是 return 一句假的，
#                  还是 raise？想想文件 5 的结论。
#
#   ② get_logistics_fee(cities, weight, is_urgent)
#        算运费 —— 54 章第 6 个文件你写过的那套规则，搬过来。
#        规则：首重 1kg 10 元；超出每 kg +5 元（向上取整）；
#              加急 +8；偏远地区（新疆/西藏/内蒙古/青海）+15
#        这是本文件里最「重」的参数 —— 有嵌套对象、有可选参数、有默认值。
#        它存在的意义：让你看清「复杂参数的工具有多容易调用失败」。
#
#   ③ get_product_info(keyword)
#        查商品价格和库存。自己编两三个商品。
#
# ⚠️ 老规矩，加完工具记得两处登记：tools 列表 + tools_by_name 电话本。
#
# TODO(你写)：
# ------------------------------------------------------------


# ============================================================
# 第 2 步：system prompt —— 本文件最重要的新东西
#
# 为什么要它？回想文件 2 那道题：
#
#   用户说「我的快递怎么还没到」，Agent 什么都没调，直接回答。
#   因为 order_id 是必填，用户没给，LLM 就放弃了。
#
#   而一个真实的客服，这时候该说：
#     「请提供一下您的订单号，我帮您查。」
#
#   ——「反问」这个行为，不会自动发生，必须你去要求它。
#
# 写一条 system prompt，把这些守则讲清楚。至少要包含：
#
#   a. 你是谁（电商客服）
#   b. 信息不全时【要先追问】，绝对不许自己编造订单号、重量、城市
#   c. 工具查不到的时候，老实告诉用户，别圆场
#   d. 回答用中文，简短一点
#
# 然后把它放进消息列表的【第一位】：
#
#     messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(...)]
#
# ⚠️ 消息列表是有顺序的，system 必须在最前面。
#    （提示：SystemMessage 和 HumanMessage 从同一个地方 import）
#
# TODO(你写)：
# ------------------------------------------------------------

try:
    while True:
        user_input = input("你：")
        if user_input in ("quit", "exit", "退出"):
            break
        messages.append(HumanMessage(content=user_input))
        ai_msg = llm_with_tools.invoke(messages)
        if not ai_msg.tool_calls:
            print(ai_msg.content) 
            # 注意：这里建议不要 break，否则聊一句就退出了
            # 如果希望连续对话，应该去掉这个 break
            continue 
        messages.append(ai_msg)
        for tool_call in ai_msg.tool_calls:
            try:
                tool_result = tools_by_name[tool_call['name']].invoke(tool_call)
                tool_msg = ToolMessage(content=tool_result, tool_call_id=tool_call['id'])
            except Exception as e:
                tool_msg = ToolMessage(content=f"工具 {tool_call['name']} 调用失败：{e}", tool_call_id=tool_call['id'])
                print(f"工具报错：{e}")
        messages.append(tool_msg)
        f_msg = llm.invoke(messages)
        print(f"客服：{f_msg.content}")

except KeyboardInterrupt:
    print("\n程序已被用户手动终止，再见！")
    sys.exit(0)

# ============================================================
# 第 3 步：对话循环 —— 把「一问一答」变成「一直聊」
#
# 前面 5 个文件都是一次性的：跑一次，得一个答案，结束。
# 现在要能连续对话，所以外面再套一层循环：
#
#     messages = [SystemMessage(...)]      ← 注意：在外面建，只建一次
#     while True:
#         user_input = input("你：")
#         if user_input in ("quit", "exit", "退出"):
#             break
#         messages.append(HumanMessage(content=user_input))
#         ...里面跑你文件 5 那个 Agent 圈...
#         print(f"客服：{最后那句回答}")
#
# 【关键判断】messages 必须建在【外层】循环的外面。
#
#   想一想：如果你写在 while True 里面（每次都被重建），
#   那么用户说「订单 12345」，Agent 查到一半；
#   下一句用户说「那运费多少」，Agent 还记得上一句的上下文吗？
#   → 你试一下就知道「失忆」是什么感觉了。
#
#   这就是「多轮对话」和「一问一答」在代码上的唯一区别：
#   ↓ messages 建立在哪一层。
#
# TODO(你写)：
# ------------------------------------------------------------


# ============================================================
# 第 4 步：把文件 5 那个圈套进对话里
#
# 就是原封不动搬过来，只有一处改动：
#
#   轮次计数器 rounds 要放【Agent 圈】的开头（每轮对话重置），
#   不能放在【对话循环】外面（那样第二句就没额度了）。
#
#   自己判断这个变量该写在哪个位置：
#
#     while True:                ← 对话循环
#         user_input = input()
#         ...
#         rounds = ?             ← 放这儿？
#         while True:            ← Agent 圈
#             rounds += 1
#
# TODO(你写)：
# ------------------------------------------------------------


# ============================================================
# 第 5 步：亲手把它问崩 —— 这才是综合案例的意义
#
# 前 5 个文件都是「喂好料、跑通就行」。现在你要主动找它的毛病。
#
# 按顺序试这 6 句，每句都记下它做了什么、你觉得对不对：
#
#   1. 你好                      → 它该不该调工具？
#   2. 我的快递怎么还没到        → 【重点】它会追问，还是编个订单号？
#   3. 订单 12345                → 它记住了上一句的上下文吗？
#   4. 那 0755 今天天气怎么样     → 它会连着调两个工具吗？
#   5. 我要寄 3 公斤到北京，多少钱 → 参数齐了吗？缺哪个？它怎么办？
#   6. 城市代码 888 天气          → 异常处理生效了吗？它老实说了吗？
#
# 第 2 句和第 5 句是本文件的考点 —— 都是「用户信息不全」，
# 看你的 system prompt 有没有真的起作用。
#
# TODO(你写)：每句的判断
# ------------------------------------------------------------


# ============================================================
# 第 6 步：思考题 —— 这一章的收尾
#
#   A. 你现在手写了一个 Agent。它一共就三件事在转圈：
#        问模型 → 执行工具 → 喂回结果
#      那 LangGraph / AutoGPT 这些框架，多出来的部分是干嘛的？
#
#   B. 你的 Agent 只能调工具。如果让它能「调另一个 Agent」呢？
#      那会发生什么？（提示：工具的本质是什么？）
#
#   C. 回头看这 6 个文件，用一句话总结：
#      写一个 Agent，最难的地方在哪儿？
#
#      （我的答案：不在代码，在于「怎么把话写给 LLM 听」——
#        工具描述、参数说明、错误消息、system prompt，
#        这四处全部是「写给模型看的中文」。代码只是搬运工。）
# ============================================================
