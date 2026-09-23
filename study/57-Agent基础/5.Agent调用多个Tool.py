'''
Date: 2026-09-23
Author: parker
FilePath: \llmops-api\study\57-Agent基础\5.Agent调用多个Tool.py
Description: Agent 基础 —— Agent 调用多个 Tool
'''
import math
from typing import Annotated

import dotenv
from langchain.agents.middleware.tool_error import ToolErrorMiddleware
from langchain_core.tools import tool
from langchain_deepseek import ChatDeepSeek
from pydantic import BaseModel, Field
from langchain.agents import create_agent
dotenv.load_dotenv()

llm  = ChatDeepSeek(model="deepseek-chat",temperature=0)
# ============================================================
# 【今天要理解的概念】：多工具时，Agent 会"一次发多张工单"
#
# 文件 4 你观察的是：一条工具失败的消息，模型怎么反应。
# 文件 5 要观察的是另一件事 —— 而且它比"选哪个工具"更有意思：
#
#     【一句话里问了 N 件事，Agent 会发 N 张工单还是一张？】
#
# ------------------------------------------------------------
# 【回想文件 2 那个比喻】
#
#   Tool Calling ≈ 子组件 emit 了一个事件，然后停在那儿。
#
#   那个比喻里藏着一个我们当时没说透的点：
#   子组件 emit 的时候，是 emit 一次，还是可以连着 emit 三次？
#
#   答案是可以。一条 AIMessage 里可以装【多个 tool_calls】。
#
#   这就是多工具场景和单工具场景最本质的区别 ——
#   不是"工具有几个"，而是"一次能发几张工单"。
#
# ------------------------------------------------------------
# 【两种调度策略，你在 55 章见过其中一种】
#
#   ┌── 串行（一个一张）──────────────────────────────┐
#   │  Human → AI(1张工单) → Tool → AI(1张工单) → Tool → AI │
#   │          └─ 等结果 ─┘        └─ 等结果 ─┘            │
#   └──────────────────────────────────────────────────┘
#
#   ┌── 并行（一次多张）──────────────────────────────┐
#   │  Human → AI(2张工单) → Tool → Tool → AI            │
#   │          └─ 同时发出去，两个结果一起回来 ─┘          │
#   └──────────────────────────────────────────────────┘
#
#   两种都合法。选哪种由【模型】决定 —— 这也是 Agent 的特点：
#   连"怎么调度"都是运行时才知道的。
#
# ------------------------------------------------------------
# 【而这里，藏着你 55 章代码的一个 bug】
#
#   打开 study/55-多个Tool调用/6.多个Tool综合案例.py，看这段：
#
#       for tool_call in ai_msg.tool_calls:
#           try:
#               tool_result = tools_by_name[tool_call['name']].invoke(tool_call)
#               tool_msg = ToolMessage(...)
#           except Exception as e:
#               tool_msg = ToolMessage(...)
#       messages.append(tool_msg)      ← ★ 注意这一行的缩进
#
#   这行 append 和 for【对齐】，也就是在循环【外面】。
#
#   一个工单的时候没问题。两个工单的时候呢？
#   → 第一次循环的 tool_msg 被第二次覆盖了，
#     最后只 append 了一份，另一个 tool_call 永远没有回执。
#
#   而 ToolMessage 是【必须一一对应】的：每一个 tool_call_id
#   都必须有一条 ToolMessage 回应它，否则模型会直接报错。
#
#   ⚠️ 所以文件 5 的第一个任务，就是让这个 bug 当场发作。
#      你会亲眼看到"为什么偏偏是这两个工具的时候崩"。
# ============================================================
def on_error(error: Exception, request) -> str:
    """将工具异常转换为模型可读的错误信息。

    两个参数都是必须的 —— 框架内部是 on_error(exc, request) 这样调的。
    request 里装着这次的 tool_call（工具名、参数、id）和当时的 state，
    所以你能写出比"调用失败"信息量更大的错误消息。
    """
    return f"工具 {request.tool_call['name']} 调用失败：{error}"

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



agent = create_agent(
    model = llm,
    tools = [get_weather_info,
get_city_name,
get_order_status,
get_logistics_fee],
system_prompt="工具返回错误时，如实告诉用户你查不到，绝对不许自己编造数据和结果, 回答用中文，简短一点",
    middleware=[ToolErrorMiddleware(on_error)]
)

question = ['火星和深圳天气的天气怎么样，查下深圳区号，火星有城市代码吗，深圳发快递去西藏加急多少钱']


print('=' * 72)
for q in question:
    result = agent.invoke({'messages':[{'role':'user','content':q}]})
    print(f"AI:{len(result['messages'][1].tool_calls)}")
    print("一共几条信息",len(result['messages']))
    

# ============================================================
# 【本文件的任务】—— 让模型自己选择调度策略，你只负责数数
#
# ------------------------------------------------------------
# 第 1 步：三个工具（都从 55 章搬，两分钟）
#
#     get_weather(city)            → 认识 深圳/北京/上海，其他 raise
#     get_order_status(order_id)   → 假订单，比如 12345 / 67890 有数据
#     get_product_price(keyword)   → 假商品，比如 K87 键盘 / 鼠标
#
#     三个工具，覆盖了三种不同的"参数形状"：
#     城市名、订单号、商品关键词。这样模型才不会混淆它们。
#
# ------------------------------------------------------------
# 第 2 步：建 agent
#
#     agent = create_agent(
#         model=llm,
#         tools=[get_weather, get_order_status, get_product_price],
#         middleware=[ToolErrorMiddleware(on_error)],   ← 从文件 4 抄
#     )
#
#     ⚠️ 注意：这里【没有】system_prompt。先不加 ——
#        我们要先看它"裸奔"时怎么调度，加上守则就分不清是它的本能还是守则在起作用。
#
# ------------------------------------------------------------
# 第 3 步：跑 4 句话，每句记录【两个数】
#
#     ① "深圳今天天气怎么样？"                    ← 单工具，1 张工单
#
#     ② "深圳和北京今天天气怎么样？"               ← ★ 测试并行
#
#     ③ "订单 12345 和订单 67890 都到哪了？"       ← ★ 测试并行
#
#     ④ "深圳天气怎么样？顺便帮我查下 K87 的价格"   ← ★★ 跨工具，最难
#
#     每句话记录：
#       a. len(result["messages"])                        ← 总条数
#       b. 每条 AIMessage 的 len(msg.tool_calls)           ← 这条消息里有几张工单
#
#     打印 b 的时候可以这样写：
#
#         if 类型 == "AIMessage":
#             print(f"  这条 AI 发了 {len(msg.tool_calls)} 张工单")
#
# ------------------------------------------------------------
# 【填完表，回答这 5 个问题 —— 这是本文件的全部价值】
#
#   1. ② 的条数是 5 还是 6？
#
#      5 条 = Human → AI(2张) → Tool → Tool → AI        （并行）
#      6 条 = Human → AI(1张) → Tool → AI(1张) → Tool → AI（串行）
#
#      也就是说：光看条数，你就能判断出它用了哪种调度策略。
#      这就是为什么文件 3 我要你数条数 —— 条数是"内部行为的指纹"。
#
#   2. ④ 那句（跨两个不同的工具），它用了哪种？
#      和 ② 一样吗？如果不一样，说明它划分"一件事"的边界在哪？
#
#   3. 【最重要】把 ② 那句的 result["messages"] 逐条打印出来。
#      看那两条 ToolMessage —— 它们的 tool_call_id 一样吗？
#
#      它们的 tool_call_id 分别等于第 2 条消息里两个 tool_calls 的哪个 id？
#
#      → 这一眼，就是 55 章你那个 for 循环在干嘛的全部答案。
#
#   4. 现在回到 55 章那个文件，把 append 的缩进挪进 for 里，
#      然后在 55 章那个文件里问一句"深圳和北京的天气"。
#      你会看到那个 bug 长什么样（多半是一个 API 报错，
#      说某个 tool_call_id 没有对应的回复）。
#
#      ⚠️ 这一步的意义不是"修 bug"，是让你看清：
#         你 55 章代码的正确性，取决于【模型会不会发多张工单】。
#         而那是运行时才知道的事 —— 你写代码的时候不知道。
#
#         这就是"手写 Agent"和"用框架"的又一个差别：
#         框架处理这些边界，你只处理业务。
#
#   5. 如果模型的调度策略每次都不一样（同样的问句，两次跑出不同的条数）——
#      这对你的代码意味着什么？
#      （提示：你的代码能不能写出 if 条数 == 5 这种判断？为什么不能？）
#
# TODO(你写)：
# ------------------------------------------------------------
# 1. 5

# 2. 

# ============================================================
# 【做完之后的思考题】
#
#   文件 3 问过你："Agent 走的步数不确定，那什么时候算结束？谁喊停？"
#
#   现在你有更多证据了：不仅步数不确定，
#   连【一步里包含几件事】都不确定。
#
#   那"一轮对话结束"到底由什么决定？
#   你可以在文件 4 的守则实验里找个线索：
#   模型看到 Error 之后，是谁决定"不再试了，直接回答"的？
#
#   → 文件 6 我们把这个"谁喊停"拆开看。
# TODO(你写)：
# ------------------------------------------------------------
