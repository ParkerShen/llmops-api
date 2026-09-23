'''
Date: 2026-09-21
Author: parker
FilePath: \llmops-api\study\54-自定义Tool与参数\6.自定义物流费用Tool.py
Description: 自定义 Tool 与参数 —— 综合题：物流费用计算
'''
import dotenv
import json
import math
from typing import Annotated, Any, Literal
from langchain_core.tools import tool
from pydantic import Field, BaseModel

from langchain_core.messages import HumanMessage, ToolMessage

dotenv.load_dotenv()

# ============================================================
# 场景
#
# 你在电商公司写一个「AI 客服 Agent」。
# 用户会说这种话：
#
#   「我从深圳寄 3 公斤到北京，多少钱？」
#   「能加急吗？加急要多少？」
#   「寄 3.5 公斤到西藏呢？」
#
# Agent 要靠工具才能算出价格。你的任务：把这个工具写出来。
# ============================================================
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


# ============================================================
# 计价规则（就用这套，别改，不然没法对答案）
#
#   1. 首重 1kg（含）以内：          10 元
#   2. 超出 1kg 的部分：             每 kg 加 5 元
#      （不足 1kg 的零头按 1kg 算，也就是向上取整）
#   3. 加急：                        额外 +8 元
#   4. 收件城市属于「偏远地区」：     额外 +15 元
#
# 偏远地区：新疆、西藏、内蒙古、青海
# ============================================================


# ============================================================
# 第 1 步：先把「偏远地区」这个名单放进代码里
#
# 想一个问题：这份名单，应该做成工具的参数，还是直接写在文件里？
#
#   它会不会因为用户说了什么而变？→ 不会，它是公司规定，固定不变
#   那 LLM 需要知道它吗？        → 不需要，算钱是工具自己的事
#
# 所以它不该是参数。这是本章最后一个、也是最重要的判断：
#
#   【参数 = LLM 从用户话里能提取出来的东西】
#   【不是参数的 = 工具自己去查/去算的东西】
#
# TODO(你写)：定义一个模块级变量，存这四个城市
# ------------------------------------------------------------


# ============================================================
# 第 2 步：想清楚「哪些是参数」
#
# 逐个问自己：这个值，LLM 能从用户那句话里读出来吗？
#
#   寄件城市 → ?        收件城市 → ?
#   重量     → ?        加急     → ?
#   运费单价 → ?        偏远名单 → ?
#
# 一个提醒：重量可能是 3.5 公斤这种小数，
# 想想该用哪个 Python 类型标注（str? int? 还有别的吗？）
#
# 另一个提醒：加急是「是/否」两种状态，用什么类型？
#            而且用户不说的时候应该怎么办？→ 想想你在第 4 个文件学过的
# ------------------------------------------------------------


# ============================================================
# 第 3 步：写工具函数
#
# 硬性要求：
#   1. 每个参数的 docstring 都要写清楚它是什么
#   2. 写清楚「用户没说的时候」是什么行为
#   3. 函数体把价格算出来，return 一句人话
#      （比如「深圳寄往北京 3.0kg，普通件，共 20 元」）
#
# 小提示：向上取整用 math.ceil，记得 import math
#        首重和续重的算法，建议先自己在纸上算两个例子验证
#
# TODO(你写)：
# ------------------------------------------------------------


# ============================================================
# 第 4 步：自测（这一步别跳！）
#
# 先自己手算答案，再写 invoke 对答案。至少测四种：
#
#   A. 正好 1kg，普通件，非偏远
#   B. 3.5kg，普通件，非偏远     ← 考小数和向上取整
#   C. 3.5kg，加急，非偏远
#   D. 3.5kg，普通件，收件地是西藏  ← 考偏远附加费
#
# 手算 → 跑 → 对不上就找原因。这一步是唯一能证明你写对的方式。
#
# TODO(你写)：
# ------------------------------------------------------------
print(get_logistics_fee.invoke({"cities": {"departure_city": "深圳", "arrival_city": "北京"}, "weight": 1.0}))
print(get_logistics_fee.invoke({"cities": {"departure_city": "深圳", "arrival_city": "北京"}, "weight": 3.5}))
print(get_logistics_fee.invoke({"cities": {"departure_city": "深圳", "arrival_city": "北京"}, "weight": 3.5, "is_urgent": True}))
print(get_logistics_fee.invoke({"cities": {"departure_city": "深圳", "arrival_city": "拉萨"}, "weight": 3.5}))


# ============================================================
# 第 5 步：打印完整 schema，用 LLM 的眼睛看一遍
#
# 把自己当成一个从没见过这个工具的大模型，只看 schema，回答：
#
#   A. 哪些参数是 required？哪些不是？
#   B. 光看 schema，你能准确知道「加急」该填什么吗？
#      （提示：看它的 type，LLM 会看到什么？）
#   C. 有没有哪个参数，光靠 schema 里的 description，
#      你还是猜不到该填什么值？→ 那就是要给 LLM 补说明的地方
#
# TODO(你写)：
# ------------------------------------------------------------

from langchain_deepseek import ChatDeepSeek


llm  = ChatDeepSeek(model="deepseek-chat",temperature=0)

llm_with_tools = llm.bind_tools([get_logistics_fee])



print(json.dumps(get_logistics_fee.args_schema.model_json_schema(), indent=2, ensure_ascii=False))

user_questions = [
"我从深圳寄 3 公斤到北京，多少钱？",
"能加急吗？加急要多少",
"寄 3.5 公斤到西藏呢？"
]

for question in user_questions:
    print(f"\n{'='*50}")
    print(f"用户提问: {question}")
    print('='*50)


    messages = [HumanMessage(content=question)]

    ai_msg = llm_with_tools.invoke(messages)

    messages.append(ai_msg)
    # 判断是否工具调用
    if ai_msg.tool_calls:
         print(f"\n模型决定调用工具: {ai_msg.tool_calls}")

         for tool_call in ai_msg.tool_calls:
             tool_result = get_logistics_fee.invoke(tool_call)

             print(f"工具执行结果:{tool_result}")

             tool_msg =ToolMessage(content=str(tool_result),tool_call_id =tool_call["id"])

             messages.append(tool_msg)

             final_response = llm_with_tools.invoke(messages)
             print(f"\n最终回答:{final_response.content}")
    else:
        #模型认为不调用工具，直接回答
        print(f"\n模型直接回答:{ai_msg.content}")
        

# ============================================================
# 第 6 步：思考题（不用写代码）
#
# 现在把这个工具交给真的 LLM 用。用户说：
#
#   「我有个 2 公斤的包裹要寄到北京」
#
# LLM 会怎么填「寄件城市」？
#
# 用户没说啊。但 schema 里它是 required，LLM 必须填点什么。
# 你觉得它会怎么办？
#
# 结论：required 的意思是「LLM 必须填」，不是「用户必须说」。
#       这两件事不一样 —— 想清楚这个区别，你就明白为什么
#       客服 Agent 的提示词里总要说「信息不全时要先追问」。
# ============================================================
