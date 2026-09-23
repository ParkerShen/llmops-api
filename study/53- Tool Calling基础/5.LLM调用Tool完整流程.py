'''
Date: 2026-09-17 18:00:55
Author: parker
FilePath: \llmops-api\study\53- Tool Calling基础\5.LLM调用Tool完整流程.py
Description: 
'''
import dotenv
from langchain_core.tools import tool
from typing import Annotated
from pydantic import Field
from langchain_core.messages import HumanMessage, ToolMessage

dotenv.load_dotenv()

from langchain_deepseek import ChatDeepSeek

# 1. 创建大语言模型
llm = ChatDeepSeek(model="deepseek-chat", temperature=0)

# 2. 定义工具
@tool
def get_order_status(no: Annotated[int, Field(description="订单号")]) -> str:
    """根据订单号查询订单状态"""
    print(">>> 函数真的被执行了！")
    return f"{no}号订单已经支付了"

@tool
def get_order_detail(no: Annotated[int, Field(description="订单号")]) -> str:
    """根据订单号查询订单详情"""
    return f"{no}号订单,用户为沈，订单金额为500元,订单状态为已支付"

# 3. 绑定工具
llm_tool = llm.bind_tools([get_order_status, get_order_detail])

# 4. 建立工具名称映射字典
tools_by_name = {t.name: t for t in [get_order_status, get_order_detail]}
print(f"tools_by_name:{tools_by_name}")

# 5. 初始化消息列表（只包含用户输入）
messages = [HumanMessage(content="订单1和订单2分别是什么状态？")]


while True:
    ai_msg = llm_tool.invoke(messages)
    print(f"AI决定调用工具：{ai_msg}")
    messages.append(ai_msg)
    # print(f"AI决定调用工具：{messages}")

    if not ai_msg.tool_calls:          # ← 什么条件代表"模型不要工具了"？
        break           # ← 跳出去

    for tool_call in ai_msg.tool_calls:
        selected_tool = tools_by_name[tool_call["name"]]
        tool_result = selected_tool.invoke(tool_call["args"])
        messages.append(ToolMessage(content=str(tool_result), tool_call_id=tool_call["id"]))

print(ai_msg.content) 
