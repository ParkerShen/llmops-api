'''
Date: 2026-09-17 17:59:38
Author: parker
FilePath: \llmops-api\study\53- Tool Calling基础\4.Tool描述与Schema.py
Description: 
'''
from langchain_core.tools import tool
from typing import Annotated
from pydantic import  Field


from typing import Literal
import json

# @tool
# def get_order_info(no: Annotated[int, Field(description="订单号")], user_id: Annotated[int, Field(description="用户id")]):
#     """根据订单号和用户的ID和订单状态和查询订单信息"""
#     return f"{no}号订单信息：用户ID为{user_id}"



# # print(get_order_info.args)


# # print(json.dumps(get_order_info.args, ensure_ascii=False, indent=1))
@tool
def get_order_status(no:Annotated[int, Field(description="订单号")]) ->str:
    """根据订单号查询订单状态"""
    return f"{no}号订单已经支付了"

@tool
def get_order_detail(no:Annotated[int, Field(description="订单号")]) ->str:
    """根据订单号查询订单详情"""
    return f"{no}号订单,用户为沈，订单金额为500元,订单状态为已支付"



# print(get_order_info.args)