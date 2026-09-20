'''
Date: 2026-09-17 17:59:38
Author: parker
FilePath: \llmops-api\study\53- Tool Calling基础\3.Tool参数与类型定义.py
Description: 
'''
from langchain_core.tools import tool

from typing import Literal
import json

@tool
def get_order_info(no: int, user_id: int,money: float,status: Literal["已支付", "未支付", "已取消"] = False, is_paid: bool = False):
    """根据订单号和用户的ID和订单状态和查询订单信息"""
    return f"{no}号订单信息：用户ID为{user_id},订单金额为{money}元,订单状态为已支付"



# print(get_order_info.args)


# print(json.dumps(get_order_info.args, ensure_ascii=False, indent=1))

print(get_order_info.args_schema.model_json_schema())