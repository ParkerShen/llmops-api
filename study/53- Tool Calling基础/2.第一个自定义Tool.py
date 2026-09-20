'''
Date: 2026-09-17 17:59:17
Author: parker
FilePath: \llmops-api\study\53- Tool Calling基础\2.第一个自定义Tool.py
Description: 
'''


from langchain_core.tools import tool


@tool
def get_order_info(no: int, user_id: int):
    """根据订单号查询订单信息"""
    return f"{no}号订单信息：用户ID为{user_id},订单金额为100元,订单状态为已支付"



print(get_order_info.invoke({"no":1, "user_id": 1}))