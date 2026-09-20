'''
Date: 2026-09-18 18:08:00
Author: parker
FilePath: \llmops-api\study\54-自定义Tool与参数\1.字符串参数Tool.py
Description: 自定义 Tool 与参数 —— 字符串参数
'''

# ============================================================
# 第 1 步：导入 @tool 装饰器
# 上一章你已经用过它了，想想是哪个模块里的哪个名字。
# TODO(你写)：写 import 语句
# ------------------------------------------------------------
import dotenv
from langchain_core.tools import tool
from typing import Annotated
from pydantic import Field
from langchain_core.messages import HumanMessage, ToolMessage

dotenv.load_dotenv()

from langchain_deepseek import ChatDeepSeek

# ============================================================
# 第 2 步：定义一个「参数是字符串」的 Tool
# 要求：
#   1. 名字你自己起，比如查天气、查用户昵称、翻译单词——只要参数是字符串就行
#   2. 参数的类型标注写 str（对比上一章你写的 int，这是本章的重点）
#   3. 函数体就 return 一句话，不要去真的联网查
#   4. 上一行加中文 docstring，LLM 靠它决定「什么时候该调这个工具」
# TODO(你写)：把下面这个占位函数改成你自己的
# ------------------------------------------------------------
@tool
def get_weather_info(city: str) -> str:
    """传进城市查询当前城市天气"""
    return f"{city}的天气今天31°"


# ============================================================
# 第 3 步：自己调用一次，确认能跑通
# 提示：上一章你用的是 xxx.invoke({"参数名": 值})
# 关键问题：字符串参数在这里该怎么传？和 int 有什么不一样？
# TODO(你写)：写 print(...)
# ------------------------------------------------------------
print(get_weather_info.args)
print(get_weather_info.invoke({"city": "深圳"}))