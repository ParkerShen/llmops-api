'''
Date: 2026-09-17 18:02:54
Author: parker
FilePath: \llmops-api\study\53- Tool Calling基础\1.普通函数与Tool的区别.py
Description: 
'''
# 53 - Tool Calling 基础
# 第 1 课：普通函数 与 Tool 的区别
#
# 【本课要理解的一件事】
#   普通函数：写给「人 / 程序」调用的。
#   Tool    ：写给「大模型」调用的。
#
#   大模型读不懂 Python 代码，它只能读懂三样东西：
#       名字(name) + 说明(description) + 参数表(args_schema)
#   所以 Tool 的本质 = 一个普通函数 + 一份写给大模型看的说明书。
#
#   这一课先不碰 LangChain，只写「普通函数」。
#   下一课我们再把它变成 Tool，你就能一眼看出多出来了什么。


# 【任务 1】定义一个普通函数
#   - 函数名：get_weather
#   - 参数：city
#   - 返回值：一句天气描述字符串（先随便编，不用真的联网查）


# def get_weather(city):
#     # TODO: 返回一句天气描述
#     return f"{city}今天天气晴朗，温度25°C"
#     ...


# # 【任务 2】调用它，并打印结果
# #   传入 "北京"

# # TODO: 调用 get_weather 并 print 结果

# print(f"{get_weather('北京')}")

# print(get_weather)

from langchain_core.tools import tool

@tool
def get_weather_tool(city: str) -> str:
   """传进来的城市，可以查询到当前城市的天气"""
   return f"{city}今天天气晴朗，温度25"




print(get_weather_tool.invoke({"city": "北京"}))

