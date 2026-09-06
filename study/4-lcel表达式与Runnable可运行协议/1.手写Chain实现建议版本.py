'''
Date: 2026-09-04 15:44:37
Author: parker
FilePath: \llmops-api\study\4-lcel表达式与Runnable可运行协议\1.手写Chain实现建议版本.py
Description: 
'''
from typing import Any

import dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_deepseek import ChatDeepSeek
from langchain_core.output_parsers import StrOutputParser

dotenv.load_dotenv()

1.#构建组件
prompt =ChatPromptTemplate.from_template("{query}")
llm = ChatDeepSeek(model="deepseek-chat")
parser = StrOutputParser()

class Chain:
    steps: list = []

    def __init__(self, steps:list):
        self.steps = steps
    def invoke(self, input: Any) ->Any:
        for step in self.steps:
            input =step.invoke(input)
            print("步骤:",step)
            print("输出：",input)
            print("===================")
        return input
chain = Chain([prompt,llm,parser])

#4.执行链并获取结果
print(chain.invoke({"query":"当前模型版本"}))


