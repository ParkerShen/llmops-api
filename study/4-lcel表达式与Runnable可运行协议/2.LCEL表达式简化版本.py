'''
Date: 2026-09-04 16:08:59
Author: parker
FilePath: \llmops-api\study\4-lcel表达式与Runnable可运行协议\2.LCEL表达式简化版本.py
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

#创建链

chain = prompt | llm | parser

#调用连得到结果
print(chain.invoke({"query":"讲一个笑话"}))