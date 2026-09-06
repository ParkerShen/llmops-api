'''
Date: 2026-09-03 19:12:15
Author: parker
FilePath: \llmops-api\study\3.OutputParser组件使用技巧\1.StrOutputParser使用技巧.py
Description: 
'''
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_deepseek import ChatDeepSeek

import dotenv

dotenv.load_dotenv()

#1.编排提示模板
prompt = ChatPromptTemplate.from_template("{query}")

# 2.创建语言模型
llm = ChatDeepSeek(model="deepseek-chat")

#3.调用大预言模型的生成结果并解析器

parser = StrOutputParser()

#4.调用大语言模型结果并解析
content = parser.invoke(llm.invoke(prompt.invoke({"query":"你好"})))

print(content)