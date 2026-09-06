'''
Date: 2026-09-03 18:12:35
Author: parker
FilePath: \llmops-api\study\2.model组件及使用技巧\Model流式输出.py
Description: 
'''
from datetime import datetime

from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

#编排prompt
prompt = ChatPromptTemplate.from_messages(
    [
        ("system","你是DeepSeek机器人，请回答用户问题，显示当前时间{now}"),
        ("human","{query}"),
    ]
).partial(now=datetime.now())

#2.创建大语言模型

llm = ChatDeepSeek(model="deepseek-chat")

response =llm.stream(prompt.invoke({"query":"简单介绍llm和llmops吗"}))

for chunk in response:
    print(chunk.content,flush=True,end="")