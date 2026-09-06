'''
Date: 2026-09-03 18:12:35
Author: parker
FilePath: \llmops-api\study\2.model组件及使用技巧\2.Model预处理.py
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

# ai_message =llm.invoke(prompt.invoke({"query":"现在是几点，讲个程序员冷笑话"}))
# print(ai_message.content)

# print(ai_message.type)

# print(ai_message.response_metadata)
#批量回给机器人，无上下文关系
ai_messages = llm.batch([
    prompt.invoke({"query":"你好,你是"}),
    prompt.invoke({"query":"十个字"}),
])

for ai_messages in ai_messages:
    print(ai_messages.content)
    print("===============================")