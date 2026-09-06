'''
Date: 2026-09-03 16:51:53
Author: parker
FilePath: \llmops-api\study\1-Prompt组件及使用技巧\3.消息身体是模板拼接.py
Description: 
'''
from langchain_core.prompts import ChatPromptTemplate

system_chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system","机器人回复,我叫{username}"),
    ]
)

human_chat_prompt = ChatPromptTemplate.from_messages(
    [("human","{query}")]
)

chat_prompt = system_chat_prompt + human_chat_prompt

print(chat_prompt.invoke(
    {
        "username":"123",
        "query":"xxx"
    }
))
