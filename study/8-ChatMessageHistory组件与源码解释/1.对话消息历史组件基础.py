'''
Date: 2026-09-08 14:27:22
Author: parker
FilePath: \llmops-api\study\8-ChatMessageHistory组件与源码解释\1.对话消息历史组件基础.py
Description: 
'''
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_postgres.chat_message_histories import PostgresChatMessageHistory

chat_history = InMemoryChatMessageHistory()

chat_history.add_user_message("你好我是慕小课")
chat_history.add_ai_message("你好，我是智谱AI助手，有什么可以帮你的吗？")

print(chat_history)
