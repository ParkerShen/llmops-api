'''
Date: 2026-09-08 16:52:04
Author: parker
FilePath: \llmops-api\study\15-Runnable组件动态添加默认调用\1.bind函数使用技巧.py
Description: 
'''
import dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_deepseek import ChatDeepSeek
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory

dotenv.load_dotenv()

# 1. 创建提示模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是智谱AI助手的开发人员，请回答用户的问题,用户输入什么你就重复他的问题，不要有其他的回答，注意不要有其他的回答"),
   
    ("human", "{query}"),
])

# 2. 创建大语言模型
llm = ChatDeepSeek(model="deepseek-chat")

# 3. 构建基础链应用
chain = prompt | llm.bind(stop="o") | StrOutputParser()


content = chain.invoke({"query": "hello world"})

print(content)