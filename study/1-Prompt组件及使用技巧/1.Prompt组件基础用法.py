'''
Date: 2026-09-03 15:20:24
Author: parker
FilePath: \llmops-api\study\1-Prompt组件及使用技巧\1.Prompt组件基础用法.py
Description: prompt组件基础用法
'''

from langchain_core.prompts import (PromptTemplate, ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate)
from datetime import datetime
from langchain_core.messages import AIMessage

prompt = PromptTemplate.from_template("What is a good name for a company that makes {product}?")
print(prompt.format(product="colorful socks"))  # 输出: What is a good name for a company that makes colorful socks?

prompt_value = prompt.invoke({"product": "colorful socks"})
print(prompt_value.to_string())  # 输出: What is a good name for a company that makes colorful

print(prompt_value.to_messages())

chat_prompt = ChatPromptTemplate.from_messages([
    ("system","你是机器人，当前时间{now}"),
    MessagesPlaceholder("chat_history"),
    HumanMessagePromptTemplate.from_template("请将一个关于{subject}的冷笑话")
]).partial(now = datetime.now())

chat_prompt_value = chat_prompt.invoke({
    
    "chat_history":[
        ("human","我叫母校可"),
       AIMessage("你好，我是xxx，什么能帮助你"),

    ],
    "subject":"程序员",
})

for m in chat_prompt_value.to_messages():
    print(f"[{m.type}] {m.content}")
