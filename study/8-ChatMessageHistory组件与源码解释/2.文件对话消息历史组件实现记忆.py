'''
Date: 2026-09-08 14:27:44
Author: parker
FilePath: \llmops-api\study\8-ChatMessageHistory组件与源码解释\2.文件对话消息历史组件实现记忆.py
Description: 使用 FileChatMessageHistory 组件保存对话历史，while True 循环实现多轮人机对话（智谱 ZhipuAI 客户端）
'''
import os

import dotenv
dotenv.load_dotenv()

from langchain_community.chat_message_histories import FileChatMessageHistory
from langchain_core.messages import SystemMessage
from zhipuai import ZhipuAI

# 1. 实例化智谱原生客户端（自动读取 .env 中的 ZHIPUAI_API_KEY）
client = ZhipuAI()
MODEL = os.getenv("ZHIPUAI_MODEL", "glm-5")

# 2. 创建文件型对话消息历史组件，对话落盘到 json 文件，重启后仍有记忆
chat_history = FileChatMessageHistory("./memory.txt")  
# 设定系统人设（仅在历史为空、即首次创建文件时添加）
if not chat_history.messages:
    chat_history.add_message(SystemMessage(content="你是智谱AI机器人，请结合对话历史回答用户问题。"))


# 3. LangChain 的 BaseMessage 转成 ZhipuAI 客户端需要的 {"role","content"} 字典
#    BaseMessage.type：system -> system，human -> user，ai -> assistant
def messages_to_dicts(messages):
    role_map = {"system": "system", "human": "user", "ai": "assistant"}
    return [{"role": role_map[m.type], "content": m.content} for m in messages]


# 4. while True 死循环，实现人机对话
while True:
    # 5. 获取用户输入
    query = input("Human: ")

    # 6. 输入 q 退出对话
    if query.strip().lower() == "q":
        print("对话结束，再见！")
        break

    # 空输入直接跳过，避免发空消息
    if not query.strip():
        continue

    # 7. 把用户输入存入历史
    chat_history.add_user_message(query)

    # 8. 把完整对话历史转成字典，交给智谱客户端生成回复（模型能看到之前所有轮次）
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages_to_dicts(chat_history.messages),
    )
    ai_reply = response.choices[0].message.content

    # 9. 把 AI 回复也存入历史，下一轮即可形成多轮记忆
    chat_history.add_ai_message(ai_reply)

    # 10. 打印 AI 回复
    print("AI:", ai_reply)
