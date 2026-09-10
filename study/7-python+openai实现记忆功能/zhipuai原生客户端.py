'''
Date: 2026-09-07 19:00:00
Author: parker
FilePath: \llmops-api\study\7-python+openai实现记忆功能\zhipuai原生客户端.py
Description: 使用原生 zhipuai SDK 创建智谱客户端，演示同步对话、流式输出与多轮对话
'''
import os

from dotenv import load_dotenv
from zhipuai import ZhipuAI

load_dotenv()

MODEL = os.getenv("ZHIPUAI_MODEL", "glm-5")

# 1. 创建智谱原生客户端（自动读取 .env 中的 ZHIPUAI_API_KEY）
client = ZhipuAI()


def chat(messages):
    """同步对话：传入消息列表，返回助手回复文本"""
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
    )
    return response.choices[0].message.content


def chat_stream(messages):
    """流式对话：逐字返回，适合做打字机效果"""
    stream = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        stream=True,
    )
    for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:
            print(content, end="", flush=True)
    print()


if __name__ == "__main__":
    # 2. 单轮同步对话
    print("=== 单轮同步对话 ===")
    reply = chat([
        {"role": "system", "content": "你是智谱机器人，简洁友好地回答问题。"},
        {"role": "user", "content": "用一句话介绍你自己，再讲个程序员冷笑话"},
    ])
    print(reply)

    # 3. 流式输出
    print("\n=== 流式输出 ===")
    chat_stream([{"role": "user", "content": "讲个程序员冷笑话"}])

    # 4. 多轮对话（手动维护消息历史，为后续记忆功能做准备）
    print("\n=== 多轮对话 ===")
    history = [
        {"role": "system", "content": "你是智谱机器人，简洁友好地回答问题。"},
        {"role": "user", "content": "我叫 parker，记住我的名字"},
    ]
    # 把助手的回复追加进历史，形成完整的上下文
    history.append({"role": "assistant", "content": chat(history)})
    history.append({"role": "user", "content": "我刚才告诉你我叫什么？"})
    print(chat(history))
