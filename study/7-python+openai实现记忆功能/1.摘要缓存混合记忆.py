'''
Date: 2026-09-07 18:52:05
Author: parker
FilePath: \llmops-api\study\7-python+openai实现记忆功能\1.摘要缓存混合记忆.py
Description: 使用原生 zhipuai SDK + 自实现 摘要缓冲混合记忆 实现多轮人机对话
'''
import os

import tiktoken
from dotenv import load_dotenv
from zhipuai import ZhipuAI

load_dotenv()

MODEL = os.getenv("ZHIPUAI_MODEL", "glm-5")

# 1. 创建智谱原生客户端（自动读取 .env 中的 ZHIPUAI_API_KEY）
client = ZhipuAI()


class ConversationSummaryBufferMemory:
    """摘要缓冲混合记忆类：近期对话原样保留，超出 max_tokens 的旧对话被压缩进摘要"""

    def __init__(self, max_tokens=600):
        # 1. max_tokens：缓冲区上限，超过则触发摘要
        self.max_tokens = max_tokens
        # 2. summary：已压缩的旧对话摘要
        self.summary = ""
        # 3. chat_histories：近期保留的原始对话 list[dict]
        self.chat_histories = []
        # tiktoken 编码器，用于估算 token 数（cl100k 对 GLM 仅为近似）
        self.encoding = tiktoken.get_encoding("cl100k_base")

    # 4. get_num_tokens：计算传入文本的 token 数
    def get_num_tokens(self, text):
        return len(self.encoding.encode(text))

    # 6. get_buffer_string：把历史对话转换成字符串
    def get_buffer_string(self, messages):
        return "\n".join(
            f"{'Human' if m['role'] == 'user' else 'AI'}: {m['content']}"
            for m in messages
        )

    # 8. summary_text：把旧摘要和被移出的对话生成新摘要
    def summary_text(self, summary, conversation):
        prompt = (
            "你的任务是把对话内容逐步总结成一份摘要。"
            "在已有摘要的基础上整合新增对话，输出更完整、简洁的新摘要。\n\n"
            f"当前摘要：\n{summary or '（暂无）'}\n\n"
            f"新增对话：\n{conversation}\n\n"
            "新摘要："
        )
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content

    # 5. save_context：存入新一轮对话，超限则把最旧部分并入摘要
    def save_context(self, input_text, output_text):
        # 先把本轮对话追加进缓冲
        self.chat_histories.append({"role": "user", "content": input_text})
        self.chat_histories.append({"role": "assistant", "content": output_text})

        # 缓冲区超限时，逐条把最旧对话移出，凑齐后一次性并入摘要
        pruned = []
        while self.chat_histories and self.get_num_tokens(
            self.get_buffer_string(self.chat_histories)
        ) > self.max_tokens:
            pruned.append(self.chat_histories.pop(0))
        if pruned:
            self.summary = self.summary_text(self.summary, self.get_buffer_string(pruned))

    # 7. load_memory_variables：加载记忆变量信息供 prompt 使用
    def load_memory_variables(self):
        buffer = self.get_buffer_string(self.chat_histories)
        history = ""
        if self.summary:
            history += f"之前的对话摘要：\n{self.summary}\n\n"
        if buffer:
            history += f"近期对话：\n{buffer}\n"
        return {"history": history}


if __name__ == "__main__":
    # 2. 初始化记忆，缓冲上限 600 token（可用 MEMORY_MAX_TOKENS 覆盖）
    memory = ConversationSummaryBufferMemory(
        max_tokens=int(os.getenv("MEMORY_MAX_TOKENS", "600"))
    )

    # 3. 死循环用于人机对话
    while True:
        # 4. 获取人类的输入
        query = input('Human: ')

        # 5. 输入 q 则退出
        if query == 'q':
            break

        # 6. 加载记忆并组装 messages
        history = memory.load_memory_variables().get("history", "")
        messages = []
        if history:
            messages.append({"role": "system", "content": history})
        messages.append({"role": "user", "content": query})

        # 7. 向智谱接口发起请求，流式获取 ai 生成的内容
        completion = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            stream=True,
        )

        print("AI: ", end="")
        ai_reply = ""
        for chunk in completion:
            if chunk.choices[0].delta.content is not None:
                text = chunk.choices[0].delta.content
                print(text, end="", flush=True)
                ai_reply += text
        print("\n")

        # 8. 保存本轮上下文（超限会自动触发摘要）
        memory.save_context(query, ai_reply)

    # 退出后打印最终记忆状态，方便观察摘要与缓冲
    print("===== 最终记忆状态 =====")
    print("摘要：", memory.summary)
    print("缓冲：", memory.get_buffer_string(memory.chat_histories))
