'''
Date: 2026-09-08
Author: parker
FilePath: \llmops-api\study\11-langchain摘要记忆组件的使用和解析\1.摘要缓冲混合记忆示例.py
Description: 摘要 + 缓冲 混合记忆（Summary-Buffer Hybrid Memory）示例
             —— 用"分步骤 + 注释"的方式讲清它是怎么搭起来的

============================================================
【它解决什么问题】
  只靠"缓冲窗口"（study/10 的做法）有个缺点：窗口外的旧信息会丢。
  只靠"摘要"呢，又丢了近几轮的精确细节。
  混合记忆 = 近几轮原样保留(缓冲) + 更早的浓缩成摘要(Summary)。
  既记得住很久以前的事，又不让每次请求的 token 无限膨胀。

【总体思路 —— 你照着这个写就能懂下面的代码】
  第1步 准备两个模型角色：
        - 对话模型：负责正常回答用户
        - 总结员：同一个模型客串，负责把旧对话"拧干"成摘要
  第2步 设计"会话仓库" session_histories：
        每个 session_id 存一个 dict，里面有 2 样东西：
        { "summary": 滚动摘要(字符串),
          "buffer":   最近几轮的原始消息(列表) }
  第3步 每次用户提问时，把 4 段内容拼给模型：
        [系统设定] → [历史摘要] → [缓冲里的原始对话] → [本次提问]
  第4步 拿到模型回答后，把"提问+回答"这一轮追加进 buffer。
  第5步 归档整理（核心步骤）：buffer 超过设定轮数时，
        - 取出最旧的那一轮
        - 让"总结员"把【旧摘要 + 这一轮】合并成新摘要
        - 把这一轮从 buffer 删掉 → buffer 永远只留最近 N 轮
  第6步 死循环对话，每轮打印 Debug，观察 summary / buffer 如何变化
============================================================
'''

import dotenv
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_deepseek import ChatDeepSeek

dotenv.load_dotenv()

# ==========================================================
# 第1步：创建模型（对话模型 llm，顺便兼任"总结员"）
#        ChatDeepSeek 会自动读 .env 里的 DEEPSEEK_API_KEY
# ==========================================================
llm = ChatDeepSeek(model="deepseek-chat")

# 缓冲最多保留的"轮数"（1 轮 = 用户问 + AI 答 = 2 条消息）
# 超过这个数，最旧的一轮就会被"归档"进摘要。设小一点方便看效果。
MAX_BUFFER_ROUNDS = 3

# ==========================================================
# 第2步：设计会话仓库
#   每个 session 的形态：{ "summary": str, "buffer": [消息...] }
#   summary：已经浓缩好的历史（字符串）
#   buffer ：最近几轮的原始消息（列表，HumanMessage/AIMessage 交替）
# ==========================================================
session_histories: dict = {}

def get_session(session_id: str) -> dict:
    """取某个会话，不存在就新建。返回的都是同样的 dict 结构。"""
    if session_id not in session_histories:
        session_histories[session_id] = {"summary": "", "buffer": []}
    return session_histories[session_id]


# ==========================================================
# 第3步：把"摘要 + 缓冲 + 本次提问"拼成给模型的完整消息列表
# ==========================================================
def build_messages(session: dict, query: str) -> list:
    messages = []
    # 3.1 系统设定（人设）
    messages.append(SystemMessage("你是智谱AI助手的开发人员，请回答用户的问题。"))
    # 3.2 历史摘要（如果没有摘要就跳过）——让模型"记得久"
    if session["summary"]:
        messages.append(SystemMessage(f"【更早的对话摘要，供参考】{session['summary']}"))
    # 3.3 缓冲里的原始对话（最近几轮）——让模型"记得准"
    messages.extend(session["buffer"])
    # 3.4 本次提问
    messages.append(HumanMessage(query))
    return messages


# ==========================================================
# 第4步：让"总结员"把【旧摘要 + 一段新对话】合并成一条新摘要
#   这是"归档"真正干活的地方：让 LLM 自己压缩，而不是简单截断
# ==========================================================
def summarize(old_summary: str, old_round) -> str:
    human_txt = old_round[0].content      # 那一轮的用户发言
    ai_txt = old_round[1].content         # 那一轮的 AI 回答
    prompt = f"""你是一个对话摘要助手。请把下面的"历史摘要"和"新对话"合并成一份
更新后的摘要。要求：
1. 只输出摘要正文，不要任何解释或开场白；
2. 保留关键事实：人物称呼、偏好、重要信息、已确定的事情；
3. 简洁，控制在 150 字以内。

【历史摘要】
{old_summary if old_summary else "（暂无）"}

【新对话】
Human：{human_txt}
AI：{ai_txt}
"""
    # invoke 不走流式，直接拿整段摘要文本
    return llm.invoke([HumanMessage(prompt)]).content.strip()


# ==========================================================
# 第5步：归档整理 —— buffer 超过轮数上限时，把最旧的一轮挪进摘要
#   每轮对话结束后调用一次，保证 buffer 永远不超过 MAX_BUFFER_ROUNDS 轮
# ==========================================================
def archive_oldest(session: dict) -> None:
    while len(session["buffer"]) // 2 > MAX_BUFFER_ROUNDS:   # 缓冲轮数超了？
        oldest_round = session["buffer"][:2]                 # 取出最旧一轮(2条消息)
        del session["buffer"][:2]                            # 从缓冲里删掉
        session["summary"] = summarize(session["summary"], oldest_round)  # 并入摘要


# ==========================================================
# 第6步：死循环对话（输入 q 退出），每轮 Debug 观察记忆怎么变
# ==========================================================
print("=== 摘要+缓冲混合记忆 Demo ===")
print(">>> 前", MAX_BUFFER_ROUNDS, "轮原样保留；再往前的对话会自动被浓缩成摘要。\n")

while True:
    query = input("Human：")
    if query.lower() == "q":
        print("退出对话...")
        break

    # 6.1 取会话，拼消息，交给模型（流式输出）
    session = get_session("user_1")
    response = llm.stream(build_messages(session, query))

    # 6.2 边流式边打印，并收集完整回答
    print("AI：", end="", flush=True)
    answer = ""
    for chunk in response:
        answer += chunk.content
        print(chunk.content, end="", flush=True)
    print()

    # 6.3 把这一轮(提问+回答)追加进 buffer
    session["buffer"].append(HumanMessage(query))
    session["buffer"].append(AIMessage(answer))

    # 6.4 归档整理：缓冲超限就把最旧的挪进摘要
    archive_oldest(session)

    # 6.5 Debug：看看此刻的 summary 和 buffer 分别存了什么
    print(f"\n[Debug] 缓冲轮数={len(session['buffer'])//2} / 上限={MAX_BUFFER_ROUNDS}")
    if session["summary"]:
        print(f"[Debug] 摘要: {session['summary']}")
    print(f"[Debug] 缓冲: {[m.content[:18] + ('...' if len(m.content) > 18 else '') for m in session['buffer']]}\n")
