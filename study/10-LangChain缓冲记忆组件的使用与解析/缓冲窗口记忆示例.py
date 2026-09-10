'''
Date: 2026-09-08 16:52:04
Author: parker
FilePath: \llmops-api\study\10-LangChain缓冲记忆组件的使用与解析\缓冲窗口记忆示例.py
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
    ("system", "你是智谱AI助手的开发人员，请回答用户的问题"),
    MessagesPlaceholder("history"),  # 历史消息占位符
    ("human", "{query}"),
])

# 2. 创建大语言模型
llm = ChatDeepSeek(model="deepseek-chat")

# 3. 构建基础链应用
chain = prompt | llm | StrOutputParser()

# 4. 使用 RunnableWithMessageHistory 包装链，实现缓冲窗口记忆
# 实际生产中通常会用 Redis/数据库 来持久化，这里用内存做演示
session_histories = {}

# 滑动窗口大小：每轮对话 = 1 条用户消息 + 1 条 AI 消息，所以"记住两轮" = 4 条
MAX_HISTORY_MESSAGES = 4

def get_session_history(session_id: str):
    if session_id not in session_histories:
        session_histories[session_id] = InMemoryChatMessageHistory()
    history = session_histories[session_id]
    # 缓冲窗口：只保留最近 N 条消息，更早的超出部分直接丢弃
    history.messages = history.messages[-MAX_HISTORY_MESSAGES:]
    return history

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="query",
    history_messages_key="history",
)

# 5. 死循环构建对话命令行
while True:
    query = input("Human：")
    if query.lower() == "q":
        print("退出对话...")
        break

    print("AI：", end="", flush=True)
    output = ""
    
    # 流式输出并拼接完整回复
    response = chain_with_history.stream(
        {"query": query},
        config={"configurable": {"session_id": "user_1"}}
    )
    
    for chunk in response:
        output += chunk
        print(chunk, end="", flush=True)
    
    print()  # 每次回答结束后换行
    print(f"[Debug] 当前窗口内消息 {len(get_session_history('user_1').messages)} 条：")
    for _msg in get_session_history('user_1').messages:
        _who = "用户" if _msg.type == "human" else "AI"
        _text = _msg.content if isinstance(_msg.content, str) else str(_msg.content)
        print(f"  {_who}: {_text[:50]}")