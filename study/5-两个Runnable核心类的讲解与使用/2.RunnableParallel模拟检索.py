'''
Date: 2026-09-04 16:44:16
Author: parker
FilePath: \llmops-api\study\5-两个Runnable核心类的讲解与使用\2.RunnableParallel模拟检索.py
Description: 
'''
import dotenv
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_deepseek import ChatDeepSeek
from langchain_core.runnables import RunnableParallel

dotenv.load_dotenv()

def retrieval(query: str) -> str:
    """一个模拟的检索"""
    return "我是慕小课"

prompt = ChatPromptTemplate.from_template("""
<context>
{content}
</context>
用户的提问是{query}""")

#构造大预言模型
llm =ChatDeepSeek(model="deepseek-chat")

parser = StrOutputParser()



chain = {
       "content": lambda x: retrieval(x["query"]),   # 键名要和模板变量 {content} 一致
       "query":    itemgetter("query"),
} | prompt | llm | parser                            # dict 要先过 prompt 变消息，才能喂 llm

content = chain.invoke({"query": "我是谁"})

print(content)

