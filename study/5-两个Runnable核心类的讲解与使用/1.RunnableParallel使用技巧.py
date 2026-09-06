'''
Date: 2026-09-04 16:27:04
Author: parker
FilePath: \llmops-api\study\5-两个Runnable核心类的讲解与使用\1.RunnableParallel使用技巧.py
Description: 
'''
import dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_deepseek import ChatDeepSeek
from langchain_core.runnables import RunnableParallel
import dotenv

dotenv.load_dotenv()

joke_prompt = ChatPromptTemplate.from_template("请1个关于{subject}的字")
poem_prompt = ChatPromptTemplate.from_template("请1个关于{subject}的诗歌")

llm = ChatDeepSeek(model="deepseek-chat")

parser = StrOutputParser()

joke_chain = joke_prompt | llm | parser
poem_chain = poem_prompt | llm | parser

# 并行链
map_chain =RunnableParallel(joke=joke_chain,poem=poem_chain)
# map_chain =RunnableParallel({
#     "joke": joke_chain,
#        "poem": poem_chain

# })

res = map_chain.invoke({"subject":"AI"})

print(res)



