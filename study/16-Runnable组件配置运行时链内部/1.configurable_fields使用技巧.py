'''
Date: 2026-09-09 19:33:14
Author: parker
FilePath: \llmops-api\study\16-Runnable组件配置运行时链内部\1.configurable_fields使用技巧.py
Description: 
'''
import dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_deepseek import ChatDeepSeek
from langchain_core.runnables import ConfigurableField

dotenv.load_dotenv()

# 1.创建提示模板
prompt = PromptTemplate.from_template("请生成一个小于{x}的随机整数")

# 2.创建LLM大语言模型，并配置temperature参数为可在运行时配置，配置键位llm_temperature
llm = ChatDeepSeek(model="deepseek-chat")

llm = llm.configurable_fields(
     temperature=ConfigurableField(
        id="llm_temperature",
        name="LLM Temperature",
        description="运行时配置LLM的temperature参数"
    )
)

# 3.构建链应用
chain = prompt | llm | StrOutputParser()

# 4.正常调用内容
content = chain.invoke({"x": 1000},config={
        "configurable": {
            "llm_temperature": 0.9
        }
    })
print(content)

print("==============================")
print(llm)

content = chain.invoke({"x": 10000})

print(content)