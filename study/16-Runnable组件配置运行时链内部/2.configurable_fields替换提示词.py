'''
Date: 2026-09-09 19:33:14

Author: parker

FilePath: \llmops-api\study\16-Runnable组件配置运行时链内部\2.configurable_fields替换提示词.py

Description:
    使用 configurable_alternatives() 在运行时切换不同的 Prompt
'''

import dotenv

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import ConfigurableField
from langchain_deepseek import ChatDeepSeek

dotenv.load_dotenv()

# 1. 创建两个不同的提示模板

prompt1 = PromptTemplate.from_template(
    "你是一个随机数字生成助手，请生成一个小于{x}的随机整数，只需要告诉我数字，不要解释。"
)

prompt2 = PromptTemplate.from_template(
    "你是一名幽默的程序员，请根据数字{x}讲一个简短的程序员冷笑话。"
)

# 2. 创建可配置的 Prompt
# 默认使用 prompt1
# 运行时可以通过 prompt_type 选择 prompt1 或 prompt2

prompt = prompt1.configurable_alternatives(
    ConfigurableField(
        id="prompt_type",
        name="Prompt Type",
        description="运行时选择使用哪一个提示词"
    ),
    default_key="prompt1",
    prompt2=prompt2
)

# 3. 创建 LLM

llm = ChatDeepSeek(
    model="deepseek-chat"
)

# 4. 构建 Chain

chain = prompt | llm | StrOutputParser()

# 5. 正常调用
# 不传 config，使用默认的 prompt1

content = chain.invoke({"x": 1000})

print("默认 Prompt：")
print(content)

print("==============================")

# 6. 运行时选择 prompt2

content = chain.invoke(
    {"x": 1000},
    config={
        "configurable": {
            "prompt_type": "prompt2"
        }
    }
)

print("切换到 Prompt2：")
print(content)

print("==============================")

# 7. 再次使用 prompt1

content = chain.invoke(
    {"x": 1000},
    config={
        "configurable": {
            "prompt_type": "prompt1"
        }
    }
)

print("切换回 Prompt1：")
print(content)