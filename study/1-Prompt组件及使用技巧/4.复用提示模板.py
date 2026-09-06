'''
Date: 2026-09-03 17:02:43
Author: parker
FilePath: \llmops-api\study\1-Prompt组件及使用技巧\4.复用提示模板.py
Description: 
'''

from langchain_core.prompts import PromptTemplate,PipelinePromptTemplate

full_template = PromptTemplate.from_template("""{instruction}{example}{start}""")

#描述模板
insruction_prompt = PromptTemplate.from_template("你正在模拟{person}")

#示例模板
example_prompt =PromptTemplate.from_template("""下面是交互例子：
Q:{example_q}
A:{examplez_a}
""")

#开始模板
start_prompt =PromptTemplate.from_template("""
现在，你是一个正式的人，请回答下列问题：
Q：{input}
A：
""")

PipelinePromptTemplate(
    fanal
)

