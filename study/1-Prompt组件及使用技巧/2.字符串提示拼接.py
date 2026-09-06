'''
Date: 2026-09-03 16:11:44
Author: parker
FilePath: \llmops-api\study\1-Prompt组件及使用技巧\2.字符串提示拼接.py
Description: 
'''

from langchain_core.prompts import PromptTemplate

prompt = (
    PromptTemplate.from_template("请讲一个关于{subject}的冷笑话")
    +",让我开心"+
    "\n使用{language}语言"


)
print(prompt.invoke({"subject": "程序员", "language": "中文"}).to_string())
