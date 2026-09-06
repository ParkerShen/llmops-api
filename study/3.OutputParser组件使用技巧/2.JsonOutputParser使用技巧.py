'''
Date: 2026-09-03 19:47:07
Author: parker
FilePath: \llmops-api\study\3.OutputParser组件使用技巧\2.JsonOutputParser使用技巧.py
Description: 业务示例：把一段故事文本 -> 结构化 JSON（人物/地点/文字+氛围标签）
'''
import dotenv
from typing import Literal
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_deepseek import ChatDeepSeek

dotenv.load_dotenv()

# 1. 定义：我想从一段文本里抽出这些字段（业务=文本结构化/自动打标）
class Story(BaseModel):
    person: str = Field(description="故事里出现的所有人名，用、分隔；没有就写'无'")
    place: str = Field(description="故事最主要的那个发生地点，只写一个")
    text: str = Field(description="用一两句话概括故事内容本身，不要评价")
    feeling: Literal["开心", "伤感", "温馨", "悬疑", "平淡"] = Field(
        description="故事的整体氛围，只能从枚举里五选一")

parser = JsonOutputParser(pydantic_object=Story)

# 2. 把 format_instructions 用 partial 焊进模板（模板只剩 query 一个待传变量）
prompt = ChatPromptTemplate.from_template(
    "请阅读下面的故事，并按结构抽取字段。\n{format_instructions}\n\n故事：{query}"
).partial(format_instructions=parser.get_format_instructions())

# 3. 构建模型
llm = ChatDeepSeek(model="deepseek-chat")

# 4. 业务输入：一段原始文本（真实业务里是留言/新闻/文档段落）

print("========== 1) 模型返回的【原始文本】(还没解析) ==========")
ai_msg = llm.invoke(prompt.invoke({"query": "这里随便给我一个包含人物地点时间的故事，关于夏天的"}))
print(ai_msg.content)          # 能看到：模型吐出的是一段 JSON 字符串

print("\n========== 2) parser.parse() 解析成 Python dict ==========")
data = parser.parse(ai_msg.content)
print(f"data", data.get("person"), type(data))
