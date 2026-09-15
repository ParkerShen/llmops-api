'''
Date: 2026-09-15 14:07:59
Author: parker
FilePath: \llmops-api\study\37-VectorStore组件深入学习与检索方法\1.带阈值的相似性搜索.py
Description: 
'''

import dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
# 1. 更换导入：使用 HuggingFace 本地的 Embedding
# 注意：langchain_community.embeddings 里的那个已被标记废弃（0.2.2 起），
# 现在要用独立的 langchain-huggingface 包，跑到 1.x 才会不被移除。
from langchain_huggingface import HuggingFaceEmbeddings

dotenv.load_dotenv()

# 2. 初始化本地 Embedding 模型（首次运行会自动下载模型）
# 推荐用 BAAI/bge-small-zh-v1.5（体积小，中文效果好）
embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")

documents = [
    Document(page_content="笨笨是一只很喜欢睡觉的猫咪", metadata={"page": 1}),
    Document(page_content="我喜欢在夜晚听音乐，这让我感到放松。", metadata={"page": 2}),
    Document(page_content="猫咪在窗台上打盹，看起来非常可爱。", metadata={"page": 3}),
    Document(page_content="学习新技能是每个人都应该追求的目标。", metadata={"page": 4}),
    Document(page_content="我最喜欢的食物是意大利面，尤其是番茄酱的那种。", metadata={"page": 5}),
    Document(page_content="昨晚我做了一个奇怪的梦，梦见自己在太空飞行。", metadata={"page": 6}),
    Document(page_content="我的手机突然关机了，让我有些焦虑。", metadata={"page": 7}),
    Document(page_content="阅读是我每天都会做的事情，我觉得很充实。", metadata={"page": 8}),
    Document(page_content="他们一起计划了一次周末的野餐，希望天气能好。", metadata={"page": 9}),
    Document(page_content="我的狗喜欢追逐球，看起来非常开心。", metadata={"page": 10}),
]

# 3. 构建 FAISS 向量库（这一步不变）
db = FAISS.from_documents(documents, embedding)

# 4. 相似度搜索（这一步不变）
results = db.similarity_search_with_relevance_scores("我有一只猫",score_threshold=0.3,k=1)
print("results", results)
for doc, score in results:
    print(f"相似度: {score:.4f} | 内容: {doc.page_content} | 元数据: {doc.metadata}")