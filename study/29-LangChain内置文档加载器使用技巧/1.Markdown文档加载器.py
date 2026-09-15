'''
Date: 2026-09-13 15:36:42
Author: parker
FilePath: \llmops-api\study\29-LangChain内置文档加载器使用技巧\1.Markdown文档加载器.py
Description: 
'''
from langchain_community.document_loaders import UnstructuredMarkdownLoader

loader = UnstructuredMarkdownLoader("./项目API资料.md")

documents = loader.load()

print(documents)
print(len(documents))
print(documents[0].metadata)