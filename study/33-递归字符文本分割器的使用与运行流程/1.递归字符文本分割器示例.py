'''
Date: 2026-09-13 19:14:38
Author: parker
FilePath: \llmops-api\study\33-递归字符文本分割器的使用与运行流程\1.递归字符文本分割器示例.py
Description: 
'''
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader =UnstructuredMarkdownLoader("./项目API资料.md")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 50,
    add_start_index = True
)
chunks =  text_splitter.split_documents(documents)
for chunk in chunks:
    print(f"块大小：{len(chunk.page_content)},元数据：{chunk.metadata}")