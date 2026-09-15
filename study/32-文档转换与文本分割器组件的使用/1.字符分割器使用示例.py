'''
Date: 2026-09-13 18:36:26
Author: parker
FilePath: \llmops-api\study\32-文档转换与文本分割器组件的使用\1.字符分割器使用示例.py
Description: 
'''
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import CharacterTextSplitter

# 1.加载对应的文档
loader = UnstructuredMarkdownLoader("./项目API资料.md")
documents =loader.load()
# 2.创建文本分割器
text_splitter = CharacterTextSplitter(
    separator="\n\n", #分隔符
    chunk_size=500,
    chunk_overlap=50,
)
# 3. 分割文本
chunks = text_splitter.split_documents(documents)

for chunk in chunks:
    print(f"块大小:{len(chunk.page_content)}")

print(len(chunks))
