'''
Date: 2026-09-13 19:14:38
Author: parker
FilePath: \llmops-api\study\33-递归字符文本分割器的使用与运行流程\2.衍生代码分割器.py
Description: 
'''
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter,Language

loader =UnstructuredMarkdownLoader("./demo.py")
documents = loader.load()

text_spliter = RecursiveCharacterTextSplitter.from_language(
 language=Language.PYTHON,
 chunk_size=500,
 chunk_overlap=50,
 add_start_index=True

)
chunks = text_spliter.split_documents(documents)

for chunk in chunks:
    print(f"块大小：{len(chunk.page_content)},元数据：{chunk.metadata}")


print(chunks[0].page_content)
