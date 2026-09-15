'''
Date: 2026-09-13 19:48:38
Author: parker
FilePath: \llmops-api\study\33-递归字符文本分割器的使用与运行流程\3.中文场景下的递归示例.py
Description: 
'''
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1.创建加载器和文本分割器
loader = UnstructuredMarkdownLoader("./项目API资料.md")
separators = [
    "\n\n",
    "\n",
    "。|！|？",
    "\.\s|\!\s|\?\s",  # 英文标点符号后面通常需要加空格
    ";|;\s",
    ",|,\s",
    " ",
    ""
]
text_splitter = RecursiveCharacterTextSplitter(
  separators= separators,
  is_separator_regex=True,
  chunk_size=500,
  chunk_overlap=50,
  add_start_index=True,
 )

# 2.加载文档与分割
documents = loader.load()

chunks = text_splitter.split_documents(documents)

for chunk in chunks:
    print(f"块大小: {len(chunk.page_content)}")

print(chunks[2].page_content)
