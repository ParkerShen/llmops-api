'''
Date: 2026-09-13 15:08:44
Author: parker
FilePath: \llmops-api\study\28-Document组件与文档加载器\1.Document与TextLoader.py
Description: Document 组件与 TextLoader 文档加载器
'''
from pathlib import Path

from langchain_community.document_loaders import TextLoader

# ---------------------------------------------------------------------------
# 关于路径:为什么 "./电商产品数据.txt" 会找到项目根目录去?
#
# 因为相对路径是相对于**进程的当前工作目录(CWD)**解析的,
# 而不是相对于这个 .py 文件所在的目录。
# 你运行脚本时 CWD 是项目根目录,所以 "./" 就等于 llmops-api/,
# 它自然去根目录找文件了。
#
# 更坑的是:同一个 "./xx.txt",在 PyCharm 里点运行能跑通,
# 换成命令行、换成定时任务、或者部署到服务器上,就可能找不到文件 ——
# 因为那些场景的 CWD 不一样,而这些代码本身一个字都没改。
#
# 正确做法:用 __file__ 拿到"本脚本自己的位置",再拼数据文件名。
#   __file__              当前脚本的路径(可能是相对的)
#   .resolve()            转成绝对路径,顺便把 .. 之类简化掉
#   .parent               取它所在的目录,也就是 28-Document组件与文档加载器/
# 这样无论从哪个目录启动脚本,都能定位到同一个文件。
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

# 1.构建加载器
#    encoding="utf-8" 必须写:Windows 上 Python 默认用 GBK 读文件,
#    不指定的话中文会变成乱码(或者直接 UnicodeDecodeError)
loader = TextLoader(BASE_DIR / "电商产品数据.txt", encoding="utf-8")

print(f"数据文件路径: {BASE_DIR / '电商产品数据.txt'}")
print(f"当前工作目录: {Path.cwd()}   ← 看,和上面的目录不是一回事\n")

# 2.加载数据
documents = loader.load()

print(documents)
print(len(documents))
print(documents[0].metadata)
