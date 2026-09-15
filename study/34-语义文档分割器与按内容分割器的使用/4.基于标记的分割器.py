'''
Date: 2026-09-13 21:00:00
Author: parker
FilePath: \llmops-api\study\34-语义文档分割器与按内容分割器的使用\4.基于标记的分割器.py
Description: 基于标记(Markup)的分割器 —— MarkdownHeaderTextSplitter / HTMLHeaderTextSplitter

运行方式(在项目根目录):
    python "study/34-语义文档分割器与按内容分割器的使用/4.基于标记的分割器.py"
'''

# ============================================================================
# 第 0 部分:什么是"基于标记"的分割器?
# ============================================================================
#
# 前面学的所有分割器,盯着的都是"文本"本身:数够多少字就切、遇到什么符号就切。
#
# 但技术文档、产品手册、API 文档这类内容,天生就带着**层级结构**:
#
#     # 第一章
#     ## 1.1 安装
#     ## 1.2 配置
#
# 如果按字数硬切,很可能把 "## 1.2 配置" 和它下面的内容切开 ——
# 结果检索出来的那段话**没有标题**,模型看着一段没头没尾的文字,
# 根本不知道在讲哪个功能。
#
# "基于标记的分割器"就是来解决这个的:它按文档的**结构标记**(Markdown 的 #、
# HTML 的 <h1>~<h6>)来切,保证每一块都带着自己的"章节归属"。
#
# ===== 一个重要前提(先记住,第 3 部分会展开)=====
# 这类分割器**没有 chunk_size 参数**。
# 它们只负责"按结构切",完全不负责"控制块的大小"。
# 一个 ## 标题下面如果写了三千字,它就给你三千字一块,眼睛都不眨。


# ============================================================================
# 第 1 部分:MarkdownHeaderTextSplitter
# ============================================================================

from langchain_text_splitters import (
    HTMLHeaderTextSplitter,
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

markdown_doc = """# 项目 API 资料

本文档描述系统对外提供的接口。

## 用户模块

### 用户登录
POST /api/login,传入用户名和密码,返回 token。

### 用户注册
POST /api/register,需要邮箱验证。

## 商品模块

### 商品列表
GET /api/products,支持分页和关键词搜索。
"""

# headers_to_split_on 是一组 (标记, 元数据字段名) 的对应关系。
# 左边是 Markdown 里的标记,右边是你想给它在 metadata 里起的名字。
md_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("#", "一级标题"),
        ("##", "二级标题"),
        ("###", "三级标题"),
    ],
    strip_headers=True,   # 默认值:把标题文字从正文里去掉(它已经进 metadata 了)
)

md_chunks = md_splitter.split_text(markdown_doc)

print("=" * 70)
print("第 1 部分:MarkdownHeaderTextSplitter")
print("=" * 70)
print(f"切成 {len(md_chunks)} 块\n")
for i, c in enumerate(md_chunks):
    print(f"  [{i}] 正文: {c.page_content}")
    print(f"       metadata: {c.metadata}")

# 划重点:metadata 里的标题是**层层累积**的。
#   看 [1] 那块 —— {'一级标题': '项目 API 资料', '二级标题': '用户模块', '三级标题': '用户登录'}
#   它把从根到这片叶子的完整路径都记下来了。
#   检索命中这一块时,你能立刻知道"这是 用户模块 下的 用户登录 接口",而不是一段孤立的文字。
#
#   这个"路径"还有个用法:可以直接拼进给模型的上下文里,比如
#       f"[{c.metadata['一级标题']} > {c.metadata['二级标题']}] {c.page_content}"
#
# 另一个细节:strip_headers=True 让正文里看不到 "### 用户登录" 这行字。
# 想保留的话设成 False。


# ============================================================================
# 第 2 部分:HTMLHeaderTextSplitter
# ============================================================================
# 爬来的网页、导出的 HTML 文档,用的是 <h1>~<h6> 标签而不是 # 号。
# 用法几乎一样,只是把 "#" 换成 "h1" 这样的标签名。

html_doc = """<html><body>
<h1>DeepSeek 接口文档</h1>
<p>本页介绍对话接口的基本用法。</p>
<h2>鉴权方式</h2>
<p>所有请求都需要在 Header 里带上 Authorization: Bearer sk-xxx。</p>
<h2>请求示例</h2>
<p>使用 POST 方法请求 /chat/completions 端点。</p>
</body></html>"""

html_splitter = HTMLHeaderTextSplitter(
    headers_to_split_on=[
        ("h1", "一级标题"),
        ("h2", "二级标题"),
    ],
)

html_chunks = html_splitter.split_text(html_doc)

print()
print("=" * 70)
print("第 2 部分:HTMLHeaderTextSplitter")
print("=" * 70)
print(f"切成 {len(html_chunks)} 块\n")
for i, c in enumerate(html_chunks):
    print(f"  [{i}] 正文: {c.page_content}")
    print(f"       metadata: {c.metadata}")

# 对比一下 Markdown 版的输出,有个**重要差异**:
#   HTML 版的标题文字**会单独成为一块**,而且会保留在正文里。
#   你看 [0] 是 'DeepSeek 接口文档'(标题本身),
#   [2] 是 '鉴权方式' 然后 [3] 才是 '所有请求都需要...'。
#
#   而 Markdown 版会把标题从正文中剥离、只放进 metadata。
#
#   为什么不一样?因为 HTML 的 <h1> 标签在源码里就是一个独立元素,
#   解析器忠实地把它解析成了一个单独的段落。这不是 bug,是设计差异。
#
#   实际影响:HTML 切出来会有一些"只有标题没有内容"的碎片块。
#   一般不用特意处理,或者用过滤把长度过短的块丢掉。


# ============================================================================
# 第 3 部分:核心 —— 它们必须配合递归分割器使用(两级分割)
# ============================================================================
# 这是这一节最重要的部分。
#
# MarkdownHeaderTextSplitter 和 HTMLHeaderTextSplitter **没有 chunk_size 参数**。
# 块多大,完全取决于原文某个小节写了多少字。
#
# 现实中的技术文档,一个小节写几千字太正常了。这时你会得到一块 3000 字的 chunk:
#   - 向量化时会被 embedding 模型**静默截断**(模型有最大输入长度)
#   - 塞进 LLM 上下文时挤掉别的内容
#   - 检索精度下降(一块里混了太多主题)
#
# 正确姿势:**两级分割**
#   第一级:按标记切 → 保证不跨章节,metadata 带好路径
#   第二级:对过大的块再用 RecursiveCharacterTextSplitter 按字数切
#
# 顺序不能反!先按结构、再按字数。
# 反过来的话,第二级的切分会把章节边界切碎,第一级就白做了。

long_section_doc = """# 使用手册

## 快速开始
这是一段很短的说明。

## 详细配置说明

配置项一共分为五类。第一类是网络配置,包括主机地址、端口号、超时时间等参数,
这些参数决定了客户端如何连接到服务端。第二类是认证配置,包括 API Key、
密钥文件路径、签名算法类型。第三类是重试配置,包括最大重试次数、重试间隔、
退避策略选择。第四类是日志配置,包括日志级别、日志文件路径、日志轮转策略。
第五类是缓存配置,包括缓存开关、缓存目录、缓存过期时间。

在实际部署时,建议把网络配置和认证配置放在环境变量里,不要硬编码到代码中。
日志级别在生产环境建议设为 INFO,开发调试时设为 DEBUG。
缓存配置在容器化部署时需要特别注意,要确保缓存目录是可写的持久化卷。
"""

# ---- 第一级:按 Markdown 标题切 ----
level1_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "一级标题"), ("##", "二级标题")],
)
level1_chunks = level1_splitter.split_text(long_section_doc)

print()
print("=" * 70)
print("第 3 部分:两级分割")
print("=" * 70)
print()
print("第一级(按标记切)的结果:")
for i, c in enumerate(level1_chunks):
    print(f"  [{i}] 长度={len(c.page_content):4d}  metadata={c.metadata}")
    print(f"       {c.page_content[:40]}...")

# 看到没?["详细配置说明"] 那一块 280 个字,远超合理的 chunk 大小。
# 而且它切不出来 —— 因为这个标题下面就没有更细的标题了。

# ---- 第二级:对每一块再用递归分割器兜底 ----
level2_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
)

print()
print("第二级(对过大的块按字数再切)的结果:")
final_chunks = []
for c in level1_chunks:
    if len(c.page_content) <= 100:
        final_chunks.append(c)              # 本来就够小,原样保留
    else:
        # split_documents 会**保留原块的 metadata** 并传给切出来的子块
        final_chunks.extend(level2_splitter.split_documents([c]))

for i, c in enumerate(final_chunks):
    print(f"  [{i}] 长度={len(c.page_content):4d}  metadata={c.metadata}")

# 关键受益点:`split_documents` 会自动把父块的 metadata 传下来。
# 所以第二级切出来的每一小块,**依然带着完整的章节路径**。
# 这就是为什么必须先按结构、再按字数 —— 结构信息一旦丢了就找不回来。


# ============================================================================
# 小结
# ============================================================================
#   什么时候用:
#     - Markdown 文档 → MarkdownHeaderTextSplitter
#     - 网页 / HTML 文档 → HTMLHeaderTextSplitter
#     - 技术文档 RAG(这是最典型的场景,能大幅提升检索准确率)
#
#   三个必须记住的点:
#     1. 它们**没有 chunk_size**,块大小完全由原文结构决定
#     2. 实战必须做**两级分割**:先按标记、再按字数,顺序不能反
#     3. metadata 里的层级路径是精华 —— 检索时能告诉你"这是哪一节的内容",
#        也可以拼进上下文增强效果
#
#   和前面分割器的关系:
#     它们是**协作**关系,不是替代关系。
#     RecursiveCharacterTextSplitter 负责"控制大小",标记分割器负责"保住结构"。
