'''
Date: 2026-09-13 20:50:00
Author: parker
FilePath: \llmops-api\study\34-语义文档分割器与按内容分割器的使用\3.递归JSON分割器示例.py
Description: RecursiveJsonSplitter —— 按 JSON 的嵌套结构递归切分,保证每块都是合法 JSON

运行方式(在项目根目录):
    python "study/34-语义文档分割器与按内容分割器的使用/3.递归JSON分割器示例.py"
'''

# ============================================================================
# 第 0 部分:为什么 JSON 需要专门的分割器?
# ============================================================================
#
# 场景很多:RAG 要喂给模型的可能是接口返回的 JSON、配置文件的 JSON、
# 电商商品详情的 JSON。如果你直接拿 RecursiveCharacterTextSplitter 去切,
# 会发生什么?—— 切出来的每一块都不是合法 JSON,json.loads() 直接报错。
#
# 下面先把这个"惨状"演一遍。

import json

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    RecursiveJsonSplitter,
)

# 一份有嵌套、有数组的 JSON(电商商品详情)
product = {
    "商品名称": "机械键盘 K87",
    "价格": 399,
    "品牌": "Keychron",
    "规格参数": {
        "轴体": "红轴",
        "键数": 87,
        "连接方式": ["有线", "蓝牙", "2.4G"],
        "键帽材质": "PBT",
    },
    "用户评价": {
        "平均评分": 4.8,
        "评价总数": 1203,
    },
    "售后政策": "七天无理由退货,一年质保。非人为损坏免费换新。",
}

raw = json.dumps(product, ensure_ascii=False)

print("=" * 70)
print("第 0 部分:普通文本分割器切 JSON 会怎样?")
print("=" * 70)

naive_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
naive_chunks = naive_splitter.split_text(raw)

print(f"原文 {len(raw)} 个字符,切成 {len(naive_chunks)} 块\n")
for i, c in enumerate(naive_chunks):
    try:
        json.loads(c)
        status = "合法 JSON"
    except json.JSONDecodeError as e:
        status = f"!! JSON 解析失败 -> {e.msg}"
    print(f"  [{i}] {status}")
    print(f"       {c!r}")

# 你会看到几乎每一块都解析失败。
# 原因:文本分割器只认字符,它不知道 "{" 和 "}" 是要配对的,
# 也不懂 "a": {"b": ...} 这种嵌套意味着什么。


# ============================================================================
# 第 1 部分:RecursiveJsonSplitter 基本用法
# ============================================================================
# 它认路:按 JSON 的**层级结构**往下走,先试大的键值对,
# 装不下就拆进下一层,直到每个块都塞进 max_chunk_size。
#
# 核心保证:每一块都是一个**合法的 JSON 对象**(Python 里就是 dict)
#
# 主要参数:
#     max_chunk_size : 每块最大字符数(注意:它数的是 json.dumps 之后的字符串长度)
#     min_chunk_size : 块的最小尺寸。比 max 小的那部分"边角料"会尝试合并,
#                      避免出现一堆只有几个字符的碎片块。

print()
print("=" * 70)
print("第 1 部分:split_json —— 切出来还是 dict")
print("=" * 70)

json_splitter = RecursiveJsonSplitter(max_chunk_size=100)

chunks = json_splitter.split_json(product)

print(f"切成 {len(chunks)} 块,每一块都是 dict:\n")
for i, c in enumerate(chunks):
    dumped = json.dumps(c, ensure_ascii=False)
    is_dict = isinstance(c, dict)
    print(f"  [{i}] dict={is_dict}  长度={len(dumped)}")
    print(f"       {dumped}")

# 划重点 1:每一块都能直接 json.loads 回去 —— 因为压根就没"切坏"过。
# 划重点 2:注意 "规格参数" 这个 key 在好几块里**重复出现**了。
#          这是故意的:每个块自带从顶层到自己的路径信息。
#          否则模型看到 {"轴体": "红轴"} 根本不知道说的是什么东西。
#          这是 JSON 分割器比普通分割器聪明的地方。


# ============================================================================
# 第 2 部分:转成 LangChain 的 Document(以及中文转义的坑!)
# ============================================================================

print()
print("=" * 70)
print("第 2 部分:create_documents —— 转成 Document 才能进向量库")
print("=" * 70)

# 注意传进去的是一个**列表**:create_documents([product])
# 它支持一次处理多份 JSON。

# --- 先看默认参数会发生什么 ---
bad_docs = json_splitter.create_documents([product])

print("!! 默认参数(ensure_ascii=True)的输出:")
print(f"   {bad_docs[0].page_content!r}")
print()

# 看到 \\u5546\\u54c1 了吗?这是默认参数把中文**转义成 ASCII** 了。
# 后果很严重:
#   - 向量化时模型看到的是 商品 这种鬼东西,语义完全丢失
#   - 关键词检索(全文匹配)永远匹配不上"商品名称"
# 所以中文场景**必须**显式传 ensure_ascii=False。

good_docs = json_splitter.create_documents([product], ensure_ascii=False)

print("正确参数(ensure_ascii=False)的输出:")
for i, d in enumerate(good_docs):
    print(f"  [{i}] {d.page_content}")
    print(f"       metadata = {d.metadata}")

# metadata 里可以塞来源信息,方便检索时溯源
docs_with_meta = json_splitter.create_documents(
    [product],
    ensure_ascii=False,
    metadatas=[{"source": "product_detail_api", "商品ID": "SKU-10086"}],
)
print()
print("带 metadata 的写法(每个块都会带上这份元数据):")
print(f"   {docs_with_meta[0].metadata}")


# ============================================================================
# 第 3 部分:一个必须知道的坑 —— 单个超长字符串值切不开
# ============================================================================
# RecursiveJsonSplitter 只按 **JSON 结构** 切,不会切开任何一个字符串值本身。
# 所以如果某个字段的值是一篇长文(比如商品详情的富文本描述),
# 那一块就会**超出 max_chunk_size**。
#
# 而且注意:它**既不报错也不打警告**,就这么静默地给你一个超长块。
# 这比第 32 章 CharacterTextSplitter 那个"至少还 print 一行 warning"更难发现 ——
# 你的日志里干干净净,但向量库里已经躺着几块超限的内容了
# (超限的块会被 embedding 模型静默截断,后半截等于没存)。

print()
print("=" * 70)
print("第 3 部分:坑 —— 单个字符串值太长会切不开")
print("=" * 70)

long_desc = "这款键盘采用人体工学设计,键帽为双色注塑工艺。" * 12  # 约 300 字

tricky = {"商品名称": "机械键盘", "详细描述": long_desc}

small_splitter = RecursiveJsonSplitter(max_chunk_size=100)
tricky_chunks = small_splitter.split_json(tricky)

print(f"max_chunk_size=100,但其中一块的实际情况:")
for i, c in enumerate(tricky_chunks):
    dumped = json.dumps(c, ensure_ascii=False)
    flag = "  <<< 超限了!" if len(dumped) > 100 else ""
    print(f"  [{i}] 长度={len(dumped):4d}{flag}")
    print(f"       {dumped[:60]}...")

# 这个行为和第 32 章 CharacterTextSplitter 的坑是同一个道理:
# 切分器只能沿着它认识的"边界"切,遇到一整块没有边界的值就束手无策。
#
# 解决办法:**两级分割**
#   第一级:RecursiveJsonSplitter 按结构切,保住 JSON 的完整性
#   第二级:对切出来仍然过大的块,再用 RecursiveCharacterTextSplitter 按文本切
#
# 这样既保住了大部分块的结构,又不会出现超长块。
# (代价是第二级切出来的块不再是合法 JSON —— 这是必然的取舍,
#  因为一个超长字符串无论怎么切都不可能是完整 JSON。)

print()
print("-" * 70)
print("两级分割的写法示意:")
print("-" * 70)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)

final_chunks = []
for c in tricky_chunks:
    dumped = json.dumps(c, ensure_ascii=False)
    if len(dumped) <= 100:
        final_chunks.append(dumped)          # 没超限,原样保留
    else:
        final_chunks.extend(text_splitter.split_text(dumped))  # 超限,降级用文本分割器

for i, c in enumerate(final_chunks):
    print(f"  [{i}] 长度={len(c):4d}  {c[:55]}...")


# ============================================================================
# 小结
# ============================================================================
#   什么时候用 RecursiveJsonSplitter:
#     - 数据本身是 JSON(接口响应、配置文件、结构化数据)
#     - 你希望检索时能定位到"是哪个字段",而不是一堆没头没尾的文本
#
#   三个必须记住的点:
#     1. create_documents 一定要传 ensure_ascii=False,否则中文变 \uXXXX
#     2. 每个块会重复父级 key,这是特性不是 bug,为了不丢上下文
#     3. 单个超长字符串值它切不开,需要接第二级文本分割器兜底
#
#   什么时候**不要**用:
#     - 你的文档是普通文章/PDF,压根没有 JSON 结构 —— 别硬凑
