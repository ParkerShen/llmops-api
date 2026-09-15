'''
Date: 2026-09-11
Author: parker
FilePath: \llmops-api\study\23-faiss向量数据库的配置与使用\1.faiss向量数据库使用示例.py
Description: FAISS 向量数据库入门 —— 概念 + 原生 API + LangChain 封装 + 持久化 + 索引选型

运行方式(在项目根目录):
    python "study/23-faiss向量数据库的配置与使用/1.faiss向量数据库使用示例.py"

前置依赖:
    pip install faiss-cpu        # 只装 CPU 版就够学习用;有 GPU 才装 faiss-gpu
'''

# ============================================================================
# 第 0 部分:概念(这一段全是注释,先读一遍,不跑代码)
# ============================================================================
#
# ---------- 0.1 为什么需要向量数据库 ----------
# 传统数据库擅长"精确/范围"查询:WHERE name = '张三'、WHERE price > 100。
# 但有一类需求它表达不了 —— "意思相近":
#     用户搜"如何退款",希望命中"怎么申请退货"
# 这两个句子一个共同关键词都没有,SQL 的 LIKE 救不了你。
#
# 解法:把文本喂给"嵌入模型(Embedding Model)",得到一串定长的浮点数,
# 也就是**向量**。语义越接近的文本,向量在多维空间里的位置就越近。
# 于是"找意思相近的文本"就变成了纯数学问题:
#     "给我一个向量,在 100 万个向量里找出距离最近的 k 个"
# 这个操作叫 **最近邻搜索(Nearest Neighbor Search)**。
#
# 麻烦在于:100 万个 1536 维向量,暴力两两算距离是 15 亿次浮点乘加,
# 每次查询都这么干完全不可接受。所以需要用专门的**索引结构**把速度拉上来。
# 这就是向量数据库存在的意义。
#
# ---------- 0.2 FAISS 是什么 ----------
# FAISS = Facebook AI Similarity Search,Meta(原 Facebook)AI 团队开源的
# C++ 库,带 Python 绑定。它只干一件事:
#     一堆向量 + 一个查询向量 → 快速返回最近的 k 个
#
# 它**不管**这些事(要划重点,面试常问):
#   ✗ 文本怎么变成向量      → 那是嵌入模型的活(OpenAI / BGE / 智谱)
#   ✗ 原文和元数据怎么存    → FAISS 只存 id,原文得你自己挂(见第 2.4 节)
#   ✗ 事务、并发写、权限、容灾、分布式 → 那是 Milvus / Qdrant / pgvector 的活
#
# 所以严格讲 FAISS 不是"数据库",它是一个**相似度检索库**。
# 没有服务进程、没有网络协议、没有 SQL,就是 import 进来当个库用。
#
# 那为什么还要学它?
#   1) 它是最纯粹的向量检索实现,API 就几个,学完再学 Milvus/Qdrant
#      基本就是"换个壳子",底层概念一模一样。
#   2) 大量教程和 LangChain 的默认示例都用它,小规模场景(几万条)它完全够用。
#   3) 它的索引类型(Flat/IVF/HNSW/PQ)是所有向量库的公共词汇表。
#
# ---------- 0.3 一个向量库的四步 ----------
#   Step 1 Embed  : 文本 → 向量(维度由模型决定:1536 / 1024 / 768 ...)
#   Step 2 Index  : 把向量灌进索引结构(Flat / IVF / HNSW / PQ)
#   Step 3 Search : 查询向量 → Top-K 最近的向量 → 拿到 id → 用 id 取原文和元数据
#   Step 4 Filter : (可选)先按元数据筛子集,再在子集里做向量检索
#
# ---------- 0.4 距离度量:FAISS 里最重要的一个选择 ----------
#   L2(欧氏距离)     : faiss.METRIC_L2            值越小越相似
#   IP(内积)         : faiss.METRIC_INNER_PRODUCT 值越大越相似
#
#   **工程上最常用的是"先归一化 + 内积"**,因为它等价于余弦相似度:
#       向量除以自己的长度后(长度=1),内积 == 余弦相似度 cos(θ)
#   数学上归一化后 cos = 1 - L2²/2,两者是单调对应的,
#   但用内积能直接读出"相似度 0~1"的语义,比 L2 距离直观得多。
#   faiss 提供了 faiss.normalize_L2(向量数组),原地归一化。
#
# ---------- 0.5 索引类型(先记住结论,第 4 部分会用实验验证) ----------
#   IndexFlatL2 / IndexFlatIP   暴力全量比对,100% 准确,复杂度 O(N)。10 万条以下首选。
#   IndexIVFFlat               k-means 聚成 nlist 个桶,只查 nprobe 个桶。需训练,快,略有精度损失。
#   IndexHNSWFlat              图索引,查询最快,代价是内存翻几倍,建索引慢。
#   IndexIVFPQ                 分桶 + 乘积量化压缩,内存小到 1/4~1/16,精度损失较大。
#
#   经验值:数据量 < 10 万 → 别折腾,直接 IndexFlat。
#           数据量 > 100 万 或 内存吃紧 → 再上 IVF / HNSW / PQ。
#
# ---------- 0.6 一个务必先建立的认知 ----------
#   FAISS 返回的是 **id + 距离**,不是文本。
#   向量库里存向量,原文存在别处,靠 id 关联。LangChain 的封装帮你做了这层关联,
#   但你要知道它背后就是"一个 FAISS 索引 + 一个 id→Document 的字典"。


# ============================================================================
# 导入区
# ============================================================================
import os
import pickle
import sys
import tempfile
import time
import zlib
import re
import numpy as np

# Windows 控制台默认是 GBK,输出 ≤ → ✓ 这类符号会 UnicodeEncodeError,统一改成 utf-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import faiss  # 本次的主角
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS

# 所有产物写到本文件所在目录,避免受"从哪个目录运行"影响
HERE = os.path.dirname(os.path.abspath(__file__))
INDEX_DIR = os.path.join(HERE, "storage")   # 持久化演示用的目录


def section(title: str) -> None:
    """打印分节标题,让输出好读"""
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ============================================================================
# 第 1 部分:一个"不用花 API 额度"的极简嵌入模型
# ============================================================================
# 为什么要自己写这个?
#   1) 你的 OpenAI key 目前额度耗尽(429 insufficient_quota),
#      智谱 key 也报 429(资源包用尽),DeepSeek 干脆没有 embedding 接口。
#      直接上真实嵌入模型,这个文件你今天就跑不起来。
#   2) 更重要:嵌入模型是个**黑盒**,FAISS 完全不关心向量是怎么来的。
#      用一个自己写的、结果可预测的嵌入模型,反而能把
#      "向量 → 索引 → 检索" 这条链路看得最清楚。
#
# 它怎么实现的:词袋(Bag of Words)+ 哈希技巧
#   - 把文本切成 token(英文/数字按整词,中文按单字)
#   - 每个 token 用 CRC32 算出一个稳定的哈希值 → 映射到向量里的某一维
#   - 该维置 1(只记"出现过",不记"出现几次"),最后归一化成长度 1 的向量
#   效果:两段文本共享的 token 越多,向量点积越大,也就是"越相似"。
#
#   为什么用 0/1 而不是词频?
#     因为词频会让"长文本"和"高频字"占便宜。实测这份语料里,
#     用词频时查 "FAISS 是什么" 排第一的是讲 HNSW 的那条(它更短、向量更集中),
#     改成 0/1 之后正确答案才浮上来。这个"文本长度影响打分"的偏差在真实
#     向量库里也存在,所以生产上通常对检索结果做长度归一化或用 Reranker 精排。
#
#   注意它**只认字面重叠,不懂语义**("退款"和"退货"在它眼里毫不相干)。
#   真实嵌入模型的价值就在这里:它把"意思相同但用词不同"也映射到相近位置。
#
# 关键点:只要实现 Embeddings 接口的两个方法,就能像真模型一样塞给 LangChain。


class BagOfWordsEmbeddings(Embeddings):
    """教学用极简嵌入模型:哈希词袋 + L2 归一化(离线可用,不产生任何费用)"""

    def __init__(self, dim: int = 256):
        self.dim = dim

    @staticmethod
    def _tokenize(text: str) -> list:
        """英文数字按整词切,中文按单字切(不引第三方分词库,够演示了)"""
        return re.findall(r"[a-zA-Z0-9]+|[一-鿿]", text.lower())

    def _embed_one(self, text: str) -> list:
        vec = np.zeros(self.dim, dtype="float32")
        for token in self._tokenize(text):
            # 用 crc32 而不是内置 hash():内置 hash 对字符串每次进程启动都会变(PYTHONHASHSEED),
            # 那样存进磁盘的索引下次加载就对不上了,这是个真实的坑。
            idx = zlib.crc32(token.encode("utf-8")) % self.dim
            vec[idx] = 1.0     # 用 0/1 记"出现过",不用 +1 记词频(原因见上面注释)
        # 归一化成长度 1,这样内积直接等于余弦相似度(见 0.4 节)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    def embed_documents(self, texts: list) -> list:
        """入库用:批量把文档变成向量。接口要求返回 List[List[float]]"""
        return [self._embed_one(t) for t in texts]

    def embed_query(self, text: str) -> list:
        """查询用:把用户问题变成向量。接口要求返回 List[float]"""
        return self._embed_one(text)


# 一份迷你知识库,后面所有实验都用它
CORPUS = [
    ("FAISS 是 Meta 开源的一个向量相似度检索库,底层用 C++ 实现,Python 调用速度也很快。",
     {"source": "faiss-intro.md", "category": "基础"}),
    ("向量数据库的核心思路是把文本变成向量之后,用最近邻搜索代替关键词匹配。",
     {"source": "vector-db.md", "category": "基础"}),
    ("RAG 的流程是先把文档切块做成向量存进向量库,用户提问时先检索相关片段,再把片段塞进 prompt 交给大模型回答。",
     {"source": "rag.md", "category": "进阶"}),
    ("IndexFlatL2 是最简单的索引,查询时和所有向量逐个算距离,结果绝对准确,但数据量大时速度会明显变慢。",
     {"source": "faiss-index.md", "category": "索引"}),
    ("IndexIVFFlat 先用 k-means 把向量聚成 nlist 个桶,查询时只搜 nprobe 个桶,用一点点精度换大量速度。",
     {"source": "faiss-index.md", "category": "索引"}),
    ("HNSW 是一种基于图的近似最近邻算法,查询速度最快,代价是内存占用会翻好几倍。",
     {"source": "faiss-index.md", "category": "索引"}),
    ("文本切分决定了每个块的大小和重叠区域,切得太大检索不精准,切得太小会丢上下文。",
     {"source": "rag.md", "category": "进阶"}),
    ("向量维度由嵌入模型决定,比如 OpenAI 的 text-embedding-3-small 是 1536 维,BGE-M3 是 1024 维。",
     {"source": "embedding.md", "category": "基础"}),
    ("余弦相似度衡量两个向量方向是否一致,取值范围是 -1 到 1,越接近 1 表示语义越接近。",
     {"source": "embedding.md", "category": "基础"}),
    ("元数据过滤可以让检索只在某个子集里进行,比如只在分类为索引的文档里查找。",
     {"source": "filter.md", "category": "进阶"}),
]

TEXTS = [t for t, _ in CORPUS]
METAS = [m for _, m in CORPUS]

embedding = BagOfWordsEmbeddings(dim=256)
VECTORS = np.asarray(embedding.embed_documents(TEXTS), dtype="float32")
DIM = VECTORS.shape[1]


# ============================================================================
# 第 2 部分:不依赖 LangChain,直接用 faiss 原生 API
# ============================================================================
# 先摸原生 API。知道底层长什么样,LangChain 那层封装就不再是魔法。

section("第 2 部分 原生 faiss API")

print(f"向量形状: {VECTORS.shape}  (共 {VECTORS.shape[0]} 条,每条 {DIM} 维)")
print(f"第一条向量的前 8 维: {VECTORS[0][:8].round(3).tolist()}")

# ---------- 2.1 建索引 + 灌数据 ----------
# IndexFlatL2 就是"暴力全量比",最简单也最准,是所有实验的基准(baseline)
index = faiss.IndexFlatL2(DIM)      # 构造时就确定维度,以后不能改
index.add(VECTORS)                  # 注意:必须是 float32 且内存连续的 numpy 数组
print(f"\n[2.1] 建好 IndexFlatL2,当前向量总数 ntotal = {index.ntotal}")

# ---------- 2.2 查询 ----------
# search 的参数:查询向量(2维数组,可以一次传多个查询)、k
query = "FAISS 是什么"
q_vec = np.asarray([embedding.embed_query(query)], dtype="float32")
k = 3
distances, ids = index.search(q_vec, k)

print(f"\n[2.2] 查询: {query!r}  (k={k})")
for rank, (dist, idx) in enumerate(zip(distances[0], ids[0]), start=1):
    # 关键认知:FAISS 只给你 id 和距离,**不给你文本**
    # 你得自己维护 id → 原文 的映射,这里就是 CORPUS 的下标
    text = TEXTS[idx] if idx != -1 else "(无结果)"
    print(f"  {rank}. L2²={dist:6.4f}  id={idx:2d}  {text[:38]}...")
    print(f"     元数据={METAS[idx]}")

# ---------- 2.3 三个必须知道的细节 ----------
# (a) ⚠️ IndexFlatL2 返回的是**平方后的** L2 距离,不是开方后的欧氏距离!
#     这是 FAISS 最经典的坑。自己验一下:两个点 (0,0) 和 (3,4),
#     真实欧氏距离是 5.0,而 FAISS 会返回 25.0。
#     影响很实际:排序不受影响(单调),但如果你想拿距离做阈值判断
#     (比如"距离 > 1.2 就认为不相关"),不先开方就会判错。
_a = np.asarray([[0.0, 0.0]], dtype="float32")
_b = np.asarray([[3.0, 4.0]], dtype="float32")
_probe = faiss.IndexFlatL2(2)
_probe.add(_a)
_d = _probe.search(_b, 1)[0][0][0]
print(f"\n[2.3a] (0,0) 到 (3,4):真实欧氏距离=5.0, FAISS 返回={_d:.1f} "
      f"→ 印证返回的是平方值,开方后={np.sqrt(_d):.1f}")

# (b) labels 里出现 -1 是什么意思?
#     当 k 大于实际向量数时,FAISS 用 -1 填充,表示"没有这一条"。
#     所以取结果前一定要判断 idx != -1,否则会拿 -1 去索引列表,
#     在 Python 里恰好取到**最后一条**(负索引),这是个很隐蔽的 bug。
d2, i2 = index.search(q_vec, 99)
print(f"[2.3b] k=99 但库里只有 {index.ntotal} 条:前 3 个 id={i2[0][:3].tolist()}, "
      f"id[3:8]={i2[0][3:8].tolist()} 全是 -1(填充位)")

# (c) 距离的绝对大小没有意义,只有**相对排序**有意义。
#     距离 0 表示完全相同,值越大越不像。跨不同库/不同模型的距离不可比。
print(f"[2.3c] 同一条向量查自己,平方距离 = {index.search(VECTORS[:1], 1)[0][0][0]:.6f}(约等于 0)")

# ---------- 2.4 换成内积 + 归一化 = 余弦相似度 ----------
# 这是生产里最常用的配置。归一化后内积直接落在 [-1, 1],读起来就是"相似度百分比"。
normed = VECTORS.copy()
faiss.normalize_L2(normed)                     # 原地归一化,长度变成 1
index_ip = faiss.IndexFlatIP(DIM)              # IP = Inner Product
index_ip.add(normed)

q_norm = np.asarray([embedding.embed_query(query)], dtype="float32")
faiss.normalize_L2(q_norm)                     # 查询向量也要归一化,别忘

# 内积索引默认返回**从大到小**排序(越大越相似),和 L2 相反,别搞混
scores, ids_ip = index_ip.search(q_norm, k)
print(f"\n[2.4] IndexFlatIP + 归一化(余弦相似度,越大越像):")
for rank, (s, idx) in enumerate(zip(scores[0], ids_ip[0]), start=1):
    print(f"  {rank}. 余弦相似度={s:6.4f}  id={idx:2d}  {TEXTS[idx][:38]}...")

# ---------- 2.5 增删改 ----------
# ⚠️ 重要:普通 FAISS 索引的 id 就是 0,1,2... 顺序整数,由 add 的顺序决定。
#   如果你删掉中间一条,后面所有 id 都会往前挪一位 —— 这就是所谓的"id 不稳定",
#   生产上不能接受。解决办法是套一层 IndexIDMap,自己指定外部 id。
#   (LangChain 的 FAISS 封装走的是另一条路:索引里还是 0..n-1,它自己额外维护
#    一个 index_to_docstore_id 字典做映射。第 3.9 节会看到这个字段。)
index.add(np.asarray([embedding.embed_query("临时插入的一条测试文本")], dtype="float32"))
print(f"\n[2.5] add 之后 ntotal = {index.ntotal}")
index.remove_ids(faiss.IDSelectorBatch(np.asarray([0], dtype="int64")))
print(f"      remove_ids([0]) 之后 ntotal = {index.ntotal}  ← 删掉的是 id=0,后面全体前移")

# 用 IndexIDMap 自己管 id,才是可用的姿势
id_map = faiss.IndexIDMap2(faiss.IndexFlatL2(DIM))      # IDMap2 支持 reconstruct,IDMap 不支持
my_ids = np.asarray([1001, 1002, 1003], dtype="int64")  # 比如用数据库主键当 id
id_map.add_with_ids(VECTORS[:3], my_ids)
_, got = id_map.search(q_vec, 2)
print(f"      IndexIDMap2 检索拿回的是自定义 id: {got[0].tolist()}(而不是 0/1/2)")
id_map.remove_ids(np.asarray([1002], dtype="int64"))
print(f"      删掉 1002 后剩下: ntotal={id_map.ntotal}, "
      f"reconstruct(1001) 能取回原向量长度={len(id_map.reconstruct(1001))}")

# ---------- 2.6 持久化:原生 faiss 只存索引 ----------
# faiss.write_index 存的是**纯索引**(一堆浮点数和结构信息),
# 不含原文、不含元数据。所以完整的持久化 = 索引文件 + 你自己的 id→原文 的存储。
# 这是 FAISS 最容易踩的坑:有人只存了索引,重启后检索出一堆 id 却不知道是什么。
os.makedirs(INDEX_DIR, exist_ok=True)
raw_index_path = os.path.join(INDEX_DIR, "raw_flat.index")
try:
    # ⚠️ 实测坑:faiss.write_index 的路径参数在 C++ 层是 const char*,
    #    **Windows 上遇到中文路径会直接报 "could not open ... No such file or directory"**,
    #    而我们的目录名 "23-faiss向量数据库的配置与使用" 正好是中文。
    #    解决办法见下面的 serialize_index 方案(官方推荐,跨平台且不挑路径)。
    faiss.write_index(index, raw_index_path)
    print(f"\n[2.6] faiss.write_index → raw_flat.index")
except RuntimeError as e:
    print(f"\n[2.6] faiss.write_index 失败(中文路径的 C++ 层限制): {str(e)[:60]}...")
    print(f"      改用 faiss.serialize_index 方案 ↓")

# ✅ 推荐姿势:serialize_index 把索引序列化成 numpy 数组,由 Python 负责写文件,
#    这样路径里有什么字符都无所谓。read 的时候用 deserialize_index 还原。
blob = faiss.serialize_index(index)
serialize_path = os.path.join(INDEX_DIR, "raw_flat_serialized.bin")
with open(serialize_path, "wb") as f:
    f.write(blob.tobytes())
with open(serialize_path, "rb") as f:
    loaded = faiss.deserialize_index(np.frombuffer(f.read(), dtype="uint8"))

print(f"      serialize_index → {os.path.getsize(serialize_path) / 1024:.2f} KB, "
      f"deserialize 回来 ntotal = {loaded.ntotal}")
print(f"      ⚠️ 注意:索引在,但原文和元数据不在!必须自己另存一份 id→文档 的映射,")
print(f"         否则重启后你能拿到 id 却不知道它对应哪段文本。")

# 用 index_factory 按字符串建索引,写配置时很方便
# "Flat"=暴力, "IVF100"=100 个桶, "PQ8"=量化成 8 字节, "HNSW32"=图索引, "SQ8"=标量量化
factory_index = faiss.index_factory(DIM, "Flat")
factory_index.add(VECTORS)
print(f"      faiss.index_factory(DIM, 'Flat') 等价于 IndexFlatL2, ntotal={factory_index.ntotal}")


# ============================================================================
# 第 3 部分:用 LangChain 的 FAISS 封装(实际写业务用这个)
# ============================================================================
# 封装帮你解决了上面那些手工活:
#   - 自动做 embed_documents / embed_query
#   - 自动维护 id → Document 的映射(反过来也有)
#   - 检索直接返回 Document 对象(带 page_content 和 metadata)
#   - 直接当 Runnable 用,可以塞进 LCEL 链(这是第四部分 RAG 的关键)
# 但它没有解决"原文和元数据要自己持久化"的问题 —— 只是用 save_local 帮你一起存了。

section("第 3 部分 LangChain FAISS 封装")

documents = [Document(page_content=t, metadata=m) for t, m in CORPUS]

# ---------- 3.1 from_documents:一步建库 ----------
# 内部做的事:embed_documents(所有文本) → new IndexFlatL2 → add → 建立 id 映射
# 默认用 L2。想要余弦相似度就传 distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT
# (注意:那样需要你自己保证向量已归一化,封装不会替你归一化)
vector_store = FAISS.from_documents(documents, embedding)
print(f"[3.1] from_documents 建库完成,共 {vector_store.index.ntotal} 条")

# 偷看它的内部结构 —— 理解这层,以后换任何向量库都不慌
print(f"      内部字段: {[f for f in ['index', 'docstore', 'index_to_docstore_id'] if hasattr(vector_store, f)]}")
print(f"      index_to_docstore_id 示例: {dict(list(vector_store.index_to_docstore_id.items())[:3])}")

# ---------- 3.2 基础检索 ----------
print(f"\n[3.2] similarity_search({query!r}, k=3)")
for i, doc in enumerate(vector_store.similarity_search(query, k=3), start=1):
    print(f"  {i}. [{doc.metadata['category']}] {doc.page_content[:38]}...")

# ---------- 3.3 带分数检索:注意分数的方向 ----------
# similarity_search_with_score 返回的是**原始距离**,默认是 L2,所以
#   **分数越小越相似**,和"相似度"的直觉相反,这是最常被误读的返回值。
print(f"\n[3.3] similarity_search_with_score(默认 L2,越小越像;值是平方距离)")
for doc, score in vector_store.similarity_search_with_score(query, k=3):
    print(f"  L2²={score:6.4f}  {doc.page_content[:38]}...")

# ⚠️⚠️ 接下来这个 API 有坑,而且是"看起来很合理所以更容易中招"的坑。⚠️⚠️
# similarity_search_with_relevance_scores 号称把分数归一化到 0~1,
# 但它内部用的是硬编码公式,并且**默认假设传进来的分数是距离(越小越好)**。
# 结果就是:文档顺序是按 L2 距离排好的,分数却可能是负数,甚至方向是反的。
# 我们把它和真实距离并排打出来,亲眼看一下:
print(f"      用 relevance_scores 归一化后(注意看分数是不是符合直觉):")
for doc, score in vector_store.similarity_search_with_relevance_scores(query, k=3):
    print(f"  相似度={score:6.4f}  {doc.page_content[:38]}...")
print(f"""
      为什么?看它内部用的公式(源码在 langchain_core/vectorstores/base.py):
          _euclidean_relevance_score_fn(distance)  = 1 - distance / sqrt(2)
          _max_inner_product_relevance_score_fn(distance) = 1 - distance
      两个函数的参数都叫 distance,潜台词是"越小越好"。但:
        · 默认的 EUCLIDEAN 策略下,FAISS 给的是**平方后的** L2,
          而公式里按**未平方**的距离来除 sqrt(2),
          所以距离一大就掉到 0 以下 —— 这就是上面出现负数的原因。
        · 如果换成 MAX_INNER_PRODUCT 策略,原始值是**相似度**(越大越好),
          公式却仍然算 1 - x,于是把排序**整个反过来**了:
          最相关的文档拿到最低分,最不相关的拿到最高分。
      结论(可以直接记下来):
        ❌ 别信 similarity_search_with_relevance_scores 的默认行为;
        ✅ 用 similarity_search_with_score 拿原始分数,**自己按策略判断方向**;
        ✅ 或者给 FAISS 构造函数传 relevance_score_fn=你自己的函数,
           把归一化规则攥在自己手里。
      """)

# 想要可控的 0~1 相似度,正确的做法是:先归一化向量 + 用内积,分数就是余弦相似度。
# 这里手工算一遍,你能看到"相似度"本该长什么样:
_q = np.asarray([embedding.embed_query(query)], dtype="float32")
faiss.normalize_L2(_q)
print(f"      手工算余弦相似度(越大越像),和前 3 名一一对应:")
for doc in vector_store.similarity_search(query, k=3):
    _d = np.asarray([embedding.embed_query(doc.page_content)], dtype="float32")
    faiss.normalize_L2(_d)
    print(f"  余弦={float(_d @ _q.T):6.4f}  {doc.page_content[:38]}...")

# ---------- 3.4 元数据过滤 ----------
# 只在本分类为"索引"的文档里检索。这在实际业务里极其常用
# (比如多租户场景只查当前用户的知识库)。
# 实现上它拿的是全量结果再过滤,所以数据量大时记得调大 fetch_k,
# 否则可能过滤完一条都不剩。
print(f"\n[3.4] 只在 category='索引' 的文档里检索: {query!r}")
for doc, score in vector_store.similarity_search_with_score(
    query, k=3, filter={"category": "索引"}, fetch_k=20
):
    print(f"  L2²={score:6.4f}  [{doc.metadata['category']}] {doc.page_content[:38]}...")

# ---------- 3.5 MMR:解决"检索结果彼此太像"的问题 ----------
# 普通 Top-K 检索有个毛病:如果知识库里同一段话被切成 3 块,
# 它们会霸占全部 3 个名额,答案的多样性就没了。
# MMR(Maximal Marginal Relevance)在"和问题相关"与"彼此不重复"之间做折中:
#   lambda_mult=1.0 → 只看相关性(退化成普通检索)
#   lambda_mult=0.0 → 只看多样性
#   lambda_mult=0.5 → 默认,平衡
print(f"\n[3.5] MMR 检索(lambda_mult=0.5):")
for i, doc in enumerate(vector_store.max_marginal_relevance_search(
    query, k=3, fetch_k=10, lambda_mult=0.5
), start=1):
    print(f"  {i}. {doc.page_content[:38]}...")

# ---------- 3.6 增量添加与删除 ----------
# 真实场景里数据是不断新增的,不能每次都全量重建索引
new_docs = [
    Document(page_content="FAISS 支持增量添加文档,不用重建整个索引。",
             metadata={"source": "faiss-intro.md", "category": "基础"}),
]
new_ids = vector_store.add_documents(new_docs)
print(f"\n[3.6] add_documents 完成,新 id={new_ids}, ntotal={vector_store.index.ntotal}")

# 删除要用 add 时返回的 id,或者用 metadata 过滤后再删
deleted = vector_store.delete(new_ids)
print(f"      delete 返回 {deleted}, 删完 ntotal={vector_store.index.ntotal}")

# ---------- 3.7 持久化 ----------
# 存两个东西,缺一不可:
#   index.faiss —— 向量索引本身(内部就是 faiss.write_index 那个东西)
#   index.pkl   —— docstore(原文和元数据)+ index_to_docstore_id 映射
# 只拷 .faiss 过去,检索出来的 Document 会是**空的**(正文和元数据都丢了)。
#
# ⚠️ 又一个中文路径坑:save_local 内部调的是 faiss.write_index,和第 2.6 节同一个
#    C++ 层限制,所以只要路径里有中文就会炸。我们的 study 目录名就是中文,
#    所以下面演示两套写法:能改路径就用 save_local,不能改就手动序列化。

# 写法一:save_local / load_local(路径必须是纯 ASCII)
ascii_dir = os.path.join(tempfile.gettempdir(), "faiss_study_storage")
try:
    vector_store.save_local(ascii_dir, index_name="index")
    print(f"\n[3.7] save_local → {ascii_dir}")
    for f in sorted(os.listdir(ascii_dir)):
        print(f"      {f:14s} {os.path.getsize(os.path.join(ascii_dir, f)) / 1024:8.2f} KB")

    # 加载时必须传**同一个** embedding 实例/配置:
    # 查询向量和入库向量必须由同一个模型产生,否则维度可能对不上;
    # 就算维度凑巧一样,向量空间也不是同一个,检索结果会完全是乱的。
    # allow_dangerous_deserialization=True 是必须的,因为 .pkl 用的是 pickle,
    # 官方靠这个参数让你确认"我知道反序列化会执行代码,这个文件是可信的"。
    # 所以:**永远不要加载别人给你的 .pkl 文件**。
    loaded_vs = FAISS.load_local(
        ascii_dir, embedding, index_name="index",
        allow_dangerous_deserialization=True,
    )
    print(f"      load_local 回来 ntotal={loaded_vs.index.ntotal}")
    top = loaded_vs.similarity_search(query, k=1)[0]
    print(f"      检索验证: {top.page_content[:34]}... / 元数据保留={top.metadata}")
except RuntimeError as e:
    print(f"\n[3.7] save_local 失败: {str(e)[:70]}...")

# 写法二:手动序列化 —— 路径爱叫什么叫什么,顺便看清 save_local 到底存了什么
# 这其实就是 save_local 的内部实现,自己写一遍,以后遇到问题能自己拆
manual_dir = os.path.join(INDEX_DIR, "lc_faiss_manual")   # 中文路径,没问题
os.makedirs(manual_dir, exist_ok=True)
with open(os.path.join(manual_dir, "index.faiss"), "wb") as f:
    # serialize_index 得到 numpy 数组,交给 Python 写文件,绕开 C++ 的路径限制
    f.write(faiss.serialize_index(vector_store.index).tobytes())
with open(os.path.join(manual_dir, "index.pkl"), "wb") as f:
    # 就这两个东西:文档仓库 + "索引第 i 行对应哪个文档 id" 的映射
    pickle.dump((vector_store.docstore, vector_store.index_to_docstore_id), f)

# 读回来
with open(os.path.join(manual_dir, "index.faiss"), "rb") as f:
    _index = faiss.deserialize_index(np.frombuffer(f.read(), dtype="uint8"))
with open(os.path.join(manual_dir, "index.pkl"), "rb") as f:
    _docstore, _i2d = pickle.load(f)
# FAISS 构造函数签名: (embedding_function, index, docstore, index_to_docstore_id, ...)
restored_vs = FAISS(embedding, _index, _docstore, _i2d)

print(f"\n      写法二 手动序列化 → {manual_dir}(中文路径 OK)")
print(f"      restore 回来 ntotal={restored_vs.index.ntotal}")
top = restored_vs.similarity_search(query, k=1)[0]
print(f"      检索验证: {top.page_content[:34]}... / 元数据保留={top.metadata}")
print(f"      ⚠️ 手动构造函数**不会**帮你恢复 distance_strategy / normalize_L2"
      f" / relevance_score_fn 这些构造参数,得自己再传一遍,")
print(f"         否则会静默退化成默认的 EUCLIDEAN_DISTANCE(这是个隐蔽的不一致)。")


# ============================================================================
# 第 4 部分:索引类型对比 —— 用实验来验证 0.5 节的结论
# ============================================================================
# 用一个假数据集横向对比:准确率(召回率)、建索引耗时、查询耗时、内存占用。
# 数字比结论更有说服力,建议自己改 N 跑几遍看趋势。

section("第 4 部分 索引类型对比实验(数据是随机生成的,只为看趋势)")

rng = np.random.default_rng(42)
N, D = 20000, 64            # 2 万条,64 维。真实场景换成 100 万 / 768 维,趋势一样
DATA = np.ascontiguousarray(rng.normal(size=(N, D)).astype("float32"))
faiss.normalize_L2(DATA)    # 归一化,好让 IP 的内积等于余弦相似度
QUERIES = np.ascontiguousarray(DATA[:50] + rng.normal(scale=0.5, size=(50, D)).astype("float32"))
faiss.normalize_L2(QUERIES)
K, GT_K = 10, 100           # 取前 10;算召回率用"暴力法前 100"当标准答案


def _ivf(nlist: int, nprobe: int):
    """IVF 索引:先聚类分桶。必须 train 之后才能 add"""
    quantizer = faiss.IndexFlatIP(D)          # 桶心(聚类中心)本身用暴力索引来算
    idx = faiss.IndexIVFFlat(quantizer, D, nlist, faiss.METRIC_INNER_PRODUCT)
    idx.nprobe = nprobe                       # 查询时扫几个桶,越大越准越慢
    return idx


def _hnsw(m: int):
    """HNSW 图索引:不需要 train,但建索引慢、内存大"""
    return faiss.IndexHNSWFlat(D, m, faiss.METRIC_INNER_PRODUCT)


def _ivfpq(nlist: int, m: int):
    """IVF + 乘积量化:内存最省,精度损失最大。m 是把向量切成几段"""
    return faiss.IndexIVFPQ(
        faiss.IndexFlatIP(D), D, nlist, m, 8, faiss.METRIC_INNER_PRODUCT
    )


def recall_at_k(found_ids: np.ndarray, truth_ids: np.ndarray) -> float:
    """召回率@K = 检索到的前 K 个里,有多少比例真的落在暴力法的 Top-K 里"""
    hits = 0
    for found, truth in zip(found_ids, truth_ids):
        hits += len(set(found.tolist()) & set(truth.tolist()))
    return hits / (len(found_ids) * K)


def bench(name: str, build, trained: bool = False) -> None:
    """统一跑:建索引 / 查询 / 算召回率,然后打印一行结果"""
    t0 = time.perf_counter()
    idx = build()
    if trained:                                   # IVF / PQ 类索引必须先 train 才能 add
        idx.train(DATA)
    idx.add(DATA)
    build_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    for q in QUERIES:
        idx.search(np.ascontiguousarray(q[None, :]), K)
    search_ms = (time.perf_counter() - t0) * 1000

    _, found_all = idx.search(QUERIES, K)
    recall = recall_at_k(found_all, TRUTH)

    # 内存:只算向量本身占的字节数(不含 HNSW 的图结构等额外开销)
    vec_bytes = N * D * 4
    if "PQ" in name:
        vec_bytes = N * 8                         # 乘积量化能压到约 1 字节/维以下
    print(f"  {name:22s} 建索引={build_ms:8.1f}ms  查询50条={search_ms:7.1f}ms  "
          f"召回率@{K}={recall:6.1%}  向量内存≈{vec_bytes / 1024 / 1024:5.1f}MB")


# 先跑暴力索引的 Top-100,当作"标准答案"(ground truth)
ground = faiss.IndexFlatIP(D)
ground.add(DATA)
_, TRUTH = ground.search(QUERIES, GT_K)

print(f"\n基准(IndexFlatIP,暴力但 100% 准):")
bench("IndexFlatIP", lambda: faiss.IndexFlatIP(D))

print(f"\n近似索引(用一点精度换速度/内存):")
bench("IndexIVFFlat nprobe=1", lambda: _ivf(nlist=100, nprobe=1), trained=True)
bench("IndexIVFFlat nprobe=10", lambda: _ivf(nlist=100, nprobe=10), trained=True)
bench("IndexHNSWFlat M=32", lambda: _hnsw(m=32))
bench("IndexIVFPQ", lambda: _ivfpq(nlist=100, m=8), trained=True)

print("""
  读表要点:
    1) IndexFlat 召回率必然 100%(它就是暴力法),它就是标准答案本身。
       查询耗时确实是全场最慢,但**注意:2 万条的时候差距还很小**,
       因为这点数据量暴力法也就几毫秒。把上面的 N 改成 100000 再跑一遍,
       Flat 的耗时才会明显地甩开其他索引 —— 这也是"小数据别折腾"的原因。
    2) IVF 的 nprobe 是"精度/速度"的旋钮:nprobe 越大越准也越慢。
       100 个桶只搜 1 个,召回率掉到 6 成;搜 10 个就追平暴力法了。
       这个旋钮能**在查询时动态调**,不用重建索引,应急时很好用。
    3) HNSW 召回率是满的,建索引最慢,内存也涨(每个点要存 M 条边)。
       ⚠️ 注意看数字:HNSW 的查询耗时这次甚至比 Flat 还慢(3.3ms vs 2.5ms)!
       这不是说 HNSW 不行,而是**它的优势在 2 万条这个量级根本体现不出来**
       —— 图索引的收益要数据量大到暴力法明显吃力时才显现。
       书上说"HNSW 查询最快"是**有前提的**,别脱离数据量背结论。
       它的定位:读多写少 + 内存管够,建一次查一万次才划算。
    4) IVFPQ 内存省到 1/20 以下,但召回率只剩三成。它适合"数据量极大、
       精度可以大幅妥协"的场景(比如先粗筛,再用别的手段精排)。
    5) ⚠️ 别照抄上面的绝对数字。耗时和 N、D、机器强相关,
       要抓着这条规律:Flat=准而慢,IVF/HNSW=快而略不准,PQ=最省内存而最不准。
    6) 实操建议:先用 IndexFlat 把流程跑通,等信息量和延迟要求上来了
       再换索引。**过早优化索引是浪费时间** —— 先把切分和嵌入模型搞好,
       收益比换索引大得多。
""")


# ============================================================================
# 第 5 部分:换成真实嵌入模型(上线时就改这一处)
# ============================================================================
# 上面为了离线可跑用了自己写的词袋模型,**它只认字面重叠,不懂语义**。
# 真实场景必须换成真正的嵌入模型。用法完全一样 ——
# 因为所有模型都实现了同一个 Embeddings 接口,这就是接口的价值。
#
# 直接替换这一行即可:
#
#     embedding = BagOfWordsEmbeddings(dim=256)     # 现在
#     embedding = OpenAIEmbeddings(model="text-embedding-3-small")   # 换成这个
#
# 三种可选方案(按当前环境实际情况给了备注):
#
#  方案 A:OpenAI(需要 OPENAI_API_KEY + 余额)
#      from langchain_openai import OpenAIEmbeddings
#      embedding = OpenAIEmbeddings(model="text-embedding-3-small")   # 1536 维,便宜,中文一般
#      embedding = OpenAIEmbeddings(model="text-embedding-3-large")   # 3072 维,效果更好更贵
#      ⚠️ 实测你 .env 里这个 key 目前返回 429 insufficient_quota(额度耗尽),
#         去 platform.openai.com 充值后才能用。代码本身是对的。
#
#  方案 B:智谱 embedding-3(需要 ZHIPUAI_API_KEY)
#      from langchain_community.embeddings import ZhipuAIEmbeddings
#      embedding = ZhipuAIEmbeddings(model="embedding-3")             # 中文友好
#      ⚠️ 实测你 .env 里这个 key 目前返回 429(资源包用尽),
#         智谱开放平台领取/购买资源包后可用。
#
#  方案 C:本地模型(不花钱、不联网,推荐长期用这个)
#      pip install langchain-huggingface sentence-transformers
#      from langchain_huggingface import HuggingFaceEmbeddings
#      embedding = HuggingFaceEmbeddings(
#          model_name="BAAI/bge-small-zh-v1.5",       # 中文小模型,约 100MB,首次运行会下载
#          encode_kwargs={"normalize_embeddings": True},  # 归一化,配合内积用
#      )
#      优点:免费、离线、数据不出本地。缺点:首次要下载模型、吃一点 CPU/内存。
#      中文场景还可以看 BAAI/bge-m3、BAAI/bge-large-zh-v1.5。
#
#  选型建议:
#      - 中文为主 + 想省钱 + 能接受首次下载 → bge-small-zh-v1.5 起步
#      - 效果优先 + 不在乎成本             → 智谱 embedding-3 或 OpenAI 3-large
#      - 一定要记住:**换嵌入模型 = 必须重建整个索引**(新旧向量不在同一个空间)。
#        这是向量库升级最痛的一点,生产上要提前规划。

section("第 5 部分 换真实嵌入模型(说明见上方注释)")
print("当前使用:BagOfWordsEmbeddings(离线教学版,只认字面重叠)")
print("切换方法:把 embedding 这一行换掉,再重建索引即可,其余代码一行不用改")


# ============================================================================
# 第 6 部分:接进 llmops-api —— 最小 RAG 雏形
# ============================================================================
# 这一步把你前面学的东西全串起来了:
#   FAISS 检索(本次) + Prompt(第 1 课) + 模型(第 2 课) + LCEL 链(第 4 课)
#
# 思路只有一句话:
#     先用向量库把"和问题最相关的几段资料"捞出来,拼进 prompt 的上下文,
#     再让大模型基于这些资料回答 —— 这就是 RAG(检索增强生成)。
#
# 为什么需要它:
#   大模型不知道你的私有数据,而且直接问容易胡说(幻觉)。
#   把检索到的原文塞进 prompt,等于"开卷考试",答案有据可查。
#
# ⚠️ 下面这段默认不执行(会真的调一次 DeepSeek 花 token)。
#    想跑就把 RUN_LLM_DEMO 改成 True。

RUN_LLM_DEMO = False

if RUN_LLM_DEMO:
    from dotenv import load_dotenv
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_deepseek import ChatDeepSeek

    load_dotenv()   # 读根目录的 .env,拿 DEEPSEEK_API_KEY

    retriever = vector_store.as_retriever(
        # search_type="mmr" 能减少重复片段;默认 "similarity" 就是纯 Top-K
        search_type="similarity",
        search_kwargs={"k": 3},
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "你是一个严谨的知识库助手。只能依据下面的参考资料回答问题,"
         "资料里没有的内容就直接说不知道,不要编造。\n\n参考资料:\n{context}"),
        ("human", "{query}"),
    ])


    def format_docs(docs) -> str:
        """把检索到的 Document 列表拼成一段纯文本喂给 prompt"""
        return "\n\n".join(f"[{i}] {d.page_content}" for i, d in enumerate(docs, 1))


    # LCEL 链:| 左边构造出 {"context": ..., "query": ...},右边一路走到字符串
    # RunnablePassthrough.assign 的妙处:在保留原有 query 字段的同时,新增 context 字段
    # (这就是第 15 课 bind / assign 的实战用法)
    rag_chain = (
        RunnablePassthrough.assign(context=lambda x: format_docs(retriever.invoke(x["query"])))
        | prompt
        | ChatDeepSeek(model="deepseek-chat")
        | StrOutputParser()
    )

    question = "怎么让检索速度更快?"
    docs = retriever.invoke(question)
    print(f"\n[6] 问题: {question}")
    print(f"    检索到 {len(docs)} 段资料,第一段: {docs[0].page_content[:40]}...")
    print(f"    DeepSeek 基于资料的回答:\n    {rag_chain.invoke({'query': question})}")
else:
    print("\n[6] RAG 链已写好但未执行(避免消耗 DeepSeek 额度)。")
    print("    想看效果:把本文件里的 RUN_LLM_DEMO 改成 True 再运行。")


# ============================================================================
# 第 7 部分:接下来要学的知识点
# ============================================================================
# 详细版见同目录的《2.学习路线与知识点清单.md》。这里只列最关键的分界线:
#
#   已经掌握(本文档覆盖):
#     ✓ 向量的概念、维度、距离度量(L2 / 内积 / 余弦)
#     ✓ FAISS 原生 API:建索引、add、search、增删、write/read_index
#     ✓ LangChain 封装:from_documents、similarity_search*、filter、MMR、
#       as_retriever、save_local/load_local
#     ✓ 索引选型:Flat / IVF / HNSW / PQ 的取舍,以及 nprobe、M 这些旋钮
#
#   下一步该学(按优先级):
#     1. 文本切分(Text Splitters)   ← 直接决定检索质量,比换向量库重要得多
#     2. 真实嵌入模型的选型与评测     ← 中文场景重点看 BGE 系列
#     3. RAG 完整链路与调优          ← 检索不到就搜不到答案,瓶颈通常在这
#     4. 混合检索 + Rerank           ← 向量检索 + BM25 关键词,再用 Reranker 精排
#     5. Milvus / Qdrant / pgvector  ← 生产级向量库,概念和 FAISS 完全相通
#     6. 评估                          ← 召回率/命中率怎么量,不然调优全靠玄学
#
#   一句话总结学习顺序:
#     先跑通 FAISS(本文档) → 再学切分和嵌入 → 再做完整 RAG → 最后换生产级向量库
#     千万别一上来就上 Milvus,会淹死在运维细节里,核心概念却还没建立起来。

section("全部演示完成")
