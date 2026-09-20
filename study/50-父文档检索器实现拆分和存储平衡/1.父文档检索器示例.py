'''
Date: 2026-09-17
Author: parker
FilePath: \llmops-api\study\50-父文档检索器实现拆分和存储平衡\1.父文档检索器示例.py
Description: ParentDocumentRetriever —— 小块负责被搜到，大块负责被返回
'''
import dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import InMemoryStore

dotenv.load_dotenv()

# 本地 Embedding，和前面几节保持一致（bge-small-zh，中文小模型，够用且不花钱）
embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")


# ============================================================================
# 第 1 部分：最小可用 —— 把"三件套"组装成一个检索器
# ============================================================================
# 上一段说的"存两份"，落到代码里就是下面 3 个对象：
#
#   ① vectorstore   装 child 的【向量库】   —— 靠它"搜"
#   ② docstore      装 parent 的【文档库】  —— 靠它"还原文"
#   ③ child_splitter 切 child 的【切分器】  —— 决定小块多小
#
# 顺序上先造零件、再组装、最后投料，别跳步。

# ① 向量库：还是熟悉的 FAISS。
#
#    ⚠️ 这里踩到一个版本坑，值得记下来：
#       课程视频里写的是 FAISS(embedding_function=embedding) —— 老版本允许这样建空库。
#       但 langchain-community 0.4.2 里，构造函数变成了必须传
#           index、docstore、index_to_docstore_id 三个参数。
#       而 index 的【维度】是由第一条向量决定的 —— 一条数据都没有，就不知道维度，
#       也就造不出 index。这是个死锁：想建空库，可空库建不出来。
#
#    解法和它的道理：
#      先用 from_texts 塞【一条占位文档】进去 —— 这一刻 FAISS 才知道
#      "哦，这个 embedding 是 512 维的"，于是把 index 建出来；
#      然后立刻把这条占位文档删掉，只剩一个 ntotal: 0 的空壳。
#
#    （另一种更干净的写法是换成 langchain_core 自带的 InMemoryVectorStore，
#      它天生支持空库。这里坚持用 FAISS 是为了跟前面 23 节保持一致，
#      也提醒你一件事：向量库是"可插拔"的，检索器并不在乎你用哪个。）
vectorstore = FAISS.from_texts(["__seed__"], embedding)   # 塞占位文档（只为撑出 index 的维度）
vectorstore.delete(list(vectorstore.index_to_docstore_id.values()))  # 立刻删掉，留下空库

# ② 文档库：这里【故意不用】向量库。
#    因为 parent 不需要算向量、也不参与相似度，它只是被"按 id 查回来"而已，
#    本质就是个字典。LangChain 给了现成的 InMemoryStore（存在内存里，关掉就没了）。
#    【填空 2】创建一个 InMemoryStore，变量名叫 docstore。
docstore = InMemoryStore()

# ③ 切 child 的切分器：小块才有准的向量，所以刻意切小。
#    这里 chunk_size 给 60 是故意的 —— 60 个字只能讲清一件事，
#    这样它的向量才会"指向明确"。等会儿第 2 部分你会看到它有多小。
#    【填空 3】用 RecursiveCharacterTextSplitter 建切分器，变量名 child_splitter，
#             chunk_size=60，chunk_overlap=10。
child_splitter = RecursiveCharacterTextSplitter(chunk_size=60, chunk_overlap=10)

# ④ 组装。这一步没有任何魔法，就是把上面三个对象塞进去而已。
#    注意 parent_splitter 这次【不传】—— 不传的含义是：
#    "原始文档本身就是 parent"，下面 ⑤ 加进去的那整篇就是大块。
#    （第 2 部分我们再传它，看两级拆分长什么样）
#    【填空 4】创建 retriever = ParentDocumentRetriever(...)，传三个参数：
#             vectorstore、docstore、child_splitter（名字要和构造函数的形参一致）
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,   # 装 child 的向量库
    docstore=docstore,         # 装 parent 的文档库
    child_splitter=child_splitter,  # 决定小块切多小
)


# ⑤ 投料。这一步看着只是"加一篇文档"，内部其实悄悄干了两件事：
#      切开 → child 进向量库（顺带把 embedding 算好）、parent 进文档库
#    也就是说：你只调了一次，它帮你写了两个库。
CONTENT = (
    "无线蓝牙耳机 Pro 采用蓝牙 5.3 技术，支持主动降噪和通透模式。"
    "单次充电可以使用约 8 小时，配合充电盒总续航时间约 32 小时。"
    "耳机支持双设备连接，适合通勤、运动和办公场景。"
    "充电盒支持 Type-C 快充，充电 10 分钟可听歌约 2 小时。"
    "耳机本体支持 IPX4 级防水，日常出汗和小雨不受影响。"
    "配对方式：打开充电盒盖，手机蓝牙列表中选择 SoundMax Pro 即可完成连接。"
)
#    【填空 5】调用 retriever.add_documents(...)，把上面这篇包成 [Document(page_content=CONTENT)] 传进去。
#    提示：为什么必须是【列表】而不是单个 Document？因为源码里就是按批处理的。
retriever.add_documents([Document(page_content=CONTENT)])

# ⑥ 检索。这里要盯住一个"反直觉"的现象：
#    你搜的关键词明明只在某一个小块里，返回的却是一整篇 parent。
#    而且返回条数可能比你预期少 —— 因为多个 child 可能领回同一个 parent，会被去重。
print("=" * 78)
print("第 1 部分：小块负责被搜到，大块负责被返回")
print("=" * 78)

QUERY = "降噪和通透模式是什么？"   # 这个词只在原文第 1 句附近出现
print(f"\n查询：{QUERY}")
print(f"（原文共 {len(CONTENT)} 字，child_splitter 的 chunk_size 只有 60）\n")

#    【填空 6】用 retriever.invoke(QUERY) 检索，for 循环打印每条结果的
#             len(doc.page_content) 和 doc.page_content[前20字]。
#    对比一下：如果这里换成普通的"切 60 字直接建 FAISS"，你拿到的会是 60 字的碎片。
results = retriever.invoke(QUERY)
print(f"检索到 {len(results)} 条：")
for i, doc in enumerate(results, 1):
    print(f"  [{i}] 长度 {len(doc.page_content)} 字 → {doc.page_content[:20]}...")


# ============================================================================
# 第 1 部分的答案与验证（2026-09-17 实跑）
# ============================================================================
# 实跑输出：
#
#   查询：降噪和通透模式是什么？
#   （原文共 195 字，child_splitter 的 chunk_size 只有 60）
#   检索到 1 条：
#     [1] 长度 195 字 → 无线蓝牙耳机 Pro 采用蓝牙 5.3 ...
#
# 两个"反直觉"的地方，答案是同一个：
#
#   ★ 问：为什么返回 195 字，而不是切成 60 字的小块？
#     答：因为向量库和文档库里存的根本不是一样东西。把两个库扒开看：
#
#         === 向量库里的 child（共 5 条）===
#           child[0] 46字 doc_id=0a6b9fd2-... | 无线蓝牙耳机 Pro 采用蓝牙 5.3 技术，支持主动降噪和通透模式。单次充...
#           child[1] 58字 doc_id=0a6b9fd2-... | 8 小时，配合充电盒总续航时间约 32 小时。耳机支持双设备连接...
#           child[2] 39字 doc_id=0a6b9fd2-... | Type-C 快充，充电 10 分钟可听歌约 2 小时。耳机本体支持 IPX4
#           child[3] 55字 doc_id=0a6b9fd2-... | IPX4 级防水，日常出汗和小雨不受影响。配对方式：打开充电盒盖...
#           child[4] 11字 doc_id=0a6b9fd2-... | Pro 即可完成连接。
#
#         === 文档库里的 parent（共 1 条）===
#           key=0a6b9fd2-...  195字
#
#     注意看 doc_id 那一列：**5 个 child 的 doc_id 完全一样**，
#     而文档库里那条 parent 的 key，正好就是这个 doc_id。
#     所以 invoke 的过程是：向量库比出最像的 child → 读出它身上的 doc_id
#     → 拿这个 id 去文档库换回 195 字的原文 → 返回给你。
#     你拿到的从来不是 child，child 只是"引路的那根线"。
#
#   ★ 问：为什么只有 1 条，不是默认的 k=4？
#     答：k=4 是去【向量库】捞 4 个 child 的上限。但这 4 个 child 很可能是
#     同一篇原文切出来的 —— 它们的 doc_id 一样，去文档库换回来就是同一篇。
#     ParentDocumentRetriever 会按 doc_id 去重，于是 4 条塌缩成 1 条。
#     本例 5 个 child 全属于同一篇，无论 k 开多大，结果都只会是 1 条。
#
# ★ 反过来想，这才是这一节叫"拆分和存储平衡"的原因：
#   小块负责【搜得准】，大块负责【答得全】，两个诉求由两个库分别承担，
#   谁也不用为对方妥协。代价是——存了两份，写入了两次，还要多维护一个库。
#
# ⚠️ 留一个坑给下一部分：上面这个例子里 parent = 原始文档本身，
#    因为我们【没有传 parent_splitter】。原文只有 195 字还好，
#    真实文档动辄几万字，一整个塞进 prompt 会撑爆上下文窗口。
#    所以下一部分要加 parent_splitter，把 parent 也切一刀。
#    ——但这样一来，"大块"到底该多大，就变成了一个新的取舍。这就是标题里的"平衡"。


# ============================================================================
# 第 2 部分：加 parent_splitter —— 两级拆分
# ============================================================================
# 加它之前，先想清楚它是干嘛的。一句话：
#
#     parent_splitter  作用在【原始文档】上，切成若干"中块"
#     child_splitter   作用在【每个 parent 内部】，再把中块切成"小块"
#
# 所以是真·两级：原始文档 →(parent_splitter)→ 中块 →(child_splitter)→ 小块
# 也正因为如此，parent_splitter 的 chunk_size 必须【大于】child_splitter 的，
# 否则小块反而比大块还长，"大小块"这个区分就没意义了。
#
# ⚠️ 还有一个必须注意的地方：第 1 部分的 vectorstore / docstore 里【已经有数据了】。
#    如果直接拿它们做第 2 部分的实验，新旧数据会混在一起，你就看不出
#    parent_splitter 到底切出了几块。所以下面要【重新建两个干净的空库】。

# 【填空 A】建 parent_splitter，变量名 parent_splitter，chunk_size=100、chunk_overlap=0。
#    想一想：为什么这里 chunk_overlap 给 0，而 child_splitter 给了 10？
parent_splitter = ____

# 【填空 B】重新建一对干净的空库（照抄第 1 部分那两行即可），
#    变量名分别叫 vs2 和 ds2，别覆盖掉上面的 vectorstore / docstore。
vs2 = ____
ds2 = ____

# 【填空 C】组装第二台检索器 retriever2。
#    参数和第一部分一样，只是这次【多传一个 parent_splitter】。
#    比较一下两次的差别到底在哪一行 —— 就只多了这一个参数。
retriever2 = ____

# 投同一篇文档，这次看两个库里的数量变化：
retriever2.add_documents([Document(page_content=CONTENT)])

print()
print("=" * 78)
print("第 2 部分：加 parent_splitter 之后")
print("=" * 78)

# 【填空 D】打印两个库各自有多少条，用这个格式：
#      向量库里的 child：{len(vs2.index_to_docstore_id)} 条
#      文档库里的 parent：{len(ds2.store)} 条
#    顺便把文档库里每条 parent 的长度和前 25 个字也打出来。
#    提示：ds2.store 是个 dict，.items() 能拿到 (doc_id, Document) 对。
____

# ★ 填完先别看结果，自己预测一下再跑：
#    原文 195 字，parent_splitter 的 chunk_size 是 100，那应该切出几块？
#    每块再按 60 字切 child，child 总数大概是多少？
#    预测完再跑，看看和你想的一不一样。猜错的地方，就是你真正学到的地方。
