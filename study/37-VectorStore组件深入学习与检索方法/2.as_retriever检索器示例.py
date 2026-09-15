'''
Date: 2026-09-15 14:08:36
Author: parker
FilePath: \llmops-api\study\37-VectorStore组件深入学习与检索方法\2.as_retriever检索器示例.py
Description: as_retriever —— 把向量库包装成"检索器(Retriever)组件"

运行方式(在项目根目录):
    python "study/37-VectorStore组件深入学习与检索方法/2.as_retriever检索器示例.py"
'''

# ============================================================================
# 第 0 部分:已经有了 similarity_search,为什么还要 as_retriever?
# ============================================================================
#
# 上一个文件里我们是这么用的:
#
#     results = db.similarity_search_with_relevance_scores("我养了一只猫，叫笨笨")
#
# 直接调 db 的方法,拿到一个列表,打印出来 —— 到这一步没问题。
#
# 但**只用到这里是不够的**。真正做 RAG 的时候,检索只是链条中间的一环:
#
#     用户提问 → 检索相关文档 → 把文档塞进提示词 → 丢给大模型 → 输出答案
#
# 如果你只有 db.similarity_search() 这个**方法**,你就得手写一大段"胶水代码"
# 把这个方法嵌进链条里。而 LangChain 里所有能串起来的东西,统一都是 Runnable
# (就是有 invoke / batch / stream 这一套标准接口的对象)。
#
# as_retriever() 干的事就一件:**把向量库这个对象,适配成一个 Runnable**。
# 适配完之后,它就能和其它组件用 | 拼起来用了。
#
# ===== 一句话对比 =====
#   db.similarity_search(q)   → 是 VectorStore 的一个**方法**,只能用在这里
#   db.as_retriever()         → 返回一个**组件**(VectorStoreRetriever),
#                               是 Runnable,能 invoke / batch,能接进链
#
# 记住这个区分,第 4 部分会用实际代码让你看到差别。


import dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

dotenv.load_dotenv()

embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")

# 这次换一批**带业务元数据**的文档 —— 一个电商客服知识库。
# 为什么换?因为 as_retriever 有个很重要的能力是"按元数据过滤",
# 上一批文档只有 page 字段,演示不出来。
knowledge_base = [
    Document(
        page_content="退货政策:自签收之日起七天内,商品未拆封可申请无理由退货。生鲜类目不支持无理由退货。",
        metadata={"category": "售后", "source": "help_aftersale.md"},
    ),
    Document(
        page_content="发票说明:下单时可申请电子发票,支持个人和企业抬头。发票将在订单完成后三个工作日内开具。",
        metadata={"category": "售后", "source": "help_aftersale.md"},
    ),
    Document(
        page_content="发货时效:现货商品在付款后24小时内发出。预售商品以商品详情页标注的发货时间为准。",
        metadata={"category": "物流", "source": "help_logistics.md"},
    ),
    Document(
        page_content="运费说明:单笔订单满99元包邮。偏远地区(新疆、西藏、内蒙古)需补收15元运费。",
        metadata={"category": "物流", "source": "help_logistics.md"},
    ),
    Document(
        page_content="会员等级:分为普通会员、黄金会员、铂金会员三档。黄金会员享9.5折,铂金会员享9折。",
        metadata={"category": "会员", "source": "help_vip.md"},
    ),
    Document(
        page_content="积分规则:每消费1元累积1积分,积分可在下单单笔抵扣,100积分抵1元。",
        metadata={"category": "会员", "source": "help_vip.md"},
    ),
    Document(
        page_content="支付方式:支持微信支付、支付宝、银联卡。部分商品支持花呗分期,最长12期免息。",
        metadata={"category": "支付", "source": "help_payment.md"},
    ),
    Document(
        page_content="支付安全:全站采用银行级TLS加密传输,平台不存储用户的银行卡号与支付密码。",
        metadata={"category": "支付", "source": "help_payment.md"},
    ),
]

db = FAISS.from_documents(knowledge_base, embedding)

# print("=" * 74)
# print("第 0 部分:准备了一个电商客服知识库")
# print("=" * 74)
# print(f"共 {len(knowledge_base)} 条文档,分属 4 个类目:")
# for cat in ["售后", "物流", "会员", "支付"]:
#     n = sum(1 for d in knowledge_base if d.metadata["category"] == cat)
#     print(f"    {cat}: {n} 条")


# # ============================================================================
# # 第 1 部分:最基本的用法 —— as_retriever() + k
# # ============================================================================
# # 最朴素的调用就是一句 db.as_retriever(),什么参数都不传。
# #
# # 然后重点来了:retriever 的用法**不是** similarity_search,而是 invoke。

# retriever = db.as_retriever()

# print()
# print("=" * 74)
# print("第 1 部分:最朴素的 as_retriever()")
# print("=" * 74)

# docs = retriever.invoke("我想退货怎么办")

# print('提问: "我想退货怎么办"')
# print(f"返回 {len(docs)} 条 Document:\n")
# for i, d in enumerate(docs):
#     print(f"  [{i}] {d.page_content}")
#     print(f"       metadata = {d.metadata}")

# 划重点 1:为什么是 4 条?
#   as_retriever() 的默认 k 就是 4 —— 和上个文件 similarity_search 默认只返回 4 条
#   是**同一个原因**:底层默认 k=4。想多拿就显式传 k。
#
# 划重点 2:返回的是 list[Document],**没有分数**。
#   回想上个文件,similarity_search_with_relevance_scores 返回的是
#   (Document, score) 的二元组。而 retriever 默认只给你文档本身。
#   这是刻意的设计:检索器是给**下游链条**用的,下游只关心"给我哪几段文字",
#   不关心分数。真要分数,得换 search_type(见第 3 部分)。

# ---- 用 search_kwargs 调 k ----
# 所有检索参数都装在 search_kwargs 这个字典里,
# 而不是直接写 as_retriever(k=6) —— 这点很容易记错。

# retriever_k6 = db.as_retriever(search_kwargs={"k": 6})
# docs_k6 = retriever_k6.invoke("我想退货怎么办")

# print()
# print("-" * 74)
# print('改成 search_kwargs={"k": 6} 之后:')
# print(f"  返回 {len(docs_k6)} 条")
# for i, d in enumerate(docs_k6):
#     print(f"  [{i}] [{d.metadata['category']}] {d.page_content[:34]}...")

# 看到没?k=6 就把 6 条捞上来了 —— 包括跟"退货"八竿子打不着的
# "支付安全""积分规则"。这就是为什么 k 不能乱调大:
# 检索回来的每一段都要**塞进大模型的提示词**,无关内容会稀释真正的答案。
# k 太小会漏,太大会稀释,一般 3~8 之间试。


# ============================================================================
# 第 2 部分:先把"相关性分数"到底是什么搞明白
# ============================================================================
# 第 3 部分要设阈值,而设阈值的前提是知道分数**是怎么算出来的**。
# 上个文件留下了 0.70 / 0.43 / 0.31 / 0.17 四个数,它们到底是不是余弦相似度?
#
# 直接实测:把底层的原始距离和手算的余弦都打出来对一遍。

# import numpy as np

# q_probe = "退货运费谁出"
# q_vec = np.array(embedding.embed_query(q_probe))

# print()
# print("=" * 74)
# print("第 2 部分:relevance_score 的换算依据(实测)")
# print("=" * 74)
# print(f'探针问题: "{q_probe}"')
# print(f"query 向量的模长 = {np.linalg.norm(q_vec):.6f}  (1.0 说明模型已做归一化)")

# # similarity_search_with_score 给的是**底层原始距离**(越小越近),
# # similarity_search_with_relevance_scores 给的是**换算后的相关性分数**(越大越相关)。
# scored = db.similarity_search_with_relevance_scores(q_probe, k=3)
# raw = db.similarity_search_with_score(q_probe, k=3)

# print()
# print(f"{'文档':<22}{'原始距离':>10}{'relevance':>12}{'手算余弦':>12}{'1-距离/√2':>12}")
# print("-" * 70)
# for (doc, rel), (_, dist) in zip(scored, raw):
#     d_vec = np.array(embedding.embed_documents([doc.page_content])[0])
#     cos = float(np.dot(q_vec, d_vec) / (np.linalg.norm(q_vec) * np.linalg.norm(d_vec)))
#     manual = 1 - float(dist) / np.sqrt(2)
#     print(
#         f"{doc.page_content[:10]:<22}{float(dist):>10.4f}{float(rel):>12.6f}"
#         f"{cos:>12.6f}{manual:>12.6f}"
#     )

# 看最后两列:relevance 和 "1-距离/√2" 完全一致。
# 所以 LangChain 对 FAISS 的换算公式就是:
#
#     relevance_score = 1 - 原始距离 / √2
#
# 再代入一个事实:归一化向量的平方欧氏距离 = 2 - 2×余弦,化简之后得到 ——
#
#     ★ relevance_score = 1 - √2 × (1 - 余弦) ≈ 1.4142 × 余弦 - 0.4142
#
# 这个结论非常关键,它说明三件事:
#
#   1. 分数**不是**余弦相似度本身,而是余弦的**线性拉伸**。
#      斜率是 √2≈1.4142,所以分数之间的差距会被放大 1.4 倍。
#
#   2. 因为是线性变换,**排序结果和按余弦排完全一样**。
#      所以"谁排第一"你可以放心信,但"分数绝对值是多少"别当余弦读。
#
#   3. 分数**可以是负数**,而且负得不少。对应的余弦换算关系:
#
#         余弦 1.00 → 分数  1.0000   (完全相同)
#         余弦 0.80 → 分数  0.7172
#         余弦 0.50 → 分数  0.2929   (★ 注意这里!)
#         余弦 0.29 → 分数  0.0000   ← 分数为 0 的点
#         余弦 0.00 → 分数 -0.4142   (正交,纯无关)
#         余弦 -1.0 → 分数 -1.8284
#
#      ★ 最重要的一条:分数 = 0 **不代表"不相关"**,它对应余弦 0.29。
#        因为 BGE 这类模型存在"向量空间各向异性"—— 任意两段中文的余弦
#        普遍偏高,0.29 在这个空间里已经算相当不相关了。
#        所以你看到负数别慌,那是正常的。
#
# 顺带解释你会看到的一条警告:
#     命令行里可能会出现 "UserWarning: Relevance scores must be between 0 and 1"。
#     那就是分数掉到负数触发的,不是你的代码写错了。
#
# ⚠️ 最后一个提醒:上面这个公式是 **FAISS 专属**的。
#   相关性分数怎么从距离换算,由每个向量库自己的实现决定
#   (Chroma、Qdrant、Milvus 的换算规则各不相同)。
#   所以 —— **别把一个项目里调好的阈值照抄到另一个项目**,它没有可比性。


# ============================================================================
# 第 3 部分:search_type="similarity_score_threshold" —— 带阈值过滤
# ============================================================================
# 第 1 部分留下一个问题:k 是**固定**的。你问一个问题,不管库里有没有相关内容,
# 它都硬塞给你 k 条。
#
# 这就很危险 —— 用户问一个知识库里**压根没有**的问题时,你照样把最不相关的那几条
# 喂给大模型,模型就会拿着这段不相干的文字**一本正经地编**(幻觉)。
#
# 正确的做法:设一个分数下限,达不到的**直接丢掉**,宁可返回空。
# 这就是 search_type="similarity_score_threshold"。
#
# ===== 阈值该设多少?先看分布,不要拍脑袋 =====
# 由第 2 部分的公式可知,阈值直接对应一个余弦下限。先量一下这个知识库的分布:

# print()
# print("=" * 74)
# print("第 3 部分:similarity_score_threshold 带阈值检索")
# print("=" * 74)
# print()
# print("先看三组问题的分数分布,再决定阈值:")
# for q in ["退货运费谁出", "你们公司创始人的生日是哪天"]:
#     print(f'\n  "{q}"')
#     for d, s in db.similarity_search_with_relevance_scores(q, k=8):
#         print(f"      {s:>8.4f}  [{d.metadata['category']}] {d.page_content[:30]}")

# 从上面能读出来:
#   - 问题在库里**有**答案时,最相关的能到 0.37 左右
#   - 问题在库里**没有**答案时,最高才 0.18,而且一大半是负数
# 所以阈值取 0.30 能把这两类分开。那就设 0.30。

# threshold_retriever = db.as_retriever(
#     search_type="similarity_score_threshold",
#     search_kwargs={"score_threshold": 0.30, "k": 4},
# )

# print()
# print("-" * 74)
# print('search_kwargs={"score_threshold": 0.30, "k": 4}')

# q_hit = "退货运费谁出"
# hits = threshold_retriever.invoke(q_hit)
# print()
# print(f'提问(库里有相关内容): "{q_hit}"')
# print(f"  → 返回 {len(hits)} 条")
# for d in hits:
#     print(f"      [{d.metadata['category']}] {d.page_content[:36]}...")

# ---- 再问一个知识库里**没有**答案的问题 ----
# q_miss = "你们公司创始人的生日是哪天"
# misses = threshold_retriever.invoke(q_miss)
# print()
# print(f'提问(库里完全没有的内容): "{q_miss}"')
# print(f"  → 返回 {len(misses)} 条")

# if len(misses) == 0:
#     print("      (空列表 —— 知识库诚实地告诉你:我不知道)")
# else:
#     print("      阈值没挡住,下面这些就是会被喂给模型的噪声:")
#     for d in misses:
#         print(f"      {d.page_content[:40]}...")

# 这就是这个 search_type 的全部价值:**敢于返回空**。
# 下游拿到空列表,你就可以回一句"抱歉,知识库里没有这方面的资料",
# 而不是让模型瞎编。

# ---- 但是,阈值不是银弹,这里必须说清楚它的局限 ----
# print()
# print("-" * 74)
# print("阈值不是银弹 —— 换一个问题就露馅了:")
# print("-" * 74)
# q_awkward = "运费怎么算"
# print(f'提问: "{q_awkward}"')
# for d, s in db.similarity_search_with_relevance_scores(q_awkward, k=3):
#     flag = "保留" if s >= 0.30 else "丢弃"
#     print(f"      [{flag}] {s:>8.4f}  [{d.metadata['category']}] {d.page_content[:30]}")

# 看到没?"积分规则"以 0.3866 混进来了 —— 它跟"运费"毫无关系,
# 但分数比上一个问题里**真正相关的**退货政策(0.3687)还高。
#
# 原因有两个,都很本质:
#   1. 一个**全局**阈值没法适应所有问题。同一个 0.37,
#      在"退货运费谁出"里是正牌答案,在"运费怎么算"里就是噪声。
#   2. 语义相似不等于业务相关。"运费"和"积分抵扣"都跟钱有关,
#      向量空间里本来就离得近。
#
# 所以真实项目里通常**组合使用**:
#   阈值(挡掉明显不相关的) + 明确业务规则的 filter(下一部分),
#   而不是指望一个数字解决所有问题。


# ============================================================================
# 第 4 部分:filter —— 按元数据过滤(真实项目里最有用的一个)
# ============================================================================
# 场景:知识库里混着售后、物流、会员、支付四个类目的文档。
# 用户问"运费怎么算",就像上面看到的,语义检索会把"积分规则"也捞回来。
#
# 但业务上你明确知道:这个问题只该在**物流**类目里找。
# 那就直接告诉检索器:**别在别的类目里浪费时间**。
#
# filter 是按 metadata 做**硬过滤** —— 不满足条件的文档直接不参与检索。
# 这比调相似度阈值更可靠,因为它是你**确定的业务规则**,不依赖模型算得准不准。

# filtered_retriever = db.as_retriever(
#     search_kwargs={"k": 4, "filter": {"category": "物流"}},
# )

# print()
# print("=" * 74)
# print("第 4 部分:filter 按元数据硬过滤")
# print("=" * 74)

# print(f'提问: "{q_awkward}"')
# print()
# print("  不加 filter,全库检索:")
# for d in db.as_retriever(search_kwargs={"k": 4}).invoke(q_awkward):
#     print(f"    [{d.metadata['category']}] {d.page_content[:34]}...")

# print()
# print('  加 filter={"category": "物流"},只在物流类目里检索:')
# for d in filtered_retriever.invoke(q_awkward):
#     print(f"    [{d.metadata['category']}] {d.page_content[:34]}...")

# 对比两组的类目标签。加了 filter 之后,返回的**每一条都是物流类目**,
# 上一部分那个混进来的"积分规则"不见了。
#
# 实际项目里这个用法非常常见:
#     filter={"user_id": "u123"}      多租户,只能检索自己的数据
#     filter={"doc_type": "合同"}      只在某类文档里找
#     filter={"date": "2026-09"}       只找最新的
# 它同时也是**权限控制**的手段 —— 不该看的数据,压根不进检索范围,
# 而不是检索出来再想办法过滤掉(那时候模型可能已经看到了)。


# ============================================================================
# 第 5 部分:retriever 是个 Runnable —— 这才是 as_retriever 的意义
# ============================================================================
# 前面说了半天"retriever 是 Runnable",现在看看这到底带来了什么。
# retriever = db.as_retriever()
# print()
# print("=" * 74)
# print("第 5 部分:作为 Runnable 的能力")
# print("=" * 74)

# # ---- 能力 1:batch 批量检索 ----
# # 数据库方法得写 for 循环,retriever 一行搞定。
# # 注意返回结构是**两层列表**:外层对应每个问题,内层是该问题的文档。
# questions = ["怎么退货", "运费多少钱", "会员有什么优惠"]
# batched = retriever.batch(questions)

# print("batch 批量检索:")
# for q, ds in zip(questions, batched):
#     print(f'  "{q}" → {len(ds)} 条, 最相关的是: {ds[0].page_content[:26]}...')

# ---- 能力 2:接进链(这是 RAG 的标准写法)----
# 因为 retriever 是 Runnable,所以可以直接用 | 拼。
#
# 下面这条链的结构,就是 RAG 的骨架:
#
#     {"context": retriever | 拼成一段文字, "question": 原样透传}
#         | 提示词模板
#         | 大模型
#         | 输出解析
#
# 左边那个字典是关键:它在调用时把用户的问题**同时**发往两个方向 ——
# 一路去检索(变成 context),一路原样当 question 保留。
# RunnablePassthrough() 就是"原样透传"的意思。

# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.runnables import RunnablePassthrough
# from langchain_deepseek import ChatDeepSeek

# prompt = ChatPromptTemplate.from_template(
#     """你是一个电商客服。请**只根据下面提供的资料**回答用户问题。
# 如果资料里没有相关内容,就直接说【资料里没有,我帮你转人工】,不要自己编。

# 资料:
# {context}

# 用户问题:{question}
# """
# )

# llm = ChatDeepSeek(model="deepseek-chat", temperature=0)

# retriever = db.as_retriever()
# def format_docs(docs):
#     """把检索到的 Document 列表拼成一段纯文本,方便塞进提示词。"""
#     return "\n\n".join(d.page_content for d in docs)


# rag_chain = (
#     {"context": retriever | format_docs, "question": RunnablePassthrough()}
#     | prompt
#     | llm
#     | StrOutputParser()
# )

# print()
# print("-" * 74)
# print("把 retriever 接进 RAG 链:")
# print("-" * 74)

# for q in ["我想退货怎么办", "你们公司创始人的生日是哪天"]:
#     print(f"\n用户: {q}")
#     print(f"客服: {rag_chain.invoke(q)}")

# 看第二个问题的回答 —— 这就是第 3 部分讲的那个道理在起作用:
# 知识库里没有答案时,你希望模型说"不知道",而不是编一个生日出来。
#
# 整条链里,retriever 就是"检索"这一环。as_retriever() 存在的全部意义,
# 就是让它能以 Runnable 的身份,和其他所有组件用同一个 | 串起来。


# ============================================================================
# 第 6 部分:还有哪些 search_type?
# ============================================================================
# search_type 目前有三个可选值:
#
#   "similarity"(默认)              纯粹的相似度检索,返回 k 条,不管分数多低
#   "similarity_score_threshold"     相似度 + 阈值过滤,达不到就丢(第 3 部分)
#   "mmr"                            最大边际相关性,追求**结果多样性**
#
# mmr 解决的是另一个问题:如果库里有一堆**内容高度重复**的文档,
# 纯相似度检索会把 k 个名额全给这些重复内容,别的角度一条都进不来。
# mmr 会先捞一批候选,再从中挑出"既相关、又彼此不重复"的 k 条。
#
# 它的参数(fetch_k、lambda_mult)稍微绕一点,而且是个独立的文件,
# 所以放在 3.最大边际相关性示例.py 里单独讲。


# ============================================================================
# 小结
# ============================================================================
#   一句话:as_retriever() = 把 VectorStore 变成 Runnable,让它能接进链。
#
#   必须记住的三点:
#     1. 用法是 retriever.invoke(q),不是 similarity_search;
#        所有参数都装在 search_kwargs 字典里,默认 k=4
#     2. 默认不返回分数。要分数就用 search_type="similarity_score_threshold",
#        它还能让检索器在**没有相关内容时返回空** —— 这是防幻觉的关键
#     3. filter 是**业务规则硬过滤**,比调相似度阈值可靠得多,
#        同时也是多租户/权限隔离的手段
#
#   关于分数的依据(第 2 部分的结论,值得单独记):
#     FAISS 的 relevance_score = 1 - √2×(1-余弦),是余弦的线性拉伸。
#     排序可信,但绝对值不是余弦,可以为负,且 0 对应余弦 0.29 而非 0。
#     换个向量库这个换算规则就变,阈值**不可移植**。
#
#   什么时候用哪个:
#     只是想看看检索效果        → db.similarity_search_with_relevance_scores(上个文件)
#     要做 RAG / 接进链         → db.as_retriever()(本文件)
#     要防幻觉、允许回答"不知道" → search_type="similarity_score_threshold"
#     有明确业务边界            → search_kwargs={"filter": {...}}
#     结果重复度太高            → search_type="mmr"(下个文件)
