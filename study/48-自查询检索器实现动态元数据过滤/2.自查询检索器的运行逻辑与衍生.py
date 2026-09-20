'''
Date: 2026-09-17
Author: parker
FilePath: \llmops-api\study\48-自查询检索器实现动态元数据过滤\2.自查询检索器的运行逻辑与衍生.py
Description: 拆开 SelfQueryRetriever 看它的四步运行逻辑,实测三个关键参数,再看它能衍生出什么用法

运行方式(在项目根目录):
    python "study/48-自查询检索器实现动态元数据过滤/2.自查询检索器的运行逻辑与衍生.py"

依赖:lark(上一节已装)。本文件自带一份精简的书库和翻译器,可以独立运行。
'''

# ============================================================================
# 第 0 部分:上一节留下的三个疑问
# ============================================================================
# 上一节我们把自查询跑起来了,但有三件事是"黑盒":
#   1. retriever.invoke("...") 内部到底走了几步?
#   2. from_llm 还有 enable_limit / use_original_query 这些参数,干嘛的?
#   3. 它跟前面学的 MultiQuery(40章)、Step-Back(43章)是什么关系?
# 这一节把这三个都拆开。**看懂运行逻辑,你才知道出问题时该去哪一层查.**

import dotenv

from langchain_classic.chains.query_constructor.base import (
    AttributeInfo,
    StructuredQueryOutputParser,
    get_query_constructor_prompt,
)
from langchain_classic.retrievers.self_query.base import SelfQueryRetriever
from langchain_core.structured_query import (
    Comparator,
    Comparison,
    Operation,
    Operator,
    Visitor,
)
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_deepseek import ChatDeepSeek
from langchain_huggingface import HuggingFaceEmbeddings

# ============================================================================
# 【名词表】本课新导入的东西,分别是干嘛的
# ============================================================================
# 上面 import 了一大堆,没一个是你之前见过的。按"它们从哪来"分三个家族:
#
#   langchain_core      -> 底层通用抽象:数据结构、基类、Runnable 协议。轻,谁都用
#   langchain_classic   -> "经典"实现:检索器、链。原来 0.x 版 langchain 包里的东西搬到这了
#   langchain_community -> 第三方集成:各种向量库、文档加载器
#
#   所以 import 路径长得像 langchain_classic.chains.xxx 还是 langchain_core.xxx,
#   本身就透露了"这是现成的功能"还是"这是我要继承/使用的抽象"。
#
# ---------------------------------------------------------------------------
# ① 查询构建三件套(来自 langchain_classic.chains.query_constructor.base)
# ---------------------------------------------------------------------------
#   AttributeInfo              **类**。一个"字段说明书"对象,三个参数:
#                                 name        字段名(和 metadata 的 key 一字不差)
#                                 description 写给模型看的说明
#                                 type        类型("string"/"integer"/"float")
#                              它是 46 章 RouteQuery 的"表亲":RouteQuery 描述"一个字段
#                              (datasource)",AttributeInfo 描述"一张表里的一列"。
#                              出现在:第 2 部分,拼进提示词,告诉模型"有哪些字段能过滤"。
#
#   get_query_constructor_prompt  **函数**。给它(书库描述 + 字段说明书列表),
#                              它返回一个**提示词模板**。这个模板里已经写好了
#                              "请把用户问题拆成 query 和 filter"这一整套指令和示例。
#                              出现在:第 3 部分。相当于第 1 章学的 PromptTemplate,
#                              只是内容 langchain 帮你写好了,不用自己编。
#
#   StructuredQueryOutputParser **类**。`.from_components()` 造实例。
#                              它专门把模型吐出的一行文本解析成 StructuredQuery 对象。
#                              出现在:第 3 部分链条的最后一环。
#                              它和你在第 3 章学的 StrOutputParser / JsonOutputParser
#                              **是同一类东西**(都是 OutputParser),只是产物是结构化查询。
#                              ⚠️ 它靠 lark 包干活,没装 lark 这行就崩。
#
# ---------------------------------------------------------------------------
# ② 检索器本体(来自 langchain_classic.retrievers.self_query.base)
# ---------------------------------------------------------------------------
#   SelfQueryRetriever         **类**。把上面三样 + 向量库 + translator 组装成一个
#                              `.invoke()` 就能用的检索器。
#                              它和 38 章那些检索器**是同类**:都继承 BaseRetriever,
#                              都有 .invoke()/.batch(),都能塞进 LCEL 管道。
#
# ---------------------------------------------------------------------------
# ③ 条件树的数据结构(来自 langchain_core.structured_query)
# ---------------------------------------------------------------------------
#   这四个名字是**模型给出的那份"拆解结果"的零件**,不是给你调用的工具:
#
#   StructuredQuery  **数据结构**。拆解的最终产物,三个字段:
#                        query  = 语义部分("Python")
#                        filter = 条件树(可能是 None)
#                        limit  = 条数(可能是 None)
#                    就是"一句话被拆成的两半"的容器。
#
#   Comparison       **数据结构**,一条比较条件 = 属性 + 比较符 + 值。
#                    条件树的**叶子**。例:year > 2020
#
#   Operation        **数据结构**,把几个子条件用 AND/OR 连起来。
#                    条件树的**树枝**。例:(year>2020) AND (price<100)
#
#   Comparator       **枚举**,比较符:EQ/GT/LT/GTE/LTE/NE。
#                    就是 = > < >= <= ≠ 用对象表示(所以写的是
#                    `comparison.comparator == Comparator.GT`,而不是 `op == ">"`)
#
#   Operator         **枚举**,逻辑符:AND / OR / NOT。
#
#   Visitor          **基类**。一个"访问者模式"的约定:框架规定你实现
#                    visit_comparison / visit_operation / visit_not 这几个方法名,
#                    然后在遇到对应节点时**回调**你写的方法。
#                    我们继承它,是为了让 langchain 知道"怎么调用我的 translator"。
#                    ★ 认准这个套路:凡是继承 Visitor/BaseXxx 的类,你只要
#                      按它规定的方法名填内容,调用时机由框架负责。
#
# ---------------------------------------------------------------------------
# 一句话总结出场顺序:
#   AttributeInfo / get_query_constructor_prompt  -> 拼提示词(输入)
#   StructuredQueryOutputParser                   -> 解析模型输出(中间)
#   StructuredQuery / Comparison / Operation ...  -> 解析出来的零件
#   Comparator / Operator                         -> 零件里的枚举值
#   Visitor                                        -> 我们继承它,写"翻译器"
#   SelfQueryRetriever                             -> 把上面全部串起来用
# ============================================================================

dotenv.load_dotenv()

embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
llm = ChatDeepSeek(model="deepseek-chat", temperature=0)

# 精简书库(和上一节同一批书,只留最必要的字段)
BOOKS = [
    ("流畅的Python:深入理解 Python 语言特性", 2017, 139, "进阶"),
    ("Python编程:从入门到实践", 2020, 89, "入门"),
    ("Python数据科学手册:用 Python 做数据分析", 2018, 109, "入门"),
    ("Python Cookbook:Python 实用技巧", 2013, 108, "进阶"),
    ("算法导论:计算机算法经典教材", 2013, 128, "专家"),
]
docs = [
    Document(page_content=text, metadata={"year": year, "price": price, "level": level})
    for text, year, price, level in BOOKS
]
db = FAISS.from_documents(docs, embedding)

document_contents = "计算机技术书籍的简介"

metadata_field_info = [
    AttributeInfo(name="year", description="出版年份", type="integer"),
    AttributeInfo(name="price", description="价格,单位元", type="integer"),
    AttributeInfo(name="level", description="难度等级:入门/进阶/专家", type="string"),
]


class MiniFAISSTranslator(Visitor):
    """和上一节一样的迷你翻译器。这一节它不是重点,不重复解释了。"""

    def visit_structured_query(self, structured_query):
        if structured_query.filter is None:
            return structured_query.query, {}
        return structured_query.query, {"filter": self._to_filter(structured_query.filter)}

    def _to_filter(self, node):
        if isinstance(node, Comparison):
            return self.visit_comparison(node)
        if isinstance(node, Operation):
            return self.visit_operation(node)
        raise ValueError(f"不支持的条件类型: {type(node).__name__}")

    def visit_comparison(self, comparison):
        attr, val, op = comparison.attribute, comparison.value, comparison.comparator
        ops = {
            Comparator.EQ: lambda m: m.get(attr) == val,
            Comparator.GT: lambda m: m.get(attr) > val,
            Comparator.LT: lambda m: m.get(attr) < val,
            Comparator.GTE: lambda m: m.get(attr) >= val,
            Comparator.LTE: lambda m: m.get(attr) <= val,
        }
        return ops[op]

    def visit_operation(self, operation):
        args = [self._to_filter(a) for a in operation.arguments]
        if operation.operator == Operator.AND:
            return lambda m: all(f(m) for f in args)
        return lambda m: any(f(m) for f in args)

    def visit_not(self, not_) -> None:
        raise NotImplementedError


# ============================================================================
# 第 1 部分:运行逻辑 —— .invoke() 里面到底走了哪四步
# ============================================================================
# 直接给你结论。下面这段是 langchain_classic/retrievers/self_query/base.py 里
# _get_relevant_documents 的真实流程(我读了源码,不是猜的):
#
#   def _get_relevant_documents(self, query):
#       ① structured_query = self.query_constructor.invoke({"query": query})
#       ② new_query, new_kwargs = self.structured_query_translator.visit_structured_query(structured_query)
#       ③ if structured_query.limit is not None: new_kwargs["k"] = structured_query.limit
#          if self.use_original_query:           new_query = query
#          search_kwargs = {**self.search_kwargs, **new_kwargs}
#       ④ return self.vectorstore.search(new_query, self.search_type, **search_kwargs)
#
# 一步步翻译成人话:
#
#   ① **一次 LLM 调用**。把用户原话喂给"提示词|模型|解析器"那条链,拿回 StructuredQuery。
#      ★ 这是自查询的全部额外成本 —— 每次检索都要多花一次模型调用。
#        普通向量检索是纯本地的,自查询变成了"每次都得联网问模型"。
#
#   ② **翻译**。把条件树交给你的 translator,拿回 (改写后的查询词, 额外搜索参数)。
#      这一步是纯 Python,不调模型,快。注意"改写后的查询词"——
#      模型可能把"2020年以后出版的价格低于100元的Python书"缩成 "Python",
#      语义部分变干净了,干扰词被丢进 filter 里了。
#
#   ③ **合并参数**。三个小动作:
#        - 模型给了 limit 就覆盖 k(但要你开了 enable_limit 它才会给)
#        - use_original_query=True 时,把查询词换回用户原话
#        - {**search_kwargs, **new_kwargs}:⚠️ **new_kwargs 在后面,所以它会赢**。
#          意思是:构造检索器时传的 search_kwargs 会被本次翻译结果覆盖。
#
#   ④ **真正的检索**。到这一步已经是普通的向量库调用了,自查询的魔法全部结束。
#
# ★ 四步里,只有 ① 是"AI",②③④ 都是普通代码。
#   所以出问题时的排查顺序是:
#     结果不对 -> 先看 ① 拆出来的 StructuredQuery 对不对(模型的问题)
#              -> 再看 ② 翻译出来的函数对不对(你的 translator 的问题)
#              -> 最后才怀疑向量库(通常最不容易出错)


# ============================================================================
# 第 2 部分:三个参数实测
# ============================================================================


def build(**kwargs):
    return SelfQueryRetriever.from_llm(
        llm=llm,
        vectorstore=db,
        document_contents=document_contents,
        metadata_field_info=metadata_field_info,
        structured_query_translator=MiniFAISSTranslator(),
        **kwargs,
    )


print("=" * 78)
print("第 2 部分:enable_limit —— 让模型能说'我要 N 条'")
print("=" * 78)

Q = "推荐三本Python书"

print(f"\n  查询: {Q}")
print("\n  【不开 enable_limit】(默认 k=4)")
for d in build().invoke(Q):
    print(f"      {d.page_content[:22]}")

print("\n  【开 enable_limit=True】")
r_limited = build(enable_limit=True)
for d in r_limited.invoke(Q):
    print(f"      {d.page_content[:22]}")

# 实测(2026-09-17):
#   不开 enable_limit  -> 4 本(默认 k=4 填满)
#   开 enable_limit    -> 3 本(模型拆出 limit=3,k 被覆盖成 3)
#
# ★ 原理:enable_limit=True 会往提示词的 schema 里**多塞一个 limit 字段** ——
#   和 46 章往 RouteQuery 里加一个字段是完全一样的动作。
#   模型看到有这个字段可填,才会在"三本"这种量词出现时填上数字。
#   不开的话,提示词里没有这个格子,模型的"三本"就白说了。
#
# ★ 但它不可靠:limit 有没有、填成几,全看模型这次的心情。
#   上面拆出来是 limit=3,你换个说法("来几本Python书")它可能就 limit=None。
#   所以**别把条数控制交给模型** —— 要固定条数就在 search_kwargs 里写死 k。

print()
print("=" * 78)
print("第 2 部分(续):use_original_query —— 到底拿哪句话去检索")
print("=" * 78)

print(f"\n  查询: {Q}")
print("\n  【默认 False:用模型改写后的查询词】")
for d in build(enable_limit=True).invoke(Q):
    print(f"      {d.page_content[:22]}")

print("\n  【use_original_query=True:用用户原话】")
for d in build(enable_limit=True, use_original_query=True).invoke(Q):
    print(f"      {d.page_content[:22]}")

# 实测(2026-09-17):两者返回的是同样 3 本,但**排序不同**。
#
# ★ 为什么排序会变?因为送进向量检索的"查询词"换了一个:
#     默认     -> 用模型改写后的 "Python"(干净、短、没有"推荐三本"这些废词)
#     开了之后 -> 用用户原话 "推荐三本Python书"(带上了"推荐三本"这些噪声)
#   同一个向量库,查询词不同,相似度排序自然不同。
#
# ★ 该怎么选?
#     默认(用改写后的)  -> 一般在"用户说话啰嗦、口语化"时更好,查询词更聚焦
#     use_original_query -> 用在"改写之后反而丢了关键信息"的情况。
#       典型场景:你用的是**语义检索**,而模型把一句话缩成了一个词,
#       把原本有用的上下文(比如"和上次那本类似的")也一起丢了。
#   没有普适答案 —— 这是个需要拿你的真实数据试的参数。


# ============================================================================
# 第 3 部分:衍生 —— 自查询的四种用法
# ============================================================================
# SelfQueryRetriever 只是"查询构建"这套能力**最方便的封装**。拆开看,它是两块东西:
#
#     ① 查询构建(query construction): 自然语言 -> StructuredQuery
#     ② 翻译 + 检索(translator + vectorstore): 把 StructuredQuery 用起来
#
# 这两块可以**分开用**,于是就衍生出下面这些用法:

# ---- 用法 A:只用 ①,自己做过滤(最实用) ----
# 不一定要用 SelfQueryRetriever。把查询构建那条链单独拿出来,
# 你自己看着 StructuredQuery 决定怎么办 —— 这是生产中很常见的做法。
#
# 好处是你能把条件翻译成**任何东西**:
#   - 翻成 SQL 的 WHERE 子句,去关系数据库查
#   - 翻成 Elasticsearch 的 DSL,去搜索引擎查
#   - 翻成你自己写的过滤逻辑(比如查字典、查权限)
#
# 因为 translator 的活只是"把条件树变成你那个库的方言",而方言你来定。
#
# ① 那条链长这样(就是上一节填空 2 填的那个):
query_constructor = (
    get_query_constructor_prompt(document_contents, metadata_field_info)
    | llm
    | StructuredQueryOutputParser.from_components()
)
#
# 它单独就能用,返回的就是 StructuredQuery 对象:
structured = query_constructor.invoke({"query": "2019年以后出版的进阶书"})
print()
print("=" * 78)
print("第 3 部分:衍生用法 A —— 只用查询构建,自己决定怎么用")
print("=" * 78)
print(f"\n  拆出: {structured!r}")
print(f"  可以拿去做什么:")
print(f"      query  = {structured.query!r}   -> 语义部分,交给向量检索")
print(f"      filter = {structured.filter!r}")
print(f"      -> 这个 filter 你可以自己遍历,翻译成 SQL WHERE / ES DSL / 你自己的字典查询")

# ---- 用法 B:把条件**回显给用户确认** ----
# 因为 filter 是结构化对象,你可以把它渲染成人话再给用户看:
#     "我理解你要找的是:难度=进阶,且 年份 > 2019,对吗?"
# 这是 RAG 系统里非常实用的一招 ——
# 过滤条件是**硬条件**,一旦理解错了就整批为空(上一节那个"2020年以后"的坑)。
# 让用户在执行前确认一次,比查空了再解释便宜得多。

# ---- 用法 C:和逻辑路由(46章)串起来 ----
# 两者管的是不同的事,而且**互补**:
#     逻辑路由(46章):决定去**哪个库**查     —— 解决"范围"问题
#     自查询(48章):  决定在库里**加什么条件** —— 解决"精度"问题
# 串起来就是一个很真实的多库检索系统:
#     用户问题 -> 逻辑路由选库 -> 选中的库里跑自查询(带条件) -> 生成答案
# 两层都是"用模型先加工一下查询再检索",但一个管横向(选哪个数据源),
# 一个管纵向(在数据源内部怎么筛)。

# ---- 用法 D:多条件 + 嵌套逻辑 ----
# filter 本身是棵树,支持 AND / OR 嵌套,比如:
#     (level == '入门' OR level == '进阶') AND price < 100
# 这是向量检索**根本做不到**的事 —— 向量只有一个"像不像"的连续分数,
# 表达不了"或者"。这也是自查询最硬的价值。

# ============================================================================
# 第 4 部分:它和前面几章那些检索器是什么关系
# ============================================================================
# 40 章 MultiQuery、43 章 Step-Back、48 章 SelfQuery —— 这三个是**同一个家族**的,
# 它们都属于"**查询转换(Query Transformation)**":检索之前,先让模型加工一下查询。
#
#   检索器                     输入 -> 加工成什么                     主要提升
#   -------------------------------------------------------------------------
#   MultiQueryRetriever (40)   1 个问题 -> N 个不同角度的问法          召回率(漏的少了)
#   StepBackRetriever    (43)  1 个问题 -> 1 个更抽象的上位问题        召回率(太具体搜不到)
#   SelfQueryRetriever   (48)  1 个问题 -> 1 个查询词 + 1 组过滤条件   精度(条件精确)
#   EnsembleRetriever    (45)  不是转换,是**多路检索后融合**,管排序
#
# ★ 一句话区分:
#     40/43 都在想办法"多找点、别漏"(放宽)
#     48 在想办法"筛掉不该出现的"(收紧)
#   你手里同时有这两类工具,该用哪个取决于你的痛点是"漏了"还是"混进来了"。
#
# ★ 共同代价:全都要**多花一次 LLM 调用**。所以它们都不是"免费的改进",
#   而是"用延迟和钱换准确率"。查询改写类的东西,最好先确认你的痛点真的在那里。

# ============================================================================
# 第 5 部分:踩坑清单(每条都是这一课真跑出来的)
# ============================================================================
#
#  1. **字段缺失会直接崩,不是跳过。**
#     translator 里写的是 m.get(attr),字段不存在时返回 None,
#     而 None > 2020 会抛 TypeError。实测:
#         TypeError: '<' not supported between instances of 'NoneType' and 'int'
#     ★ 这意味着你的库里**不能有一部分文档缺字段**。要么保证字段齐全,
#       要么在 translator 里先判 `if attr not in m: return False`。
#
#  2. **"以后/以前/左右"这类词有歧义,差一个等号结果全空。**
#     "2020年以后" 会被拆成 year > 2020(不含 2020),库里那本 2020 年的书就被排除了。
#     说"2020年及以后"才会变成 >=。而且是**静默返回空**,不报错。
#
#  3. **过滤只筛不缩,检索器照样把 k 条填满。**
#     问"价格低于100元的 Python 书",返回里混进了 JS 和 Go 的书 ——
#     因为价格条件只筛掉了贵的,剩下的全合格,向量只负责排序,不负责"够不够格"。
#     这和 46 章"检索器永远不会说我不知道"是同一个坑。
#
#  4. **不传 translator 报的错和你的向量库毫无关系。**
#     实测报的是 `ImportError: cannot import name 'DatabricksVectorSearch'` ——
#     因为它在尝试 import 一堆**别的**向量库的 translator(FAISS 不在内置表里),
#     而那个包你没装,于是先炸在别处。找不到原因时,记住这一条。
#
#  5. **元数据没有可比较字段的话,自查询纯属白搭。**
#     如果你的 metadata 只有 {"source": "xxx.pdf"},那没有任何东西能当过滤条件,
#     白多花一次 LLM 调用。用之前先问自己:**我的元数据里有能被 > < 比较的字段吗?**
#
# ============================================================================
# 小结
# ============================================================================
#   运行逻辑一句话:拆解(调模型) -> 翻译(纯 Python) -> 合并参数 -> 普通向量检索。
#   只有第一步是 AI,后三步都是普通代码 —— 出问题先查第一步拆得对不对。
#
#   两个参数:
#     enable_limit=True       让模型能输出 limit(会覆盖 k),但不可靠,条数别交给它
#     use_original_query=True 用用户原话检索而不是改写后的,影响排序,要实测
#
#   三种衍生用法:单独用查询构建 / 把条件回显给用户确认 / 和逻辑路由串起来选库后再筛。
#
#   家族关系:40 MultiQuery、43 Step-Back 在"放宽多找",48 SelfQuery 在"收紧筛掉";
#            45 Ensemble 是另一类(多路融合)。全都要多花一次 LLM 调用。
