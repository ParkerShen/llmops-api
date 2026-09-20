'''
Date: 2026-09-17
Author: parker
FilePath: \llmops-api\study\48-自查询检索器实现动态元数据过滤\1.查询构建与自查询检索器.py
Description: 自查询检索器(SelfQueryRetriever) —— 让模型把"2020年以后100元以内的Python书"拆成"语义查询 + 元数据过滤条件"

运行方式(在项目根目录):
    python "study/48-自查询检索器实现动态元数据过滤/1.查询构建与自查询检索器.py"

依赖:需要 lark 包(2026-09-17 已装,1.3.1)。没装的话:
    pip install lark -i https://mirrors.cloud.tencent.com/pypi/simple
    (解析器要用它把模型吐出的一行字解析成结构化对象,没有它 StructuredQueryOutputParser 直接报错)

★ 本文件有 4 个空要你填,都标了 【填空 N】。
'''

# ============================================================================
# 第 0 部分:自查询要解决的问题
# ============================================================================
# 前面的检索器只会干一件事:**猜哪条文档意思最像**。它不懂"条件"。
#
# 用户说:"2020年以后出版的价格低于100元的 Python 书"
#   向量检索怎么处理?它把整句话编码成一个向量,然后找最像的文档。
#   "2020年以后""低于100元"这些**精确条件**,在它眼里只是语义的一部分 ——
#   它会返回一堆"Python 书",年份价格全看运气,可能给你一本 2013 年的 108 元的。
#   更糟的是它**看起来很对**(确实是 Python 书),你不会发现年份错了。
#
# 而元数据过滤能精确做到 year > 2020 且 price < 100 —— 但要求你**用程序写出条件**。
# 用户不会说 `{"year": {"$gt": 2020}}`,他说的是"2020年以后"。
#
# ★ 自查询(Self-Query)就是把这两半接起来:
#
#       用户的一句话
#            |
#            v
#       大模型:拆成两半
#            |
#            +---> 语义部分(query)  : "Python"          -> 交给向量检索,负责"像不像"
#            +---> 过滤条件(filter) : year>2020 且 price<100 -> 交给元数据过滤,负责"符不符合"
#
#   两半各干各的擅长的事。这不是"更聪明的检索",是**把一件事拆成两件**。

import dotenv

# AttributeInfo:描述"我这个库里有哪些字段可以当过滤条件"
# StructuredQueryOutputParser:把模型吐出的那一行文本解析成 StructuredQuery 对象(靠 lark)
# get_query_constructor_prompt:内置的提示词,它会把你的字段说明拼成"请拆解"的指令
from langchain_classic.chains.query_constructor.base import (
    AttributeInfo,
    StructuredQueryOutputParser,
    get_query_constructor_prompt,
)
from langchain_classic.retrievers.self_query.base import SelfQueryRetriever

# Visitor:langchain 定义的"访问者"基类,我们要继承它写自己的翻译器
from langchain_core.structured_query import (
    Comparator,
    Comparison,
    Operation,
    Operator,
    StructuredQuery,
    Visitor,
)
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_deepseek import ChatDeepSeek
from langchain_huggingface import HuggingFaceEmbeddings

# ---------------------------------------------------------------------------
# 【名词表】本文件新导入的名字都是干嘛的(完整版见同目录 2.xxx.py 开头)
# ---------------------------------------------------------------------------
#   先记住三个包的分工,import 路径会自己说话:
#     langchain_core      -> 底层抽象:数据结构、基类、Runnable 协议
#     langchain_classic   -> "经典"实现:检索器、链(原 0.x 版 langchain 包搬来的)
#     langchain_community -> 第三方集成:各种向量库、加载器
#
#   AttributeInfo                类。一张"字段说明书":name(字段名) +
#                                description(给模型看的说明) + type(类型)。
#                                它是 46 章 RouteQuery 的"表亲":RouteQuery 描述一个字段,
#                                它描述"一张表里的一列"。用在哪:第 2 部分。
#
#   get_query_constructor_prompt 函数。给它(书库描述 + 字段说明书),它返回一个
#                                提示词模板 —— 里面 langchain 已经写好了
#                                "请把用户问题拆成 query 和 filter"这套指令和示例。
#                                相当于第 1 章的 PromptTemplate,只是内容不用自己编。
#                                用在哪:第 3 部分。
#
#   StructuredQueryOutputParser 类。把模型吐出的一行文本解析成 StructuredQuery 对象。
#                                和你在第 3 章学的 StrOutputParser / JsonOutputParser
#                                是同一类东西(都是 OutputParser),只是产物变成结构化查询。
#                                ⚠️ 它靠 lark 包干活。用在哪:第 3 部分链条末环。
#
#   SelfQueryRetriever           类。检索器本体,把上面三样 + 向量库 + 翻译器组装起来。
#                                和 38 章那些检索器是同类:都继承 BaseRetriever,
#                                都能 .invoke()、都能塞进 LCEL。用在哪:第 5 部分。
#
#   ---- 下面这些来自 langchain_core.structured_query,是"拆解结果的零件",不是给你调用的工具:
#
#   StructuredQuery   数据结构。拆解的产物,三个字段:query(语义) / filter(条件树) / limit
#   Comparison        数据结构。一条比较条件 = 属性 + 比较符 + 值,是条件树的**叶子**
#   Operation         数据结构。把子条件用 AND/OR 连起来,是条件树的**树枝**
#   Comparator        枚举。比较符 EQ/GT/LT/GTE/LTE/NE —— 就是 = > < >= <= ≠ 的对象形式
#   Operator          枚举。逻辑符 AND/OR/NOT
#   Visitor           基类。框架约定:你按它规定的 visit_comparison / visit_operation
#                     等**方法名**填内容,遇到对应节点时它就会回调你写的方法。
#                     ★ 记住这个套路:凡是继承 BaseXxx / Visitor 的类,你只管按
#                       规定的方法名填内容,调用时机由框架负责。
# ---------------------------------------------------------------------------

dotenv.load_dotenv()

embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
llm = ChatDeepSeek(model="deepseek-chat", temperature=0)


# ============================================================================
# 第 1 部分:一个有"结构化元数据"的书库
# ============================================================================
# 自查询能不能用,**全看你的元数据够不够"结构化"**。
# 如果元数据只有 {"source": "某本书.pdf"},那自查询没东西可过滤,白搭。
# 下面这批书特意带上了 year / price / level 这种**能比较**的字段。

# (简介, 作者, 年份, 价格, 难度)
BOOKS = [
    ("流畅的Python:深入理解 Python 语言特性与惯用法", "Luciano Ramalho", 2017, 139, "进阶"),
    ("Python编程:从入门到实践,零基础学 Python", "Eric Matthes", 2020, 89, "入门"),
    ("Python Cookbook:Python 实用技巧与最佳实践", "David Beazley", 2013, 108, "进阶"),
    ("深入理解计算机系统:从程序员视角讲计算机原理", "Randal Bryant", 2016, 139, "进阶"),
    ("JavaScript高级程序设计:JS 语言完整参考", "Matt Frisbie", 2020, 129, "进阶"),
    ("你不知道的JavaScript:JS 语言核心机制", "Kyle Simpson", 2015, 79, "进阶"),
    ("Go语言圣经:Go 语言权威指南", "Alan Donovan", 2016, 79, "进阶"),
    ("Python数据科学手册:用 Python 做数据分析", "Jake VanderPlas", 2018, 109, "入门"),
    ("算法导论:计算机算法经典教材", "Thomas Cormen", 2013, 128, "专家"),
    ("重构:改善既有代码的设计", "Martin Fowler", 2019, 118, "进阶"),
]

# 简介进 page_content(拿去算向量),其余进 metadata(拿去过滤)。
# ★ 这个分界就是自查询的前提:**能算相似度的放 content,能比较的放 metadata。**
docs = [
    Document(
        page_content=text,
        metadata={"author": author, "year": year, "price": price, "level": level},
    )
    for text, author, year, price, level in BOOKS
]

db = FAISS.from_documents(docs, embedding)

# 一句话描述这个库装的是什么。它会拼进提示词,帮模型判断"该往哪个方向理解"
document_contents = "计算机技术书籍的简介"


# ============================================================================
# 第 2 部分:告诉模型"有哪些字段可以过滤"
# ============================================================================
# 这就是自查询版的 schema —— 和 46 章那个 RouteQuery 是一回事:
# 你定义形状,模型往里填。区别是 46 章只有一个字段,这里要描述"一张表有哪些列"。

metadata_field_info = [
    # 【填空 1】照着下面三条的样子,把 author 这一条补完整。
    #   三个参数分别是什么:
    #     name        = 元数据里的**字段名**。⚠️ 必须和上面 Document metadata 的 key
    #                   **一字不差**,写错了模型会生成一个不存在的字段,过滤结果直接空。
    #     description = 写给**模型**看的说明(和 46 章 Field(description=...) 一个道理)。
    #                   模型靠它判断"用户说作者的时候,对应的是哪个字段"。
    #     type        = 字段类型。文本用 "string";整数用 "integer";小数 "float"。
    #                   ★ 这个 type 很关键:模型会据此决定**能不能用 > < 比较** ——
    #                     声明成 "string" 的话,"2020年以后"这类条件它就表达不出来了。
    # ✅ 填空 1 答案(2026-09-17 填):
    #   name 必须和 Document metadata 的 key 一字不差,这里是 "author"
    #   description 是写给模型看的,要说清"用户怎么说的时候该选我"
    #   type 是 "string" —— 人名的比较只有"等于/不等于",没有大小,所以是字符串
    #   ★ 反例:要是把 year 写成了 "string",那"2020年以后"这种条件
    #     模型就没法表达成 year > 2020 了(字符串比较大小毫无意义),只能退化成等于。
    AttributeInfo(
        name="author",                       # 元数据里的字段名
        description="作者姓名",               # 给模型看的说明
        type="string",                       # 文本类型
    ),
    AttributeInfo(name="year", description="出版年份", type="integer"),
    AttributeInfo(name="price", description="价格,单位元", type="integer"),
    AttributeInfo(name="level", description="难度等级:入门/进阶/专家", type="string"),
]


# ============================================================================
# 第 3 部分:查询构建 —— 先不看检索,只把"拆解"这一步看清楚
# ============================================================================
# get_query_constructor_prompt 是 langchain 内置的提示词,它把 document_contents
# 和 metadata_field_info 拼成一段"请你把用户的问题拆成 query 和 filter"的指令。
prompt = get_query_constructor_prompt(document_contents, metadata_field_info)

# 【填空 2】把三样拼成一条链:提示词 -> 模型 -> 解析器。
#   三样都已经有了:上面的 prompt、上面的 llm、下面这个解析器。
#   用 | 连起来,顺序照着 "先给提示词,再问模型,最后解析它吐的字" 排。
output_parser = StructuredQueryOutputParser.from_components()

# ✅ 填空 2 答案(2026-09-17 填):就是第 3 章学过的 LCEL 管道,顺序不能颠倒 ——
#   先给提示词(把用户问题 + 字段说明拼成完整指令),
#   再问模型(它吐出一行字),
#   最后解析(把那一行字变成 StructuredQuery 对象,这一步靠 lark)。
#   颠倒任何一处都会报错:比如把解析器放前面,它收到的是 PromptValue,不是字符串。
query_constructor = prompt | llm | output_parser

print("=" * 78)
print("第 3 部分:只看拆解 —— 一句话变成了什么")
print("=" * 78)

PROBES = [
    "2020年以后出版的价格低于100元的Python书",
    "Martin Fowler 写的那本关于重构的书",
    "适合入门的书有哪些",
    "只想看 JavaScript 相关的书",
]

for q in PROBES:
    structured = query_constructor.invoke({"query": q})
    print(f"\n  原话: {q}")
    print(f"  拆出: {structured!r}")

# 实测输出(2026-09-17 实跑,你填完跑出来应该一样):
#
#   "2020年以后出版的价格低于100元的Python书"
#     -> query='Python'
#        filter=Operation(AND, [year > 2020, price < 100])
#
#   "Martin Fowler 写的那本关于重构的书"
#     -> query='重构'
#        filter=author == 'Martin Fowler'          <- 人名的"作者"语义被认出来了
#
#   "适合入门的书有哪些"
#     -> query=' '                                <- ★ 语义部分是空的!
#        filter=level == '入门'
#
#   "只想看 JavaScript 相关的书"
#     -> query='JavaScript'
#        filter=None                              <- 没有条件,退化成普通向量检索
#
# ★ 看第 3 和第 4 个:自查询的两半是**可以各自为空**的。
#   纯条件问题 -> query 空,全靠 filter;纯语义问题 -> filter 空,退回普通检索。
#   这说明它不是"另一种检索",而是**在普通检索外面加了一层可选的条件约束**。


# ============================================================================
# 第 4 部分:翻译器 —— 把 StructuredQuery 变成 FAISS 听得懂的东西
# ============================================================================
# 模型拆出来的 StructuredQuery 是一棵"条件树"(可能套着 AND/OR),但 FAISS 不认这棵树。
# FAISS 只认一种东西:**一个函数**,收 metadata 字典,返回 True/False。
# 中间这个"把树翻译成函数"的角色,就是 translator(翻译器)。
#
# ⚠️ 你环境里**没有现成的 FAISS translator** —— LangChain 1.x 把这个模块删了
#    (老版本在 langchain_community.query_constructors.faiss,现在整个目录里都搜不到)。
#    所以下面手写一个。手写反而好:写一遍你就知道这层没任何魔法,
#    而且换任何向量库,你要写的都是**同一个东西的方言版**。


class MiniFAISSTranslator(Visitor):
    """把 StructuredQuery 翻译成 FAISS 的 filter 函数。

    继承 Visitor 是 langchain 约定的写法:每碰到一种节点类型,就调对应的 visit_xxx。
    """

    def visit_structured_query(self, query: StructuredQuery):
        """入口。返回 (给向量检索用的查询词, 额外的搜索参数)。"""

        # 没有过滤条件 -> 什么额外参数都不传,退化成普通向量检索
        if query.filter is None:
            return query.query, {}

        # 有过滤条件 -> 翻译成一个函数,塞进 filter 参数。
        # FAISS 的 similarity_search(..., filter=fn) 会对每条候选文档调 fn(doc.metadata),
        # 返回 False 的就丢掉。
        return query.query, {"filter": self._to_filter(query.filter)}

    def _to_filter(self, node):
        """按节点类型分发。langchain 的 Visitor 基类没有通用 dispatch,所以自己写。"""
        if isinstance(node, Comparison):
            return self.visit_comparison(node)
        if isinstance(node, Operation):
            return self.visit_operation(node)
        raise ValueError(f"不支持的条件类型: {type(node).__name__}")

    def visit_comparison(self, comparison: Comparison):
        """把一条比较条件,翻译成一个"收 metadata、返回 True/False"的小函数。"""
        attr, val, op = comparison.attribute, comparison.value, comparison.comparator

        # 【填空 3】把剩下的比较符补齐。
        #   左边是 langchain 定义的符号,右边是**你要交给 Python 的表达式**。
        #   注意:lambda 的 m 就是一条文档的 metadata 字典,取值一律用 m.get(attr)。
        #   已经写好的 EQ 就是范例 —— 对照它,把 > < >= <= 四个补齐。
        # ✅ 填空 3 答案(2026-09-17 填):就是 Python 那五个比较运算符,一一对应。
        ops = {
            Comparator.EQ: lambda m: m.get(attr) == val,      # 等于
            Comparator.GT: lambda m: m.get(attr) > val,       # 大于
            Comparator.LT: lambda m: m.get(attr) < val,       # 小于
            Comparator.GTE: lambda m: m.get(attr) >= val,     # 大于等于
            Comparator.LTE: lambda m: m.get(attr) <= val,     # 小于等于
        }
        # ⚠️ 一个容易忽略的点:这里用的是 m.get(attr) —— 字段不存在时返回 None。
        #    None > 2020 会**直接抛 TypeError**,不是返回 False。
        #    所以如果某条文档缺了这个字段,整个检索会崩,而不是"跳过这条"。
        #    (真实项目里要么保证字段齐全,要么在这里先判 `if attr not in m: return False`
        #     或者用 `m.get(attr, 0)` 给个默认值 —— 这是个常见的线上事故点。)
        if op not in ops:
            raise ValueError(f"不支持的比较符: {op}")
        return ops[op]

    def visit_operation(self, operation: Operation):
        """逻辑组合:AND / OR。"""
        args = [self._to_filter(a) for a in operation.arguments]

        # 【填空 4】args 是一堆"收 metadata 返回 True/False"的函数,现在要把它们合起来。
        #   AND:每一个都成立才算通过 -> 用哪个内置函数?
        #   OR :任意一个成立就通过   -> 又是哪个?
        #   提示:两个都是**接收一个可迭代对象、返回 bool** 的内置函数,名字就是英文意思。
        #   写法是 lambda m: <那个函数>(f(m) for f in args)
        # ✅ 填空 4 答案(2026-09-17 填):all 和 any,两个 Python 内置函数。
        #   all(生成器) -> 里面全为真才返回 True(AND 的语义)
        #   any(生成器) -> 有一个为真就返回 True(OR 的语义)
        #   写成 lambda 是必须的:我们不能立刻算结果 —— 此刻还不知道要看哪条文档。
        #   lambda 把这个"组合方式"打包成一个函数,等 FAISS 拿着具体某条文档的
        #   metadata 来调它时,才真正开始算。这就是**闭包**——和 47 章里
        #   route_examples 那个循环外面的变量是同一类思路:先记下"怎么算",等调用时再算。
        if operation.operator == Operator.AND:
            return lambda m: all(f(m) for f in args)
        if operation.operator == Operator.OR:
            return lambda m: any(f(m) for f in args)
        raise ValueError(f"不支持的逻辑符: {operation.operator}")

    def visit_not(self, not_) -> None:
        # 本课不支持 NOT,留个明确的报错,比默默返回错误结果好
        raise NotImplementedError("本课的迷你翻译器不支持 NOT")


# ============================================================================
# 第 5 部分:组装自查询检索器,跑起来
# ============================================================================
# from_llm 的五个参数,和你手写一条链需要的东西一一对应:
#   llm                          -> 负责"拆解"的模型
#   vectorstore                  -> 拆完之后去哪儿查
#   document_contents            -> 拼进提示词:这个库装的是什么
#   metadata_field_info          -> 拼进提示词:有哪些字段能过滤
#   structured_query_translator  -> 上面那个翻译器:把条件树变成库能懂的过滤

retriever = SelfQueryRetriever.from_llm(
    llm=llm,
    vectorstore=db,
    document_contents=document_contents,
    metadata_field_info=metadata_field_info,
    structured_query_translator=MiniFAISSTranslator(),
)

print()
print("=" * 78)
print("第 5 部分:自查询检索器实际查到什么")
print("=" * 78)


def show(q):
    print(f"\n  问: {q}")
    hits = retriever.invoke(q)
    if not hits:
        print("      (空)     <- ★ 注意:过滤条件一旦写死,很容易整批空掉")
    for d in hits:
        m = d.metadata
        print(f"      {m['year']}年 {m['price']:>3}元 {m['level']:<3} | {d.page_content[:24]}")


for q in [
    "2020年以后出版的价格低于100元的Python书",   # ★ 这个会是空的,原因见下面注释
    "2020年及以后出版的价格低于100元的Python书",  # 加一个"及"字就命中
    "价格低于100元的Python书",
    "适合入门的书有哪些",
    "2019年以后出版的进阶书",
]:
    show(q)

# 实测输出(2026-09-17 实跑):
#
#   "2020年以后出版的价格低于100元的Python书"
#       (空)                                  <- ★★ 见下面那个坑
#
#   "2020年及以后出版的价格低于100元的Python书"
#       2020年  89元 入门 | Python编程:从入门到实践
#
#   "价格低于100元的Python书"
#       2020年  89元 入门 | Python编程:从入门到实践
#       2015年  79元 进阶 | 你不知道的JavaScript     <- 不是 Python 书!
#       2016年  79元 进阶 | Go语言圣经               <- 不是 Python 书!
#
#   "适合入门的书有哪些"
#       2020年  89元 入门 | Python编程:从入门到实践
#       2018年 109元 入门 | Python数据科学手册
#
#   "2019年以后出版的进阶书"
#       2020年 129元 进阶 | JavaScript高级程序设计
#
#
# ===== ★ 坑一:"以后"到底是 > 还是 >=? =====
#
# 第一个查询返回空,不是 bug,是**过滤条件太硬**。
# 模型把"2020年以后"理解成了 year > 2020(不含 2020),而库里那本 Python 书
# 恰好是 2020 年的 —— 被 > 挡在了外面。加个"及"字,模型就换成 >= ,立刻命中。
#
# 这里有两个教训:
#   1. **自然语言的模糊 + 精确过滤 = 差一个等号就全军覆没。**
#      向量检索时候差点还能"意思意思"给几条,过滤是**硬筛**,一条不留。
#   2. **失败是静默的**:它返回空列表,不报错、不警告。
#      真实系统里必须专门处理"过滤后为空"这种情况(提示用户放宽条件),
#      否则用户只会看到"没找到",甚至更糟 —— 空结果被塞给大模型,它开始编。
#
# ===== ★ 坑二:过滤条件不会缩小"返回条数",它只是**筛** =====
#
# 第三个查询问的是"Python 书",却返回了 JavaScript 和 Go 的书。
# 为什么?拆出来的是 query='Python' + filter=(price<100)。
#   价格条件把 >=100 的全筛掉了,剩下 3 本(89/79/79)全都满足价格,
#   于是这 3 本**全被返回**,而向量检索只负责给它们**排序**(按像 Python 的程度)。
#
# 也就是说:**filter 管"要不要",向量管"像不像",但它俩都不管"够不够格"。**
# 检索器永远会把 k 条填满(默认 k=4)—— 这跟 46 章那个
# "检索器永远不会说我不知道"是**同一个坑**,只是这次伪装成了"过滤成功"的样子。
#
# ===== 什么时候该用自查询 =====
#   你的文档有**真正结构化、能用比较运算符**的字段(年份/价格/状态/版本号)
#                                                          -> 用,收益很大
#   元数据只有 source / page 这种(几乎所有 RAG 都是这样)      -> 别用,
#                                                          没有可过滤的字段,白搭一层
#   用户几乎不提条件,都是"帮我讲讲 X"                        -> 别用,
#                                                          徒增一次 LLM 调用
