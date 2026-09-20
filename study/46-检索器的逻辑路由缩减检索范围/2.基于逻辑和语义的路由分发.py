'''
Date: 2026-09-15 14:31:03
Author: parker
FilePath: \llmops-api\study\46-检索器的逻辑路由缩减检索范围\2.基于逻辑和语义的路由分发.py
Description: 
'''
import dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_deepseek import ChatDeepSeek
from pydantic import BaseModel, Field
dotenv.load_dotenv()
from typing import Literal
# 初始化本地 Embedding 模型
embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
class RouteQuery(BaseModel):
    """把用户问题分诊到某一个知识库分区。"""

    # 只有一个字段:datasource(数据源)。
    #   Literal[...]      -> 值只能是这三个之一,模型想返回 "c++" 都返回不了
    #   Field(description=...) -> 这段中文会**真的发**给模型,
    #                              它靠这句话理解"我该在什么情况下选哪个"
    #   ⚠️ 忘了写 description 也能跑,但模型选错的概率会明显上升 ——
    #      它只能靠字段名 datasource 猜你的意图。
    datasource: Literal["python", "js", "other"] = Field(
        description="用户问题最相关的技术栈分区,只能三选一"
    )
# 原始数据
KB_PY = [
    ("装饰器", "Python 装饰器是一个接收函数并返回新函数的可调用对象,用 @ 语法糖贴在函数定义上方。"),
    ("GIL", "CPython 的全局解释器锁(GIL)让同一时刻只有一个线程执行字节码,所以多线程跑不满多核,CPU 密集任务要用 multiprocessing。"),
    ("虚拟环境", "用 python -m venv .venv 创建独立环境,激活后 pip install 的包只装在这个环境里,避免不同项目互相污染。"),
    ("列表推导式", "列表推导式 [x*2 for x in nums if x > 0] 把循环和过滤压缩成一行,比 for + append 更快也更易读。"),
]

KB_JS = [
    ("事件循环", "JavaScript 是单线程的,靠事件循环(event loop)把宏任务和微任务排队执行,所以 setTimeout 的回调不会打断当前同步代码。"),
    ("Promise", "Promise 表示一个未来才有结果的异步操作,用 .then() 或 async/await 取结果,避免回调地狱。"),
    ("闭包", "闭包指函数记住了它定义时所在的作用域,即使外层函数已经返回,内部变量依然活着。"),
    ("模块化", "ES Module 用 import/export 组织代码,node 里也可以用 CommonJS 的 require/module.exports。"),
]

llm = ChatDeepSeek(model="deepseek-chat", temperature=0)

class RouteQuery(BaseModel):
    """把用户问题分诊到某一个知识库分区。"""

    # 只有一个字段:datasource(数据源)。
    #   Literal[...]      -> 值只能是这三个之一,模型想返回 "c++" 都返回不了
    #   Field(description=...) -> 这段中文会**真的发**给模型,
    #                              它靠这句话理解"我该在什么情况下选哪个"
    #   ⚠️ 忘了写 description 也能跑,但模型选错的概率会明显上升 ——
    #      它只能靠字段名 datasource 猜你的意图。
    datasource: Literal["python", "js", "other"] = Field(
        description=(
            "用户问题最相关的技术栈分区。"
            "判断依据是**问题的答案会落在哪个生态**,而不是问题里出现了哪个词:"
            "即使问题里没写 Python,只要答案只可能来自 Python 生态"
            "(如 GIL、多线程限制、装饰器、虚拟环境、列表推导式),就选 python;"
            "即使问题里没写 JS,只要答案只可能来自 JavaScript/Node 生态"
            "(如事件循环、Promise、闭包、回调地狱、模块化),就选 js。"
            "如果问题跟这两个生态都无关(闲聊、天气、其他语言、泛泛而谈),"
            "或者根本看不出在问什么,一律选 other —— 选 other 不算答错,"
            "硬选一个才是错的。"
        )
    )


# 1. 转换为 Document 列表，metadata 使用 topic 字段
docs_py = [Document(page_content=desc, metadata={"topic": name}) for name, desc in KB_PY]
docs_js = [Document(page_content=desc, metadata={"topic": name}) for name, desc in KB_JS]

# 2. 分别构建两个 FAISS 向量库
db_py = FAISS.from_documents(docs_py, embedding)
db_js = FAISS.from_documents(docs_js, embedding)

# 3. 创建两个检索器，k=2
retriever_py = db_py.as_retriever(search_kwargs={"k": 2})
retriever_js = db_js.as_retriever(search_kwargs={"k": 2})

# 4. 测试检索并打印结果，用 metadata 里的 topic 来识别
query = "什么是闭包？"

print(f"=== 查询: {query} ===\n")

print("--- Python 知识库检索结果 ---")
for doc in retriever_py.invoke(query):
    print(f"[{doc.metadata['topic']}] {doc.page_content}")

print("\n--- JavaScript 知识库检索结果 ---")
for doc in retriever_js.invoke(query):
    print(f"[{doc.metadata['topic']}] {doc.page_content}")

structured_llm = llm.with_structured_output(RouteQuery)

QUESTIONS = [
    "装饰器怎么用",              # 期望 python
    "回调地狱是什么",            # ★ 注意：一个 JS/JavaScript 字样都没有 —— 期待它靠语义选 js
    "为什么多线程跑不满多核",     # 期望 python（GIL）
    "今天天气怎么样",            # 期望 other（跟两个库都无关）
    "Python 和 JS 哪个好",       # ★ 两个都沾边,看模型怎么判 —— 见第 3 部分的讨论
]


# ============================================================================
# 第 2 部分:逻辑路由 —— 让大模型先分诊,再去对应的库查
# ============================================================================
# 整个路由就干两件事:① 问模型"这题归哪个库" ② 按答案去查。
# 注意第 ② 步没有任何魔法,就是个普通的 if —— 路由的"智能"全在 ① 里,
# 而 ① 之所以可靠,靠的是上一节学的 with_structured_output 把答案锁成枚举。


def route(query: str):
    """先分诊,再只查被选中的那一个库,返回 Document 列表。"""

    decision = structured_llm.invoke(query)   # 一次 LLM 调用,只为拿一个标签
    print(f"  分诊结果: {decision.datasource}")

    # 这里写 if 之所以安全,是因为 decision.datasource 只可能是这三个字符串之一,
    # 不存在"模型说了句别的话"的情况 —— 这就是上一节把输出规范化的回报。
    if decision.datasource == "python":
        return retriever_py.invoke(query)

    if decision.datasource == "js":
        return retriever_js.invoke(query)

    # ★ 兜底分支。走到这里说明模型选了 other —— 它认为这个问题两个库都不该查。
    # 返回空列表,表示"没检索到东西"。
    # ⚠️ 但"空列表"和"告诉用户我不知道"是两回事:
    #    返回 [] 只是让调用方知道没结果,要不要给用户一句"这个问题不在服务范围"
    #    是**上层**的事。真实系统里千万别让空列表一路滑到大模型那里去生成答案 ——
    #    大模型接到空上下文,会**自己编**一个答案出来,这就是幻觉的经典来源。
    return []


print()
print("=" * 78)
print("第 2 部分:逻辑路由(LLM 分诊 + if 分发)")
print("=" * 78)

for q in QUESTIONS:
    print(f"\n查询: {q}")
    docs = route(q)
    for d in docs:
        print(f"    [{d.metadata['topic']}] {d.page_content}")

# 实测输出(2026-09-16 实跑):
#
#   查询: 装饰器怎么用
#     分诊结果: python
#       [装饰器] Python 装饰器是一个接收函数并返回新函数的可调用对象...
#       [虚拟环境] 用 python -m venv .venv 创建独立环境...
#
#   查询: 回调地狱是什么          <- ★ 一个 "JS" 字样都没有
#     分诊结果: js                <- 靠语义选对了
#       [Promise] Promise 表示一个未来才有结果的异步操作...避免回调地狱。
#       [事件循环] JavaScript 是单线程的,靠事件循环(event loop)...
#
#   查询: 为什么多线程跑不满多核
#     分诊结果: python
#       [GIL] CPython 的全局解释器锁(GIL)...多线程跑不满多核...
#       [列表推导式] ...        <- 第 2 条还是无关的,但至少有救了
#
#   查询: 今天天气怎么样
#     分诊结果: other
#       (没有检索结果)          <- 谁都没查,因为两个库都不该查
#
#   查询: Python 和 JS 哪个好
#     分诊结果: other
#       (没有检索结果)          <- 两个都沾边,反而谁都不选,见第 3 部分
#
# 对比第 1 部分那次"不做路由"的检索:同一个"什么是闭包",
# Python 库照样吐了两条毫不相干的文档回来。现在这两条垃圾**根本不会被检索**,
# 因为 Python 库压根没被调用 —— 这就是"缩减检索范围"的字面含义。
#
# 但注意:路由**只**解决了"进错库"的问题,不解决"进了对的库但库内排序不准"。
# 上面第 3 个查询的第 2 条 [列表推导式] 就是库里自带的噪声,那要靠重排(rerank)解决。


# ============================================================================
# 第 3 部分:★ 那个"出口"(other)—— 以及为什么必须显式写进 description
# ============================================================================
# 上面 RouteQuery 里的 other 不是装饰,是这个设计能用的前提。
# 把 other 去掉(只留 python/js)去问几个不该路由的问题,实测(2026-09-16):
#
#     '今天天气怎么样'         -> python     ← 问天气,答 Python
#     '帮我写一首关于春天的诗'  -> python
#     'Python 和 JS 哪个好'    -> python
#
# 全是瞎猜。原因有两层:
#
#   1. **Literal 是硬约束**。只有两个格子,模型必须填一个,
#      它没有"都不像"这个选项可用 —— 不是你提示词没写好,是选项本来就不存在。
#
#   2. **模型的默认倾向是"选一个最像的",而不是"承认不知道"**。
#      所以光加一个 other 选项还不够,description 里必须**明确授权**它选 other:
#          "如果跟这两个都无关,一律选 other —— 选 other 不算答错,硬选一个才是错的"
#      不写这句,它加了 other 也照样不用,还是硬选 python。
#      这是和模型打交道的一条通用经验:**你想让它"不做某事",得明确给它许可**。
#
# 加 other 之后的实测(2026-09-16):
#
#     '今天天气怎么样'         -> other    ✓ 现在它敢说"不归我管"了
#     '帮我写一首关于春天的诗'  -> other    ✓
#     'Python 和 JS 哪个好'    -> other    ← 见下面第 1 条
#
# ===== ★ 中间踩的一个真坑:改描述改出了"漏检" =====
#
# 上面那个"为什么多线程跑不满多核",加了 other 之后**反而判错了**:
# 它从 python 变成了 other。同一句话、同一个模型,只因为我改了
# description 的措辞 —— 当时写的是"只有问题**确实在问** Python 时才选 python",
# 而这句话里根本没有"Python"三个字(GIL 是 Python 特有的,但句子没说),
# 模型就严格按字面执行,判成 other。
#
# 改法是在 description 里补上第一句"判断依据是**答案会落在哪个生态**,
# 不是问题里出现了哪个词",并列了几个人们问的时候不会带语言名的典型词
# (GIL / 装饰器 / 虚拟环境 / 事件循环 / 回调地狱)。改完就对了。
#
# 这个坑的教训比 other 本身更重要:
#   **没有出口时,模型会硬选(错选);有了出口,它又会变得保守(漏检)。**
#   你放松哪一头,另一头就变差 —— 这是同一枚硬币的两面,没有两全的写法。
#   而且调这个平衡的工具只有一个:description 的**措辞**。
#   所以路由类的 prompt 必须拿真实问题反复跑,不能凭感觉写完就上线。
#
# ===== 还没解决的两个问题 =====
#
# 1) "Python 和 JS 哪个好" 这种**跨类**问题,路由只能二选一,必然丢掉一半信息。
#    真实系统里有两种做法:
#      - 把 datasource 改成 **list[Literal[...]]**(多选),两边的库都查一遍再合并
#      - 或者干脆**不做路由**,退化成"全库检索 + 重排"
#    也就是说:**路由不是越细越好**。它用"缩小范围"换准确率,
#    代价是"只此一路"——问题本身跨类时,这个代价就变成错误。
#
# 2) 路由**判错**了怎么办?模型说 python,问题其实是 js,那么检索回来的全是垃圾,
#    而且你会**毫无察觉** —— 最终答案会很自信地胡说。常见对策:
#      - 检索后加一道相关性校验(比如对比分数阈值),太低就走兜底
#      - 或者保留一个"全库检索"的降级路径
#      - 生产环境还应把 decision 记进日志,否则出了错你根本不知道是哪一步坏的
#
# ★ 一句话:逻辑路由的**收益**是缩小范围、减少噪声;
#   **成本**是多一次 LLM 调用、多一个判错的故障点、以及跨类问题上的信息损失。


# # ============================================================================
# # 第 4 部分:语义路由 —— 不调大模型,纯算向量比相似度
# # ============================================================================
# # 逻辑路由是"问模型:这题归哪个库"。语义路由换了个思路:
# #   **不给模型看,直接算**。你事先给每个意图写一句描述,把描述和问题
# #   都转成向量,谁跟问题最像就选谁 —— 一次 API 都不用调。
# #
# # 好处:快、免费、能离线、意图加到 100 个也不额外花钱。
# # 代价:它只会"比像不像",不懂因果。问题里的措辞一变,可能就认不出来了。
# #      (这正是我们马上要实测的 —— 拿那五个问题跑一遍,看它在哪翻车)

# import numpy as np   # 算余弦相似度用,你环境里已经装了

# # 每个意图配一句"人话描述"。这句描述就是语义路由的 prompt ——
# # 写得越贴合用户真实的提问方式,匹配得越准。
# ROUTES = {
#     "python": "Python 编程语言相关的问题,包括语法、标准库、包管理、并发模型",
#     "js": "JavaScript 和 Node.js 相关的问题,包括异步、事件循环、浏览器 API",
# }

# route_names = list(ROUTES.keys())     # ['python', 'js']
# route_texts = list(ROUTES.values())   # 上面那两句描述


# # ① 【填空】把这两句描述转成向量。
# #    提示:embedding 对象有两个方法 ——
# #         embed_documents([...])  传一个**列表**,返回**一批**向量
# #         embed_query("...")      传一个**字符串**,返回**一个**向量
# #    这里要转的是一批(两句),该用哪个?
# route_vectors = ____


# def cosine(a, b):
#     """余弦相似度:两个向量夹角的余弦。方向越一致越接近 1。"""
#     return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# def semantic_route(query: str):
#     """跟 route() 干一样的活:返回 (选中的意图名, 它的相似度得分)。"""

#     # ② 【填空】把这一句查询转成向量。
#     #    注意这次是**一个**字符串,不是列表。
#     q_vec = ____

#     # ③ 逐个意图算相似度,保留分数最高的那个
#     best_name, best_score = None, -1.0
#     for name, vec in zip(route_names, route_vectors):
#         score = cosine(q_vec, vec)
#         print(f"      {name:<8} 相似度 {score:.4f}")
#         # ④ 【填空】什么时候该把 best 换成当前的 name / score?
#         #    提示:我们要"最高分",所以拿 score 和现有的 best_score 比。
#         if ____:
#             best_name, best_score = name, score

#     return best_name, best_score


# print()
# print("=" * 78)
# print("第 4 部分:语义路由(不调大模型)")
# print("=" * 78)

# for q in QUESTIONS:
#     print(f"\n查询: {q}")
#     name, score = semantic_route(q)
#     print(f"    -> 选中: {name}  (得分 {score:.4f})")