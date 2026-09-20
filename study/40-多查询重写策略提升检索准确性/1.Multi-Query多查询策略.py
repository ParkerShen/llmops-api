'''
Date: 2026-09-15 17:19:44
Author: parker
FilePath: \llmops-api\study\40-多查询重写策略提升检索准确性\1.Multi-Query多查询策略.py
Description: Multi-Query 多查询策略 —— 用 LLM 把一个问题改写成多个查询,取检索结果的并集

运行方式(在项目根目录):
    python "study/40-多查询重写策略提升检索准确性/1.Multi-Query多查询策略.py"

⚠️ 本文件需要联网调用 DeepSeek(第 3、4、5 部分),会消耗少量 token。
'''

# ============================================================================
# 第 0 部分:单查询的天花板
# ============================================================================
#
# 前面几章用的都是"一个查询 → 一次向量检索"。它有个天生的局限:
#
#   用户那句话,经过 embedding 之后,只是向量空间里的**一个点**。
#   一次检索就是从那个点出发,划一个圈,把最近的 k 条捞回来。
#
# 这个圈只能覆盖**一个语义方向**。于是有两种翻车方式:
#
#   1. 用户的话和文档的话**用词不同** —— 用户说"五险一金",文档写"社会保险与住房公积金",
#      两个向量离得不近,检索不到。
#
#   2. 用户的问题**本身很宽** —— 比如"入职要准备什么",这里面其实藏着好几个子问题:
#      要什么材料?社保怎么办?设备找谁领?试用期怎么算?
#      但一次检索只能从一个点出发,捞回来的 k 条**全挤在同一个语义区域**,
#      别的侧面一条都进不来。
#
# 思路很自然:既然一个查询覆盖不了,那**多生成几个查询**不就行了?
# 人来做这件事就是"换个说法再搜一遍"。让 LLM 自动做,就是 Multi-Query 策略。
#
# ===== 一句话原理 =====
#   LLM 把原问题改写成 N 个不同查询 → 每个查询各检索一次 → 结果**取并集、去重**。
#
# 听上去稳赚不赔,对吧?**不完全是** —— 本文件第 4 部分会用实测数据告诉你
# 它什么时候有用、什么时候纯属浪费钱。请务必看到那部分。


# ============================================================================
# 第 1 部分:最小用法 —— 以及"怎么看它到底生成了什么查询"
# ============================================================================
# ⚠️ 第一个坑就是导入路径。视频教程里写的是:
#
#       from langchain.retrievers.multi_query import MultiQueryRetriever
#
#   但你现在装的是 LangChain 1.x,`langchain` 这个包**根本没装** —— 它被拆开了。
#   这个类现在住在 langchain_classic 里,正确写法是下面这样。

import logging

import dotenv
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_deepseek import ChatDeepSeek
from langchain_huggingface import HuggingFaceEmbeddings

dotenv.load_dotenv()

# ---- 看生成查询的关键:配 logging ----
# MultiQueryRetriever 内部是用 logger.info() 把生成的查询打出来的,
# 而 Python 的 logging **默认级别是 WARNING,INFO 根本不显示**。
# 所以你如果不配下面这两行,就会觉得"它怎么什么反应都没有"——
# 它其实一直在工作,只是你没开开关。
#
# 第一行:配置一个输出格式。level 设成 WARNING 是为了**不**让别的库刷屏。
# 第二行:只把 multi_query 这个 logger 单独调到 INFO。
#        logger 的名字就是模块的 __import__ 路径。
logging.basicConfig(format="    [%(name)s] %(message)s", level=logging.WARNING)
logging.getLogger("langchain_classic.retrievers.multi_query").setLevel(logging.INFO)

# ---- 一个员工入职知识库,故意做成"一个问题的多个侧面"----
# 这批文档是特意设计的:它们都跟"入职"有关,但分属完全不同的侧面
# (材料、社保、设备、账号、考核…)。这样才能看出多查询到底有没有帮上忙。
KB = [
    ("入职材料", "报到当天需携带身份证原件、学历证书复印件、离职证明和近三个月的体检报告。"),
    ("入职体检", "新员工应在报到前一周内到三甲医院完成入职体检,费用入职后可凭发票全额报销。"),
    ("社保公积金", "公司在新员工入职当月为其办理社会保险与住房公积金的开户手续,次月起正常缴纳。"),
    ("工位分配", "报到当日由行政部门分配工位,并说明所在楼层的茶水间与洗手间位置。"),
    ("办公设备", "IT 部门为每位新员工配发一台笔记本电脑和一台外接显示器,离职时需归还。"),
    ("邮箱账号", "信息技术部会在入职首日为新员工开通企业邮箱与内部办公系统账号。"),
    ("门禁权限", "行政前台发放工牌与门禁卡,门禁卡工本费二十元,离职时退卡返还。"),
    ("试用期考核", "试用期为三个月,转正需要直属主管与部门负责人双重签字确认考核结果。"),
    ("导师制度", "每位新员工入职后会分配一位导师,负责解答前三个月的业务疑问。"),
    ("年假规则", "员工入职满一年后可享受带薪年假,首年为五天,此后每满三年增加一天。"),
    ("薪资发放", "工资于每月十五日发放至员工本人银行卡,遇节假日提前至最近一个工作日。"),
    ("费用报销", "差旅与办公费用凭合规发票在内部系统提交报销单,审批通过后随当月工资发放。"),
    ("入职培训", "公司每月第一周组织新员工入职培训,内容包括企业文化、信息安全与合规。"),
    ("团建活动", "各部门每季度组织一次团建活动,费用由公司统一承担,员工自愿参加。"),
]

embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
db = FAISS.from_documents(
    [Document(page_content=text, metadata={"topic": topic}) for topic, text in KB],
    embedding,
)

llm = ChatDeepSeek(model="deepseek-chat", temperature=0)

# ---- 最小用法:就两行 ----
#   retriever : 底层用哪个检索器(这里用上一章的 as_retriever)
#   llm       : 用哪个模型来改写查询
base_retriever = db.as_retriever(search_kwargs={"k": 4})
mq_retriever = MultiQueryRetriever.from_llm(retriever=base_retriever, llm=llm)

QUESTION = "入职要准备什么?"

# print("=" * 74)
# print("第 1 部分:最小用法 + 打开 logging 看生成的查询")
# print("=" * 74)
# print(f'\n原始问题: "{QUESTION}"')
# print("调用 mq_retriever.invoke(...) ...\n")

# # ⚠️ 这里只调用**一次**,把结果存下来。
# #    千万别在 print 里连着调两次(比如先 len() 再取内容)——
# #    每调一次就会重新请求一次 LLM,不但烧钱,而且两次结果可能不一样(见第 6 部分)。
mq_docs = mq_retriever.invoke(QUESTION)

# print(f"\n最终返回 {len(mq_docs)} 条:")
# for i, d in enumerate(mq_docs):
#     print(f"  [{i}] [{d.metadata['topic']}] {d.page_content[:30]}...")

# 上面那几行 "Generated queries: [...]" 就是它自动生成的查询。
# 划重点:这些查询**不是同义改写**吗?先别急着下结论,第 4 部分会算这笔账。
#
# 小提示:如果直接双击运行或在终端里跑,这些日志会**整整齐齐地插在上面的输出中间**。
# 但如果你把输出重定向到文件(`python xxx.py > out.txt`)或者用管道过滤,
# 可能会发现日志**全挤到最前面**去了 —— 这不是 bug。
# 原因是 logging 默认写到 **stderr**,而 print 写 **stdout**;两者缓冲策略不同,
# 被重定向时顺序就会错乱。想让日志也进 stdout,加一句就行:
#     logging.basicConfig(..., stream=sys.stdout)


# ============================================================================
# 第 2 部分:它内部到底干了什么
# ============================================================================
# 三个动作,非常简单:
#
#   1. 调 **一次** LLM,让它输出 N 条查询(一次请求里全生成,不是循环调 N 次)
#   2. 拿这 N 条查询,各去检索一次,把所有结果**堆在一个大列表里**
#   3. 去重,返回并集
#
# 对应到源码(langchain_classic/retrievers/multi_query.py):
#
#       def _get_relevant_documents(self, query, *, run_manager):
#           queries = self.generate_queries(query, run_manager)   # 第 1 步
#           if self.include_original:
#               queries.append(query)
#           documents = self.retrieve_documents(queries, run_manager)  # 第 2 步
#           return self.unique_union(documents)                        # 第 3 步
#
# 由这段代码能推出几个**很重要但教程通常不讲**的结论:
#
#   ★ 结论 1:结果**没有按相关性排序**。
#     `retrieve_documents` 就是简单地把每条查询的结果 extend 在一起,
#     所以顺序是"第 1 条查询的结果、第 2 条查询的结果…"。
#     单查询的 similarity_search 返回的是按分数排好序的列表,
#     多查询**丢掉了这个顺序信息** —— 它只保证"捞回来了"。
#     如果你下游只取前 3 条,拿到的其实是"第 1 条生成查询的前 3 条",
#     这是个很容易踩的坑。
#
#   ★ 结论 2:去重靠的是 Document 相等判断(`doc not in documents[:i]`)。
#     FAISS 返回的是存进去的同一批对象,id 一样,所以去重是有效的。
#
#   ★ 结论 3:`include_original` 默认是 **False**(源码里写死的)。
#     意味着**用户原话压根不参与检索**,只拿 LLM 改写的查询去搜。
#     很多人以为原查询一定在里面,这是个想当然。第 7 部分会专门验证它。
#
#   ★ 结论 4:同步路径下 N 次检索是 **for 循环串行**的;
#     异步路径(ainvoke)才用 asyncio.gather 并发。
#     本地 FAISS 检索很快,这点差别可以忽略;但如果你的向量库在远端,值得用异步。


# ============================================================================
# 第 3 部分:实测对比 —— 它到底有没有提升?
# ============================================================================
# 光讲原理没用,直接量。对比三组:
#   A. 单查询 k=4                     —— 上一章的基线
#   B. 多查询(默认提示词)            —— 3 条生成查询 × k=4,去重
#   C. 单查询 k=12                    —— 公平参照:多查询用了 3×4 的预算,那单查询也给它 12

# 先看一下基线:单查询能捞到什么
# single_k4 = db.as_retriever(search_kwargs={"k": 4}).invoke(QUESTION)


def show(label, docs):
    topics = [d.metadata["topic"] for d in docs]
    print(f"  {label:<26} {len(docs):>2} 条: {topics}")


# print()
# print("=" * 74)
# print(f'第 3 部分:实测对比(问题:"{QUESTION}")')
# print("=" * 74)
# print()
# show("A. 单查询 k=4", single_k4)
# show("B. 多查询(默认提示词)", mq_docs)

# C 组故意用了 12 —— 多查询开了 3 条查询、每条 k=4,一共花了 12 个名额的检索预算。
# 要跟它比,单查询也得给到 12,否则就是拿 4 条比 12 条,赢得不光彩。
# show("C. 单查询 k=12(同预算)", db.as_retriever(search_kwargs={"k": 12}).invoke(QUESTION))

# print()
# print("  对照:C 组捞到的全部内容:")
# for d in db.as_retriever(search_kwargs={"k": 12}).invoke(QUESTION):
#     print(f"      [{d.metadata['topic']}]")

# ===== 现在看这份数据说明了什么 =====
#
# A(4 条)确实比 B 少 —— 多查询看起来有效果。
# 但拿 C 一对照,事情就不对劲了:
#
#   B 捞回来的那些,B 里有的 C 里**全都有**。
#   也就是说在**这套知识库**上,"多查询"完全等价于"单查询把 k 调大"。
#
# 这不是知识库太小导致的偶然。它指向一个更本质的问题 —— 下一部分说。


# ============================================================================
# 第 4 部分:★ 默认提示词的短板(这是本文件最重要的一节)
# ============================================================================
# 回头看第 1 部分打印出来的那三条生成查询:
#
#     '入职需要准备哪些材料？'
#     '新员工入职前需要做哪些准备工作？'
#     '报到入职时一般要带什么文件和物品？'
#
# 读三遍 —— 它们是**同一个问题的三种说法**。
#
# 这不是模型偷懒,是**默认提示词就是这么要求的**。看它的原文
# (multi_query.py 里的 DEFAULT_QUERY_PROMPT,注意是**英文**的):
#
#     "...generate 3 different versions of the given user question...
#      By generating multiple perspectives on the user question,
#      your goal is to help the user overcome some of the limitations
#      of distance-based similarity search..."
#
# 它要的是 "different **versions**" —— 不同**版本**,也就是同义改写。
# 虽然最后提了一句 multiple perspectives,但"换个说法"这个指令太强了,
# 模型就照着换说法去了。
#
# ===== 为什么同义改写没用? =====
#
# 这正是第 0 部分讲的第 1 种翻车方式,而且它是最轻的一种。
# 三句同义的话,embedding 之后是**三个挨得极近的点**,
# 检索出来的圈子几乎完全重叠。
#
#   改写查询落在同一个区域 → 并集 ≈ 单个查询的结果 → 白花一次 LLM 调用
#
# 所以要解决"问题太宽"这个真正的毛病,需要的不是"换个说法",
# 而是**从不同角度提问**。这两件事区别很大:
#
#     同义改写:要带什么材料 → 需要哪些证件 → 报到要交什么文件
#               (同一个侧面,措辞不同)
#
#     不同角度:要带什么材料 → 社保从哪个月开始缴 → 试用期多长 → 有什么常见坑
#               (不同侧面,检索到的是知识库里**不同区域**的文档)
#
# 默认提示词做不到后者。那就自己写一个 —— 下一部分。


# ============================================================================
# 第 5 部分:自定义提示词 —— 要"不同角度",不要"不同说法"
# ============================================================================
# 自定义提示词有两条**硬性要求**:
#
#   1. 必须是一个 PromptTemplate,而且 input_variables 里得有 "question"
#      —— 因为框架调用时只传 {"question": ...} 这一个变量。
#   2. 输出必须是**一行一条**查询。
#      因为解析用的是 LineListOutputParser,它就是简单地按 \n 切开、丢掉空行。
#      所以你**在提示词里要求"不要加编号"**,否则 "1. xxx" 里的 "1." 会混进查询文本,
#      污染向量检索(虽然影响不大,但没必要)。

from langchain_core.prompts import PromptTemplate

FACET_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""你是检索助手。请针对下面的问题,从**互不相同的关注角度**生成 3 条检索查询,
覆盖该问题的不同侧面(例如:需要哪些材料、涉及哪些时间节点、有哪些权利与义务、
常见坑与注意事项)。要求每条查询的侧重点都不一样,**不要只是把原问题换一种说法**。

每条查询占一行,行首不要加编号或符号,不要输出任何解释。

问题:{question}""",
)

mq_facet = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=llm,
    prompt=FACET_PROMPT,
)

# print()
# print("=" * 74)
# print("第 5 部分:换成【不同角度】的提示词")
# print("=" * 74)
# print(f'\n原始问题: "{QUESTION}"')
# print("调用中...\n")

# facet_docs = mq_facet.invoke(QUESTION)

# print()
# show("B. 多查询(默认提示词)", mq_docs)
# show("D. 多查询(不同角度提示词)", facet_docs)

# 对比 B 和 D 的 topic 列表。D 捞到的文档覆盖了**明显不同**的侧面 ——
# 这才是多查询本来该有的样子。
#
# ⚠️ 但这里必须诚实说三件事:
#
#   1. 我让它"生成 3 条",它实际输出了 **4 条**。
#      条数**不受参数控制**,完全由模型输出决定,解析器是"有几行就收几行"。
#      想严格控数量,得在提示词里反复强调,或者在解析后手动切片。
#      条数直接决定检索预算(N 条 × k),所以这一点要有意识地管。
#
#   2. 效果**不稳定**。提示词写得好不好、模型当时的状态,都会影响结果。
#      第 6 部分会给出实测证据。
#
#   3. 它仍然不是"稳赢"。看第 3 部分的 C 组 —— 只要知识库小到
#      "把 k 调大就能全覆盖",单查询就足够,多查询只是多花一次 LLM 调用。
#      多查询的真正价值在**知识库大到不能靠 k 硬吃**的时候:
#      一个知识库几万条文档,你不可能设 k=5000 把上下文塞爆,
#      这时"从几个不同角度各捞 k 条"才是划算的。


# ============================================================================
# 第 6 部分:两个版本坑 + 一个必须知道的非确定性
# ============================================================================
# ---- 坑 1:parser_key 已经废弃 ----
# 老教程会让你这么写:
#
#       MultiQueryRetriever.from_llm(..., parser_key="lines")
#
# 在你这个版本里**这个参数完全不起作用**,源码注释里明写了:
#
#       parser_key: str = "lines"
#       """DEPRECATED. parser_key is no longer used and should not be specified."""
#
# 传了不会报错,也不会有任何效果 —— 这种"静默失效"的参数最难排查。
#
# ---- 坑 2:include_original 默认 False ----
# 源码里:  include_original: bool = False
# 也就是说用户原话默认**不参与检索**。想让原查询也搜一遍,得显式传:
#
#       MultiQueryRetriever.from_llm(..., include_original=True)
#
# 下一部分用一个确定性的例子把它验证清楚。
#
# ---- 非确定性:同一个问题,两次运行结果可能不一样 ----
# 这个坑最隐蔽。下面**故意**把同一个问题问两次,并开启日志对比:

# print()
# print("=" * 74)
# print("第 6 部分:同一问题连问两次,生成的查询一样吗?")
# print("=" * 74)
# print("(注意看下面两行 Generated queries,不是第 1 部分的缓存,是**重新生成**的)\n")

# for i in (1, 2):
#     print(f"  ---- 第 {i} 次 ----")
#     mq_facet.invoke(QUESTION)

# 即使 temperature=0,两次结果也**很可能不一样**。
# 原因:大模型的推理本身不保证逐位可复现(并行计算、批处理、MoE 路由等),
# temperature=0 只降低了随机性,不等于确定性。
#
# 这件事对多查询策略的杀伤力比别人大 —— 因为生成查询是**检索的第一步**,
# 它一变,后面检索到哪几条、LLM 最终基于什么上下文回答,全都跟着变。
#
# 后果很实际:
#   - 同一个问题,用户今天问和明天问,答案的**依据**可能不同
#   - 线上出问题时**难以复现**,因为重跑一次可能就"好了"
#   - 写测试时**不能用真实 LLM** —— 所以下一部分改用假 LLM


# ============================================================================
# 第 7 部分:用假 LLM 做确定性验证 —— include_original 到底管什么
# ============================================================================
# 要验证一个参数的作用,前提是**把其他变量固定住**。
# 真实 LLM 每次都生成不同的查询,根本没法做对照实验。
# 那就把"生成查询"这一步换成写死的假模型。
#
# FakeListChatModel 是 langchain_core 自带的测试替身:
# 你给它一组固定回复,它就照本宣科地返回,不联网、不花钱、完全可复现。

from langchain_core.language_models.fake_chat_models import FakeListChatModel

# 让它永远只"生成"一条跟年假有关的查询。
# 而下面问的原问题是"入职材料",两者指向知识库里**不同的文档**。
# 于是:原查询参不参与检索,结果会有一目了然的差别。
fake_llm = FakeListChatModel(responses=["年假有几天"])

# 注意 k=1,让每条查询只取 1 条 —— 结果最干净,一眼看清来源
probe_retriever = db.as_retriever(search_kwargs={"k": 1})

mq_no_orig = MultiQueryRetriever.from_llm(
    retriever=probe_retriever, llm=fake_llm, include_original=False
)
mq_with_orig = MultiQueryRetriever.from_llm(
    retriever=probe_retriever, llm=fake_llm, include_original=True
)

PROBE_Q = "报到当天要带哪些材料"

# print()
# print("=" * 74)
# print("第 7 部分:假 LLM 验证 include_original(结果完全可复现)")
# print("=" * 74)
# print()
# print('  固定的"生成查询" = ["年假有几天"]  (由假 LLM 写死)')
# print(f'  用户原问题       = "{PROBE_Q}"')
# print()
# show("include_original=False", mq_no_orig.invoke(PROBE_Q))
# show("include_original=True ", mq_with_orig.invoke(PROBE_Q))

# 看明白了:
#   False → 只捞到"年假规则",因为用户原话压根没去检索
#   True  → 多出"入职材料",因为原话也搜了一遍
#
# 而且顺序也印证了第 2 部分的结论:原查询的结果**追加在最后**,
# 不是按相关性插进去的 —— 因为它就是 `queries.append(query)` 加到列表末尾。
#
# 这个假模型还有个附加价值:写单元测试时用它,
# 断言就变成了确定的,不会因为模型今天心情不同而忽红忽绿。


# ============================================================================
# 第 8 部分:成本与适用场景
# ============================================================================
#   LLM 调用次数:**1 次**(注意不是 N 次 —— N 条查询是一次请求里一起生成的)
#   向量检索次数:N 次(每条查询一次)
#   额外延迟:1 次 LLM 调用的时间(通常几百毫秒到几秒)
#
# 以本文件为例:每条查询 k=4,生成 4 条查询 → 检索 4 次、最多返回 16 条(去重后更少)。
# 相比单查询,多花的时间几乎全在那**一次 LLM 调用**上。
#
# ===== 什么时候该用 =====
#
#   知识库很大(几万条以上),不能靠调大 k 来覆盖   → 用,这是它的主场
#   实测发现检索召回不足,且问题是"多侧面"的        → 用,但要**自定义提示词**
#   知识库只有几十条,调大 k 就全覆盖了             → **别用**,白花一次调用
#   对延迟敏感(比如逐字流式对话)                    → 慎用,它给检索加了 LLM 往返
#   需要结果严格按相关性排序                        → 别直接用,它不排序(第 2 部分结论 1)


# # ============================================================================
# # 小结
# # ============================================================================
# #   机制一句话:1 次 LLM 生成 N 条查询 → 各检索一次 → 取并集去重。
# #
# #   必须记住的四点:
# #     1. 导入路径是 langchain_classic.retrievers.multi_query(1.x 没有 langchain 包了)
# #     2. 不配 logging 你看不到生成的查询 —— 它用 logger.info 输出,默认不显示
# #     3. **默认提示词生成的是同义改写,不是不同角度**。
# #        三句话落在同一个语义区域,并集几乎不变宽。
# #        要用它就得**自己写提示词**要求"不同侧面"。
# #     4. 返回的并集**没有按相关性排序**,顺序是"第1条查询的结果、第2条查询的结果…"
# #
# #   三个坑:
# #     parser_key 已废弃(传了静默失效)
# #     include_original 默认 False(用户原话默认不参与检索)
# #     同一个问题两次运行结果可能不同(非确定性,测试时请用假模型)
# #
# #   最该记住的一句:
# #     多查询**不是免费的午餐**。如果调大 k 就能解决,就别用它。
