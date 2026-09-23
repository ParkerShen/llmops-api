'''
Date: 2026-09-22
Author: parker
FilePath: \llmops-api\study\57-Agent基础\1.Chain与agent的区别.py
Description: Agent 基础 —— Chain 与 Agent 的区别
'''

# ============================================================
# 【今天要理解的概念】
#
# 55 章最后一个文件里，你手写了一个 while 圈：
#
#     问模型 → 模型说要调哪个工具 → 执行工具 → 喂回去 → 再问模型
#
# 那个圈，就是一个 Agent 的内核。
# 所以今天不是学新东西，是学两件事：
#   ① 给它取个名字：Agent ≠ Chain
#   ② 以后不用自己手写那个 while 了，框架帮你跑
#
# ------------------------------------------------------------
# 【Chain 和 Agent 差的不是"能力"，是"谁做决定"】
#
#   Chain = 你在【写代码的时候】就把步骤排好了
#           第一步做什么、第二步做什么，全写在代码里，写死。
#           LLM 只是流水线上的一个工位，它不知道自己在第几步。
#
#   Agent = 你在【写代码的时候】不知道要走几步
#           LLM 在【运行时】决定下一步干什么，可能一步就结束，
#           也可能转 10 圈。你的代码只负责"提供工具 + 负责执行"。
#
# ------------------------------------------------------------
# 【用你熟悉的前端思维类比】
#
#   Chain  ≈   a().then(b).then(c)        —— 或者你手写的一条 Promise 链
#              redux 的 middleware 顺序是你定的，永远按这个顺序走。
#
#   Agent  ≈   一个事件循环 + 路由表
#              下一步 dispatch 哪个 action，取决于"外部传进来的指令"。
#              只不过这里的指令不是用户点的按钮，是 LLM 输出的 tool_calls。
#
#   一句话：Chain 的流程图是你画的，Agent 的流程图是 LLM 画的。
#
# ------------------------------------------------------------
# 【那 Chain 有什么用？】—— 别急着觉得它低级
#
#   流程确定的事（翻译、总结、分类、RAG 检索问答），Chain 更快、更便宜、
#   更可控。你写 RAG 那一章用的全是 Chain，那是对的。
#   只有"流程不确定、要模型自己看着办"的时候，才需要 Agent。
#   （55 章你那个电商客服就是：用户可能问天气、可能问订单、可能问运费，
#     你写代码时不知道他问哪个 —— 所以必须让 LLM 运行时决定。）
# ============================================================
import dotenv
from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

# 加载环境变量并初始化模型
dotenv.load_dotenv()
llm = ChatDeepSeek(model="deepseek-chat", temperature=0)

# ============================================================
# 【今天的第 1 个小任务】—— 先体会 Chain 有多"死"
#
# 写一个两步的 Chain：
#
#   输入：一个主题字符串，比如 "无线蓝牙耳机"
#   第 1 步：让 LLM 根据这个主题起一个中文产品名（8 个字以内）
#   第 2 步：让 LLM 把第 1 步得到的中文名，翻译成英文
#
# 【要求】
#   - 用 LCEL 管道写法 |  ，你在 RAG 章用过
#   - 两个 prompt，各一个 ChatPromptTemplate（from langchain_core.prompts）
#   - 末尾用 StrOutputParser() 拿纯字符串（from langchain_core.output_parsers）
#   - 模型那两行、dotenv 那两行，从 55 章文件原样抄：
#         dotenv.load_dotenv()
#         from langchain_deepseek import ChatDeepSeek
#         llm = ChatDeepSeek(model="deepseek-chat", temperature=0)
#   - 最后 print 出两个结果：中文名 和 英文名
#
# 【今天这个文件里，不允许出现 if / for / while】
#
#   一个判断都不许有。这不是我矫情 —— 这就是今天要你亲手体会的：
#   Chain 里没有"决定"，只有"顺序"。它连"要不要做第二步"都问不了自己。
#
# ⚠️ 提示：两个 prompt 怎么串起来？第一个 chain 的输出，就是第二个
#    prompt 的输入变量。写成 {"topic": chain1, "中文名": ...} 这种形式试试。
#    报错不可怕，报错说明你摸到边界了。
#
# TODO(你写)：
# ------------------------------------------------------------
# 第一步Prompt： 起中文名
# import dotenv

# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser

# dotenv.load_dotenv()
# from langchain_deepseek import ChatDeepSeek
# llm = ChatDeepSeek(model="deepseek-chat", temperature=0)


# 第一步
prompt1 = ChatPromptTemplate.from_template(
    """
根据下面的主题，起一个中文产品名。

要求：
1. 8个字以内
2. 简洁、有吸引力
3. 只输出产品名，不要解释

主题：{topic}
"""
)

chain1 = prompt1 | llm | StrOutputParser()


# 第二步
prompt2 = ChatPromptTemplate.from_template(
    """
请把下面的中文产品名翻译成自然、简洁的英文产品名。
只输出英文名称，不要解释。

中文产品名：{中文名}
"""
)

chain2 = prompt2 | llm | StrOutputParser()
full_chain = chain1 | prompt2 | llm | StrOutputParser()

# 第一步
中文名 = chain1.invoke({
    "topic": "无线蓝牙耳机"
})

# 第二步
英文名 = chain2.invoke({
    "中文名": 中文名
})


print("英文名：", full_chain.invoke({"topic": "无线蓝牙耳机"}))
# ============================================================
# 【写完自问 3 句话】—— 这个文件就是让你回答这 3 句
#
#   1. 这个 Chain 跑起来，第 2 步有可能被跳过吗？
#
#   2. 如果我想改成"先判断主题里有没有数字，再决定要不要翻译"，
#      我该改哪儿？—— 改 Python 代码，对吧？那这就是 Chain 的代价。
#
#   3. 如果这个判断只有 LLM 做得出来呢？
#      （比如"判断这个主题适合走高端风还是便宜风"）
#      Chain 还能胜任吗？如果不能，那缺的是什么？
#
#      答完第 3 句，你就已经摸到 Agent 了。
#
# 【答案】
#
#   1. 不可能被跳过。
#      full_chain 一旦 invoke()，管道就从左走到右，一步不落。
#      | 这个符号只有一种语义：把左边吐出来的东西交给右边。
#      它没有"这一步要不要做"的概念 —— 因为"要不要"是个判断，
#      而管道里没有判断的位置。想跳过第 2 步，你只能去改代码。
#
# ------------------------------------------------------------
#   2. 改 Python。把 full_chain 拆开，自己写 if：
#
#         topic = "无线蓝牙耳机"
#         if any(c.isdigit() for c in topic):
#             英文名 = full_chain.invoke({"topic": topic})
#         else:
#             英文名 = chain1.invoke({"topic": topic})
#
#      这能跑。但代价是两条，而且会越来越大：
#
#        ① 这个判断是【你】写的，不是 LLM 写的。
#           所以你只能判断你事先想到的情况。
#           明天产品说"带'智能'两个字的走另一套流程"，你就得再改一次代码。
#
#        ② 每加一个分支，你的管道就碎一块。
#           | 组合起来的东西被拆成 if/else，链越写越短，if 越写越长。
#           最后你不是在"写 prompt 的人"，而是在"写 if-else 的人"。
#
# ------------------------------------------------------------
#   3. 胜任不了。缺的是【决定权】和【回头路】。
#
#      拆开看 Chain 的两条硬伤：
#
#        · 单向：管道只往前走，走完就结束，它没有"回到上一步再看看"的能力。
#        · 无判断：走哪条分支，是 Python 的 if 决定的，不是 LLM。
#
#      要补上这两条，需要的东西恰好是：
#
#        让 LLM 输出"下一步做什么" → 代码去执行它 → 再回头问 LLM "现在呢？"
#
#      那个"再回头问"，就是循环。Chain 是直的一条线，Agent 是一个环。
#
#      ┌─ Chain ─────────────────────────┐
#      │  prompt → llm → 解析 → prompt → llm → 解析 → 结束 │
#      └─────────────────────────────────┘
#
#      ┌─ Agent ─────────────────────────┐
#      │  ┌→ 问 LLM：下一步做什么？        │
#      │  │      ↓ 它说要调工具 A          │
#      │  │   执行工具 A                   │
#      │  │      ↓ 结果喂回去               │
#      │  └── 再问 LLM：现在呢？──────────┘
#      │         ↓ 它说"我答完了"           │
#      │       输出最终回答                 │
#      └─────────────────────────────────┘
#
#      —— 这就是你 55 章最后那个文件里手写的 while 圈。
#         它多出来的唯一一样东西，就是那个【环】。
#
#      所以 LangChain 的 Agent = 那个 while 圈 + 一堆工具
#                                + 让 LLM 自己决定什么时候停下来。
# ------------------------------------------------------------
