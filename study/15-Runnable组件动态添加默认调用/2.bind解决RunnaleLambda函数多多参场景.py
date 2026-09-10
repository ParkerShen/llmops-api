'''
Date: 2026-09-09
Author: parker
FilePath: \llmops-api\study\15-Runnable组件动态添加默认调用\2.bind解决RunnableLambda函数多参场景.py
Description:
    第 1 课(llm.bind(stop="o"))是给【模型】预先固定调用参数;
    这一课用同一个 bind,给【普通函数 / RunnableLambda】固定参数。

    核心问题(为什么用 bind):
        RunnableLambda 包装函数放进链后,链真正调用它时,只把"上一步的输出"
        当作【第 1 个位置参数】传进去。如果你的函数需要 2 个以上参数,
        多出来的参数没有任何来源 -> 直接 TypeError 崩掉。

    解决办法(怎么用 bind):
        .bind(参数名=固定值) 会返回一个新的 Runnable,它把这份参数"绑"在身上,
        之后每次真正执行时自动带上:func(上一步输出, 参数名=固定值)。
'''

from langchain_core.runnables import RunnableLambda, RunnablePassthrough


# ============================================================
# Part 1  问题现场:一个函数要 2 个参数,直接丢进链就崩
# ============================================================

def format_reply(query: str, current_date: str) -> str:
    """模拟真实场景:把"用户问题"和"今天的日期"拼成一句回复。

    query:        每次调用都在变 —— 应该由链的输入 / 上一步来给
    current_date: 演示用的"多余参数" —— 链并不会自动把它传进来
    """
    return f"[{current_date}] 收到问题:{query}"


try:
    # 链里写 | format_reply 等价于 RunnableLambda(format_reply)
    RunnableLambda(format_reply).invoke("什么是 bind")
except TypeError as e:
    # 真正的报错: format_reply() missing 1 required positional argument: 'current_date'
    print(f"[1] 多参函数直接调用 -> 报错:\n    {type(e).__name__}: {e}")

# 补充:就算你把两个值都塞进一个 dict 里 invoke,照样崩。
# Runnable 不会"按参数名把 dict 自动解包"成函数形参,它只会把整个 dict
# 当成第 1 个位置参数塞给函数 -> 所以 query=整个dict, current_date 还是没人给。
try:
    RunnableLambda(format_reply).invoke(
        {"query": "什么是 bind", "current_date": "2026-09-09"}
    )
except TypeError as e:
    print(f"[2] 传 dict 也一样崩(不会自动解包成具名参数):\n    {type(e).__name__}: {e}")


# ============================================================
# Part 2  用 bind 修复:把"链里不流转的参数"提前绑到 Runnable 身上
# ============================================================

runnable = RunnableLambda(format_reply).bind(current_date="2026-09-09")
# bind 返回一个新的 Runnable(内部叫 _RunnableBinding),它存着一份默认参数。
# 每次 invoke 真正执行时,等价于:
#     format_reply(上一步输出, current_date="2026-09-09")
# 也就是: 链的输出 填 第1个位置参数,  bind 的关键字参数 填 剩余的形参。

print(f"[3] bind 补上参数后:  {runnable.invoke('什么是 bind')}")
print(f"[4] 换问题也一样:      {runnable.invoke('bind 和 partial 像吗')}")
# 注意:current_date 是绑死在上面的,所以每次调用都带同一个日期,
# 这就是标题说的"动态添加【默认】调用参数"。


# ============================================================
# Part 3  在真正的链里用(纯离线示例,不需要联网调模型)
# ============================================================

def assemble(content: dict) -> str:
    """把字典拼成一段"伪模型输出"。只吃 1 个参数,没问题。"""
    return f"你问:{content['question']}\n我的看法:{content['idea']}"


def polish(text: str, style: str) -> str:
    """对上一步的输出(text)再包装一层。

    text:   来自链上一步(assemble 的返回值)—— 链会自动传
    style:  展示风格,链里没有这个值 —— 属于业务上的"固定参数",用 bind 补
    """
    return f"== {style}风格 ==\n{text}"


chain = (
    # 第 5 课学过的 assign:给输入 dict 追加一个 idea 字段
    # (assign 的字段要么给 Runnable、要么给可调用对象;_ 表示忽略"上一步的输入")
    RunnablePassthrough.assign(idea=lambda _: "bind 用来补链里不流转的固定参数")
    | RunnableLambda(assemble)                    # 1 个参数,链直接喂
    | RunnableLambda(polish).bind(style="轻松")   # 2 个参数,style 靠 bind 给 ← 重点
)

print("[5] 链内使用 bind:")
print(chain.invoke({"question": "bind 怎么用?"}))
print("-" * 40)
# 换个 question 再跑:idea、style 都不变,只有 question 变化 ——
# 这正好说明 style 这种"每个链都固定"的参数不该每次塞进输入里,
# 用 bind 绑一次,这个 Runnable 走到哪都自带这份参数。


# ============================================================
# Part 4  一句话总结(配合第 1 课一起记)
#   llm.bind(stop="o")           -> 给【模型调用】加默认参数
#   RunnableLambda(f).bind(x=..) -> 给【普通函数调用】加默认参数
#   两者底层同一个 API:Runnable.bind(**kwargs) 返回新 Runnable,
#   执行时 = 原调用(正常入参, **kwargs)。

#   什么时候需要 bind:
#     链里某个函数需要 >1 个参数,而除了"上一步输出"外,
#     其余参数不会从链的数据流里过来(是固定值 / 环境值 / 配置值)。

#   怎么设计函数:
#     把"会随每次调用变的值"放在第 1 个位置参数(链自动喂),
#     把"固定的值"放在后面,在链上 .bind(固定参数=值)。

#   一个常见的"反面写法"也别学:为了让多参函数能跑,
#   故意把函数改成只收一个 dict、再在函数里手动 data["xxx"] 取值,
#   既丑又把函数的职责搞乱 —— 该用 bind 就用 bind。
# ============================================================
