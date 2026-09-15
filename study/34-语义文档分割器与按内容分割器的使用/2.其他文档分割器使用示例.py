'''
Date: 2026-09-13 20:35:00
Author: parker
FilePath: \llmops-api\study\34-语义文档分割器与按内容分割器的使用\2.其他文档分割器使用示例.py
Description: 其他文档分割器 —— TokenTextSplitter(按 token 切) / NLTKTextSplitter(按句子切)

运行方式(在项目根目录):
    python "study/34-语义文档分割器与按内容分割器的使用/2.其他文档分割器使用示例.py"
'''

# ============================================================================
# 第 0 部分：为什么需要换一把"尺子"？
# ============================================================================
#
# 前两章的 CharacterTextSplitter 和 RecursiveCharacterTextSplitter,
# chunk_size 数的都是**字符个数**(len(text))。
#
# 但真正决定下面这两件事的是 **token 数**,不是字符数:
#   1. 模型的上下文窗口塞不塞得下(DeepSeek 是 128K tokens)
#   2. 你花多少钱(按 token 计费,不是按字)
#
# 麻烦在于:字符数和 token 数的比例**不稳定**,中英文差别巨大。
# 下面用真实的 tiktoken 分词器量给你看。

import tiktoken

zh = "人工智能正在改变世界。机器学习是其中的核心技术。" * 5
en = "Artificial intelligence is changing the world. Machine learning is the core technology." * 5

enc = tiktoken.get_encoding("cl100k_base")  # GPT-4 的分词表,国产模型的分词比例与之接近

print("=" * 64)
print("第 0 部分:字符数 vs token 数")
print("=" * 64)
for name, text in [("中文", zh), ("英文", en)]:
    n_char, n_tok = len(text), len(enc.encode(text))
    print(f"  {name}: 字符 {n_char:4d}  |  token {n_tok:4d}  |  "
          f"1 token ≈ {n_char / n_tok:.2f} 个字符")

# 记住这个结论,下面会反复用到:
#   中文 ≈ 1 个 token 对应 1 个字
#   英文 ≈ 1 个 token 对应 4 个字符(~0.75 个单词)
#
# 所以 chunk_size=500 这个配置:
#   按"字符"算 → 中文约 500 字,英文约 500 字符(才 80 个词,碎得没法看)
#   按"token"算 → 中文约 500 字,英文约 2000 字符(才差不多对等)
#
# 用 len() 当尺子,你的英文分块会**碎得莫名其妙**,中文的大致还能用。
# 这就是 TokenTextSplitter 存在的理由。


# ============================================================================
# 第 1 部分:TokenTextSplitter —— 按 token 数量切
# ============================================================================
# 它用 tiktoken 把文本先切成 token,再按数量攒块。
# 关键参数:
#     encoding_name : 用哪张分词表。默认是 "gpt2",那个表对中文极不友好(一个汉字可能切
#                     成 2~3 个 token),**强烈建议显式指定 "cl100k_base"**。
#     chunk_size    : 这里数的是 token 个数,不再是字符个数!
#     chunk_overlap : 同样是 token 个数。

from langchain_text_splitters import TokenTextSplitter

print()
print("=" * 64)
print("第 1 部分:TokenTextSplitter(按 token 切)")
print("=" * 64)

long_zh = (
    "人工智能正在深刻改变着我们的生活方式。从智能手机里的语音助手,到电商平台的推荐算法,"
    "再到医疗影像的辅助诊断,人工智能技术已经渗透到各行各业。机器学习的核心思想是让计算机"
    "从数据中自动发现规律,而不是由程序员手工编写每一条规则。深度学习则进一步引入了多层"
    "神经网络,使得模型能够自动学习到数据的层次化特征表示。"
)

token_splitter = TokenTextSplitter(
    encoding_name="cl100k_base",   # 关键!默认的 gpt2 表不适合中文
    chunk_size=50,                 # 注意:50 个 token,不是 50 个字
    chunk_overlap=5,
)

token_chunks = token_splitter.split_text(long_zh)

print(f"原文: {len(long_zh)} 个字符, {len(enc.encode(long_zh))} 个 token")
print(f"切成 {len(token_chunks)} 块:\n")
for i, c in enumerate(token_chunks):
    # 每一块的真实 token 数,验证它确实在按 token 切
    print(f"  [{i}] token={len(enc.encode(c)):3d}  字符={len(c):3d}  {c!r}")

# 划重点:数一下上面每一块的 token 数,都在 50 左右(不会超太多),
# 而字符数一列是**跳来跳去**的 —— 这正是它和 CharacterTextSplitter 的本质区别。


# ============================================================================
# 第 2 部分:NLTKTextSplitter —— 按"句子"切
# ============================================================================
# 前面的分割器都靠"分隔符字符串"猜句子边界(比如按 "。" 切)。
# 那种做法在遇到 "Dr. Smith" "3.14" "et al." 这种带点的缩写时会切错。
#
# NLTK 是自然语言处理工具包,它的 punkt 分词器是**训练出来的**句子边界模型,
# 能识别缩写、小数点、省略号,切句子比正则准得多。
#
# 注意:它是"先按句子切开,再用 separator + chunk_size 往回合并",
#       所以 chunk_size 仍然是**字符数**。它只改变了"在哪里允许断开"。

from langchain_text_splitters import NLTKTextSplitter

print()
print("=" * 64)
print("第 2 部分:NLTKTextSplitter(按句子切)")
print("=" * 64)

# NLTK 的句子分词器默认是英文的,所以这里用英文举例
en_text = (
    "Dr. Smith went to Washington. He arrived at 3.5 p.m. on Monday. "
    "The meeting covered AI, machine learning, and robotics. "
    "Prof. Lee said the results were promising. However, more data is needed."
)

nltk_splitter = NLTKTextSplitter(
    separator="\n\n",
    chunk_size=120,        # 仍然是字符数
    chunk_overlap=0,
)
nltk_chunks = nltk_splitter.split_text(en_text)

print(f"原文共 {len(en_text)} 个字符\n")
print("NLTK 识别出的句子边界:")
for s in nltk_chunks:
    print(f"  → {s!r}")

# 看 "Dr. Smith" 和 "3.5 p.m." —— 它们都没有被误切成两个句子,
# 这就是训练出来的模型比正则强的地方。

print()
print("-" * 64)
print("!! 关键提醒:拿它切中文会怎样?")

zh_sent = (
    "人工智能正在改变世界。机器学习是核心技术!它需要大量数据吗?"
    "是的,而且需要高质量的数据。数据清洗往往占据整个项目八成的时间。"
)

import nltk

print("  先直接看 NLTK 的句子分词器对中文的输出:")
for s in nltk.sent_tokenize(zh_sent):
    print(f"    → {s!r}")

print()
print("  整段中文被当成了【一个句子】——")
print("  因为 NLTK 默认加载的 english 模型根本不认识 。！？ 这三个符号。")
print("  于是 NLTKTextSplitter 拿到的是【一整句】,切无可切。")
print(f"  注意下面 chunk_size 设成了 60,而这段中文有 {len(zh_sent)} 个字 —— 它照样原样返回:")
for s in NLTKTextSplitter(chunk_size=60, chunk_overlap=0).split_text(zh_sent):
    print(f"    → 长度 {len(s)}  {s!r}")

print()
print("  这正是第 32 章踩过的那个坑:CharacterTextSplitter / NLTKTextSplitter")
print("  都是【先按某个规则切开,再往回合并】,一旦切出来的是一个超大片段,")
print("  它们就无能为力了 —— chunk_size 只是个【尽量不超过】的软目标。")
print("  RecursiveCharacterTextSplitter 之所以是默认选择,就是因为它会【逐级降级】")
print("  继续往下切,不会卡死。")

# 结论:**NLTKTextSplitter 主要适合英文**。中文场景用章节 33 讲的
# RecursiveCharacterTextSplitter + 中文标点分隔符(。！？)就够了。


# ============================================================================
# 第 3 部分:还有两个同类分割器,但你机器上没装依赖
# ============================================================================
# 了解即可,需要时再装:
#
#   SpacyTextSplitter                        pip install spacy
#       用 spaCy 做句子切分,支持多语言(含中文 zh_core_web_sm)。
#       如果你要处理中文句子切分,这个比 NLTK 靠谱。
#
#   SentenceTransformersTokenTextSplitter    pip install sentence-transformers
#       最"讲究"的一个 —— 用**和你的 embedding 模型完全相同的分词器**切。
#       为什么重要?因为 embedding 模型有最大输入长度(比如 512 token),
#       如果切出来的块超过这个长度,会被静默截断,后半截等于白写。
#       用它切能保证每块都不超模型上限。
#       代价是要装 torch,约 2.5GB。


# ============================================================================
# 小结
# ============================================================================
#   TokenTextSplitter   → 该用它的时候:要和模型的上下文窗口 / 计费对齐时
#   NLTKTextSplitter    → 该用它的时候:英文文本,想要干净的句子边界
#
#   两者可以**串联使用**:先用 NLTK 按句子切,再用 TokenTextSplitter 控 token 上限。
#   但注意 —— 99% 的项目里 RecursiveCharacterTextSplitter 就够用了,
#   不要为了"用高级工具"而用它们。
