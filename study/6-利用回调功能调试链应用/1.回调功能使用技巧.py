'''
Date: 2026-09-04 18:14:52
Author: parker
FilePath: \llmops-api\study\6-利用回调功能调试链应用\1.回调功能使用技巧.py
Description: 
'''
'''
Date: 2026-09-04 17:29:34
Author: parker
FilePath: \llmops-api\study\5-两个Runnable核心类的讲解与使用\3.RunnablePassthrough简化innoke.py
Description: 
'''
'''
Date: 2026-09-04 16:44:16
Author: parker
FilePath: \llmops-api\study\5-两个Runnable核心类的讲解与使用\2.RunnableParallel模拟检索.py
Description: 
'''
import time
from typing import Any

import dotenv
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_deepseek import ChatDeepSeek
# from langchain_core.runnables import RunnableParallel
from langchain_core.runnables import RunnablePassthrough 
from langchain_core.callbacks import StdOutCallbackHandler, BaseCallbackHandler

dotenv.load_dotenv()

class LLMOpsCallbackHandler(BaseCallbackHandler):
    """自定义回调处理器"""
    start_at: float = 0

    def on_chat_model_start(self, serialized, messages, *, run_id, parent_run_id = None, tags = None, metadata = None, **kwargs)-> Any:
        print("聊天模型开始执行了")
        # print("messages:", messages)
        # print("serialized", serialized)
        self.start_at = time.time()


    def on_llm_new_token(self, token, *, chunk = None, run_id, parent_run_id = None, tags = None, **kwargs):
        print("token:", token)

    def on_chain_end(self, outputs, *, run_id, parent_run_id = None, **kwargs):
       end_at: float = time.time()
       print("完整输出:",outputs)
       print("程序小号：",end_at -self.start_at)
       


prompt = ChatPromptTemplate.from_template("{query}")
#构造大预言模型
llm =ChatDeepSeek(model="deepseek-chat")

parser = StrOutputParser()



chain = (
    {"query":RunnablePassthrough()} 
    | prompt
    | llm
    | parser
)

content = chain.invoke({"query": "你好"},config={"callbacks":[StdOutCallbackHandler(),LLMOpsCallbackHandler()]})


print(content)

