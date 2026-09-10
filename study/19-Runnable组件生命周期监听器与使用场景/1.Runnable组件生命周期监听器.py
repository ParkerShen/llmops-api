'''
Date: 2026-09-10 15:15:33
Author: parker
FilePath: \llmops-api\study\19-Runnable组件生命周期监听器与使用场景\1.Runnable组件生命周期监听器.py
Description: 
'''
import time

from langchain_core.runnables import RunnableLambda

# # 1. 创建RunnableLambda与链
# runnable = RunnableLambda(lambda x: time.sleep(x))
# chain = runnable

# # 2. 调用并执行链
# chain.invoke(input=2, config={"configurable": {"name": "慕小课"}})

def test(x):
    print(f"开始执行，输入：{x}")
    time.sleep(x)
    print("执行结束")
    
runnable = RunnableLambda(test)

runnable.invoke(2)