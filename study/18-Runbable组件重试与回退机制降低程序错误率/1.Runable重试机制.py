'''
Date: 2026-09-09 20:35:01
Author: parker
FilePath: \llmops-api\study\18-Runbable组件重试与回退机制降低程序错误率\1.Runable重试机制.py
Description: 
'''
from langchain_core.runnables import RunnableLambda

counter = -1

def func(x):
    global counter
    counter += 1
    print(f"当前的值为 {counter=}")
    return x / counter

chain = RunnableLambda(func).with_retry(stop_after_attempt=2)

resp = chain.invoke(2)

print(resp)