'''
Date: 2026-08-28 14:52:33
Author: parker
FilePath: \llmops-api\internal\handler\app_handler.py
'''
from dataclasses import dataclass
import os
from typing import Any, Dict
import uuid

from injector import inject
from langchain_core.tracers import Run
import requests
from flask import request
from werkzeug.datastructures import MultiDict

from internal.schema.app_schema import CompletionReq

from internal.service import AppService

from pkg.response import success_json, validation_error_json, success_message, not_found_message

from operator import itemgetter

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
# 下面是课程 0.x 时代的"经典版"记忆组件,在 1.x 里搬了家
from langchain_classic.memory import ConversationBufferWindowMemory
from langchain_classic.memory.chat_memory import BaseMemory
from langchain_community.chat_message_histories import FileChatMessageHistory

from langchain_deepseek import ChatDeepSeek
@inject
@dataclass

class AppHandler:
    """应用控制器"""
    app_service: AppService

    def create_app(self):
        # 调用服务创建新的app记录
        app = self.app_service.create_app()
        return success_message("应用创建成功, id={app.id}")

    def get_app(self, id: uuid.UUID):
        # 调用服务获取app记录
        app = self.app_service.get_app(id)
        return success_message(f"应用信息: id={app.id}, name={app.name}, description={app.description}, account_id={app.account_id}, icon={app.icon}")

    def update_app(self, id: uuid.UUID):
        # 调用服务更新app记录
        app = self.app_service.update_app(id)
        return success_message(f"应用更新成功: id={app.id}, name={app.name}")

    def delete_app(self, id: uuid.UUID):
        # 调用服务删除app记录
        app = self.app_service.delete_app(id)
        return success_message(f"应用删除成功: id={app.id}, name={app.name}")


    def completion(self):
        # 聊天接口：解析 JSON body，交给 FlaskForm 校验
        data = request.get_json(silent=True) or {}
        req = CompletionReq(formdata=MultiDict(data))
        if not req.validate():
            return validation_error_json(req.errors)
        query = req.query.data
        prompt = ChatPromptTemplate.from_template("{query}")
        llm = ChatDeepSeek(model="deepseek-chat")
        parser = StrOutputParser()

        print(f"用户输入: {query}")

        # 3. 构建模型
        chain = prompt | llm | parser
       
        

        return success_json({"content": chain.invoke({"query":query})})
    @classmethod
    def _load_memory_variables(cls,input: Dict[str,Any],config: RunnableConfig) -> Dict[str,Any]:
        """加载记忆变量信息"""
        #1.从conifg中获取configurable
        configurable  =config.get("configurable",{})
        configurable_memory = configurable.get("memory",None)
        if configurable_memory is not None and  isinstance(configurable_memory,BaseMemory):
            return configurable_memory.load_memory_variables(input)
        
        return {"history":[]}

    @classmethod
    def _save_context(cls, run_obj: Run, config: RunnableConfig) -> None:
        """存储对应的上下文信息到记忆实体"""
        configurable = config.get("configurable", {})
        configurable_memory = configurable.get("memory", None)
        if configurable_memory is not None and isinstance(configurable_memory, BaseMemory):
            # 监听器拿到的 run_obj.inputs 形如 {"query": ...}、run_obj.outputs 形如 {"output": ...},
            # 正好对应 memory 的 input_key / output_key,所以原样丢给 save_context 就行
            configurable_memory.save_context(run_obj.inputs, run_obj.outputs)
    def debug(self, app_id: uuid.UUID):
        """调试聊天接口(带窗口记忆)"""
        # 1. 提取从接口中获取的输入，POST
        req = CompletionReq()
        if not req.validate():
            return validation_error_json(req.errors)

        # 2. 创建 prompt 与记忆
        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个强大的聊天机器人，能根据用户的提问回复对应的问题"),
            MessagesPlaceholder("history"),
            ("human", "{query}"),
        ])

        memory = ConversationBufferWindowMemory(
            k=3,
            input_key="query",
            output_key="output",
            return_messages=True,
            chat_memory=FileChatMessageHistory("./storage/memory/chat_history.txt"),
        )

        # 3. 创建 llm(原课程是 ChatOpenAI,本机没配 OpenAI 的 key,统一换成 DeepSeek)
        llm = ChatDeepSeek(model="deepseek-chat")

        # 4. 创建链应用
        # 同步 invoke 用 with_listeners;with_alisteners 是异步版,会给回调 await,
        # 同步函数套进去会报 "object NoneType can't be used in 'await' expression"
        chain = (RunnablePassthrough.assign(
            history=RunnableLambda(self._load_memory_variables) | itemgetter("history")
        ) | prompt | llm | StrOutputParser()).with_listeners(on_end=self._save_context)

        # 5. 调用链生成内容
        # 记忆的读取(_load_memory_variables)与写入(_save_context)都由链自己经监听器完成,
        # 所以这里不用再手动 memory.save_context(...),否则每轮会重复存一遍
        chain_input = {"query": req.query.data}
        content = chain.invoke(chain_input, config={"configurable": {"memory": memory}})

        return success_json({"content": content})
