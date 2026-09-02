'''
Date: 2026-08-28 14:52:33
Author: parker
FilePath: \llmops-api\internal\handler\app_handler.py
'''
from dataclasses import dataclass
import os
import uuid

from injector import inject
import requests
from flask import request
from werkzeug.datastructures import MultiDict

from internal.schema.app_schema import CompletionReq

from internal.service import AppService

from pkg.response import success_json, validation_error_json, success_message, not_found_message

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
        print(f"用户输入: {query}")

        # 调用 DeepSeek 官方 API（https://api.deepseek.com）
        resp = requests.post(
            "https://api.deepseek.com/chat/completions",
            headers={
                "Authorization": f"Bearer {os.getenv('DEEPSEEK_API_KEY')}",
                "Content-Type": "application/json",
            },
            json={
                "model": os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
                "messages": [
                    {"role": "system", "content": "请根据用户回复对应信息"},
                    {"role": "user", "content": query},
                ],
            },
        )
        resp.raise_for_status()

        data = resp.json()
        content = data["choices"][0]["message"]["content"]

        return success_json({"content": content})
