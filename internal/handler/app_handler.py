'''
Date: 2026-08-28 14:52:33
Author: parker
FilePath: app_handler.py
'''
import os

import requests
from flask import request
from werkzeug.datastructures import MultiDict

from internal.schema.app_schema import CompletionReq


class AppHandler:
    """应用控制器"""
    def completion(self):
        # 聊天接口：解析 JSON body，交给 FlaskForm 校验
        data = request.get_json(silent=True) or {}
        req = CompletionReq(formdata=MultiDict(data))
        if not req.validate():
            return {"error": req.errors}, 400
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

        return resp.json()["choices"][0]["message"]["content"]
