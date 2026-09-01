'''
Date: 2026-09-01 11:11:18
Author: parker
FilePath: \llmops-api\pkg\response\response.py
'''
from dataclasses import field, dataclass
from typing import Any
from enum import Enum
from .http_code import HttpCode
from flask import jsonify


@dataclass
class Response:
    # 基础响应
    code: HttpCode = HttpCode.SUCCESS
    message: str = ""
    data: Any = field(default_factory=dict)  # 使用 default_factory 来创建一个新的字典实例


def json(data: Response):
    # 统一把对象转成 dict 序列化，code 枚举转成字符串
    body = {
        "code": data.code.value if isinstance(data.code, Enum) else data.code,
        "message": data.message,
        "data": data.data,
    }
    return jsonify(body), 200


def message(code: HttpCode = HttpCode.SUCCESS, msg: str = "", data: Any = None):
    # 通用消息返回：自定义 code 和 message，可带 data
    return json(Response(code=code, message=msg, data=data))

def success_message(msg: str = "请求成功", data: Any = None):
    return json(Response(code=HttpCode.SUCCESS, message=msg, data=data))

def fail_message(msg: str = "请求失败", data: Any = None):
    return json(Response(code=HttpCode.FAIL, message=msg, data=data))

def not_found_message(msg: str = "未找到", data: Any = None):
    return json(Response(code=HttpCode.NOT_FOUND, message=msg, data=data))

def unauthorized_message(msg: str = "未授权", data: Any = None):
    return json(Response(code=HttpCode.UNAUTHORIZED, message=msg, data=data))

def forbidden_message(msg: str = "无权限", data: Any = None):
    return json(Response(code=HttpCode.FORBIDDEN, message=msg, data=data))


def success_json(data: Any = None, message: str = "请求成功"):
    return json(Response(code=HttpCode.SUCCESS, message=message, data=data))


def fail_json(message: str = "请求失败", data: Any = None):
    return json(Response(code=HttpCode.FAIL, message=message, data=data))



def validation_error_json(errors: dict = None):
    # 返回验证错误的 JSON 响应
    msg = ""
    if errors:
        first_key = next(iter(errors))
        msgs = errors[first_key]
        if msgs:
            msg = msgs[0]
    return json(Response(code=HttpCode.VALIDATION_ERROR, message=msg, data=errors))
