'''
Date: 2026-08-28 15:25:35
Author: parker
FilePath: \llmops-api\internal\server\http.py
'''
import os

from flask import Flask
from internal.router import Router
from config import Config
from internal.exception import CustomException
from pkg.response import HttpCode, json, Response

class Http(Flask):
    """HTTP服务"""
    def __init__(self, *args, router: Router, config: Config, **kwargs):
        super().__init__(*args, **kwargs)
        # 1.注册路由
        router.register_routes(self)
        # 2.应用配置
        self.config.from_object(config)

        # 3.注册异常处理器
        self.register_error_handler(Exception, self._register_error_handler)

    def _register_error_handler(self, error: Exception):
        # 1.异常信息是不是我们自定义异常，如果是可以提取出异常信息code 和message
        if isinstance(error, CustomException):
            code = error.code
            message = error.message
            data = error.data if error.data is not None else {}
        else:
            # 非自定义异常，用通用失败信息
            code = HttpCode.FAIL
            message = str(error)
            data = {}

        if self.debug or os.getenv("FLASK_ENV") == "development":
            # 2.如果是开发环境，打印异常堆栈信息
            raise error
        else:
            # 3.如果是生产环境，返回异常信息给前端
            return json(Response(code=code, message=message, data=data))