'''
Date: 2026-08-28 15:25:35
Author: parker
FilePath: http.py
'''
from flask import Flask
from internal.router import Router
from config import Config

class Http(Flask):
    """HTTP服务"""
    def __init__(self, *args, router: Router, config: Config, **kwargs):
        super().__init__(*args, **kwargs)
        # 1.注册路由
        router.register_routes(self)
        # 2.应用配置
        self.config.from_object(config)