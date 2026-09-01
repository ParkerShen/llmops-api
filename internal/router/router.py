'''
Date: 2026-08-27 15:46:19
Author: parker
FilePath: router.py
'''
from injector import inject
from flask import Flask, Blueprint
from internal.handler import AppHandler

@inject
class Router:
    """路由"""
    app_handler: AppHandler

    def __init__(self, app_handler: AppHandler):
        self.app_handler = app_handler

    def register_routes(self, app: Flask):
        """注册路由"""
        #创建一个蓝图
        bp = Blueprint("llmops", __name__, url_prefix="" )
        
        # 2.将url于对应的控制器方法做绑定
        # bp.add_url_rule("/ping", view_func=self.app_handler.ping)

        bp.add_url_rule("/app/completion", methods=["POST"],  view_func=self.app_handler.completion)

        # 3.将蓝图注册到Flask应用中
        app.register_blueprint(bp)