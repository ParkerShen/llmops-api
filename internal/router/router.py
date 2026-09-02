'''
Date: 2026-08-27 15:46:19
Author: parker
FilePath: \llmops-api\internal\router\router.py
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
        # 创建一个蓝图
        bp = Blueprint("llmops", __name__, url_prefix="")

        # 1. 聊天接口
        bp.add_url_rule("/app/completion", methods=["POST"], view_func=self.app_handler.completion)

        bp.add_url_rule("/app", methods=["POST"], view_func=self.app_handler.create_app)
        bp.add_url_rule("/app/<uuid:id>", methods=["GET"], view_func=self.app_handler.get_app)
        bp.add_url_rule("/app/<uuid:id>", methods=["POST"], view_func=self.app_handler.update_app)
        bp.add_url_rule("/app/<uuid:id>/delete", methods=["POST"], view_func=self.app_handler.delete_app)

        

        # 将蓝图注册到 Flask 应用
        app.register_blueprint(bp)
