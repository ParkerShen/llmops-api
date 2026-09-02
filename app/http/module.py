'''
Date: 2026-09-02 15:22:24
Author: parker
FilePath: \llmops-api\app\http\module.py
'''

from injector import  Binder, Module

from internal.extension.database_extension import db

from pkg.sqlalchemy import SQLAlchemy

#拓展模块依赖注入
class ExtensionModule(Module):
    def configure(self, binder: Binder) -> None:
        # 绑定数据库扩展
        binder.bind(SQLAlchemy, to=db)
      