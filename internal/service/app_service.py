'''
Date: 2026-09-02 17:48:40
Author: parker
FilePath: \llmops-api\internal\service\app_service.py
Description:
'''

from dataclasses import dataclass
import uuid

from injector import inject
from internal.model import App
from pkg.sqlalchemy import SQLAlchemy

@inject
@dataclass

class AppService:
    #应用服务器逻辑
    db: SQLAlchemy  # 依赖注入的数据库扩展(含 auto_commit)

    def create_app(self) -> App:
        with self.db.auto_commit():
        #创建模型的实体类
            app =App(name="测试应用", description="这是一个测试应用",account_id=uuid.uuid4(),icon="")
            
            #将实体类添加到session会话中
            self.db.session.add(app)
            return app

    def get_app(self,id:uuid.UUID) -> App:
        #根据id获取应用
        app = self.db.session.query(App).get(id)
        return app

    def update_app(self,id:uuid.UUID) -> App:
        with self.db.auto_commit():
        #根据id获取应用
            app = self.db.session.query(App).get(id)
        return app

    def delete_app(self,id:uuid.UUID) -> None:
        with self.db.auto_commit():
            #根据id获取应用
            app = self.get_app(id)
            self.db.session.delete(app)
        
        return app