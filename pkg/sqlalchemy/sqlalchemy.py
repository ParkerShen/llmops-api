'''
Date: 2026-09-02 19:40:24
Author: parker
FilePath: \llmops-api\pkg\sqlalchemy\sqlalchemy.py
Description: 核心类实现自动提交
'''
from contextlib import contextmanager
from flask_sqlalchemy import SQLAlchemy as _SQLAlchemy

class SQLAlchemy(_SQLAlchemy):
    @contextmanager
    def auto_commit(self):
        try:
            yield
            self.session.commit()
        except Exception as e:
            self.session.rollback()
            raise e
