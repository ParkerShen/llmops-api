'''
Date: 2026-09-01
Author: parker
FilePath: \llmops-api\internal\extension\database_extension.py
'''
from pkg.sqlalchemy import SQLAlchemy

# 全局唯一的 SQLAlchemy 实例(相当于"数据库操作入口")
# 使用 pkg.sqlalchemy.SQLAlchemy 子类,自带了 auto_commit() 事务上下文
# 所有模型和 handler 都通过这个 db 来增删改查
db = SQLAlchemy()
