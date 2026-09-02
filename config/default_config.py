'''
Date: 2026-08-27 15:32:49
Author: parker
FilePath: \llmops-api\config\default_config.py
'''
DEFAULT_CONFIG = {
    #wft配置
    "WTF_CSRF_ENABLED": "False",
    # ===== PostgreSQL 数据库连接 =====
    "SQLALCHEMY_DATABASE_URI": "",
    "SQLALCHEMY_POOL_SIZE": 30,
    "SQLALCHEMY_POOL_RECYCLE": 3600,
    "SQLALCHEMY_ECHO": "True",

    # ===== SQLAlchemy 配置 =====
}