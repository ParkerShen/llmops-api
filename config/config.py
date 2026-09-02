'''
Date: 2026-08-27 15:32:07
Author: parker
FilePath: \llmops-api\config\config.py
'''
import os
from config.default_config import DEFAULT_CONFIG
from typing import Any


def _get_env(key: str) -> Any:
    # 先从环境变量里取,如果没有再从默认配置里取
    return os.getenv(key, DEFAULT_CONFIG.get(key))

def _get_bool_env(key: str) -> bool:
    """读取布尔配置:环境变量/默认值转成 bool
    比如 "true"/"True"/"TRUE" 都算 True,大小写不敏感"""
    value = _get_env(key)          # ① 记得加 (key) 调用函数,而不是把函数本身赋过去
    return value.lower() == "true" if value is not None else False  # ② 方法是小写 .lower()

class Config:
    """应用配置:从环境变量读取(值在 .env 文件里配置)"""

    def __init__(self):
        # 禁用 CSRF 验证(学习阶段方便接口调试)
        self.WTF_CSRF_ENABLED = _get_bool_env("WTF_CSRF_ENABLED")

        # ===== PostgreSQL 数据库连接 =====
        self.SQLALCHEMY_DATABASE_URI = _get_env("SQLALCHEMY_DATABASE_URI")
        self.SQLALCHEMY__ENGINE_OPTIONS = {
            "pool_size": int(_get_env("SQLALCHEMY_POOL_SIZE") or 30),
            "pool_recycle": int(_get_env("SQLALCHEMY_POOL_RECYCLE") or 3600),
        }
        # 是否打印 SQL 语句到控制台(学习阶段方便调试)
        self.SQLALCHEMY_ECHO = _get_bool_env("SQLALCHEMY_ECHO")
