'''
Date: 2026-09-01 11:05:50
Author: parker
FilePath: http_code.py
'''
from enum import Enum

class HttpCode(Enum):
    SUCCESS = "success"
    FAIL="fail" #失败
    UNAUTHORIZED = "unauthorized" #未授权
    FORBIDDEN = "forbidden" #无权限
    NOT_FOUND = "not_found" #未找到
    VALIDATION_ERROR = "validation_error" #验证错误
   