'''
Date: 2026-08-27 15:37:51
Author: parker
FilePath: \llmops-api\internal\exception\exception.py
'''

from dataclasses import field
from typing import Any

from pkg.response import HttpCode

class CustomException(Exception):
    code: HttpCode =HttpCode.FAIL
    message: str = ""
    data: Any = field(default_factory=dict)

    def __init__(self, message: str = None, data: Any = None):
        super().__init__()
        self.message = message
        self.data = data

class FailException(CustomException):
    #通用失败异常
    pass

class NotFoundException(CustomException):
    #资源未找到异常
    code: HttpCode = HttpCode.NOT_FOUND

class UnauthorizedException(CustomException):
    #未授权异常
    code: HttpCode = HttpCode.UNAUTHORIZED

class ForbiddenException(CustomException):
    #禁止访问异常,无权限异常
    code: HttpCode = HttpCode.FORBIDDEN

class ValidationErrorException(CustomException):
    #参数验证异常
    code: HttpCode = HttpCode.VALIDATION_ERROR

