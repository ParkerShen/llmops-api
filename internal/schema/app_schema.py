'''
Date: 2026-08-31 17:54:18
Author: parker
FilePath: app.schema.py
'''
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired, Length

class CompletionReq(FlaskForm):
    # 基础聊天接口验证
    query = StringField(
        "query",
        validators=[
            DataRequired(message="请输入查询内容"),
            Length(min=1, max=1000, message="查询内容长度应在1到1000个字符之间"),
        ],
    )