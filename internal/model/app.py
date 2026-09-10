'''
Date: 2026-09-02 16:35:41
Author: parker
FilePath: \llmops-api\internal\model\app.py
'''
import uuid
from datetime import datetime  # datetime.now() 取当前时间(下面 created_at/updated_at 默认值要用)
from sqlalchemy import Column, String, DateTime,UUID,Text,PrimaryKeyConstraint,Index,text

from internal.extension.database_extension import db

class App(db.Model):
    #AI应用基础模型类
    __tablename__ = "app"
    __table_args__ = (
        PrimaryKeyConstraint('id', name='pk_app_id'),
        Index('idx_app_account_id', 'account_id'),
    )

    id = Column(UUID, nullable=False, primary_key=True, comment="主键",server_default=text("gen_random_uuid()"))
    account_id = Column(UUID, nullable=False, comment="所属账号ID")
    name = Column(String(255), nullable=False,server_default=text("''::character varying"))
    icon = Column(String(255),default="", nullable=False,server_default=text("''::character varying"))
    description = Column(Text,default="", nullable=False,server_default=text("''::text"))
    status = Column(String(255),default="", nullable=False,server_default=text("''::character varying"))
    created_at = Column(DateTime,default=datetime.now, nullable=False,server_default=text("''::CURRENT_TIMESTAMP(0)"))    
    updated_at = Column(DateTime,default=datetime.now,onupdate=datetime.now, nullable=False,server_default=text("''::CURRENT_TIMESTAMP(0)"),server_onupdate=text("''::CURRENT_TIMESTAMP(0)"))