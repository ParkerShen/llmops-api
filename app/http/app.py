'''
Date: 2026-08-28 15:43:00
Author: parker
FilePath: \llmops-api\app\http\app.py
'''
import sys
import os

# 把项目根目录加进 sys.path,保证直接运行 `python app/http/app.py` 时
# 也能 import 到 internal/config/pkg 等包
# app.py 在 app/http/ 下,往上走两级是 app/,还要再走一级才是项目根目录,共 3 级
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
from flask_migrate import Migrate
from injector import Injector, Binder, Module

from internal.router import Router
from internal.server import Http
from config import Config
from pkg.sqlalchemy import SQLAlchemy

from internal.extension.database_extension import db

from app.http.module import ExtensionModule


load_dotenv()



injector = Injector([ExtensionModule])



app = Http(__name__, config=Config(),db=injector.get(SQLAlchemy),router=injector.get(Router))

# 注册 Flask-Migrate,提供 `flask db init/migrate/upgrade` 等 CLI 命令
# directory 显式指到项目根目录的 migrations/,避免因 flask 导入方式导致位置漂移
migrate = Migrate(app, db, directory=os.path.join(PROJECT_ROOT, "migrations"))

if __name__ == "__main__":
    # 用 VSCode/debugpy 调试时，禁用 Flask 自带 reloader：
    # 它会在子进程里重启应用，导致 SystemExit: 3 且断点不命中
    under_debugger = "debugpy" in sys.modules
    app.run(debug=True, use_reloader=not under_debugger)