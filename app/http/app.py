'''
Date: 2026-08-28 15:43:00
Author: parker
FilePath: app.py
'''
import sys

from dotenv import load_dotenv
from injector import Injector

from internal.router import Router
from internal.server import Http
from config import Config

load_dotenv()

injector = Injector()

app = Http(__name__, config=Config(), router=injector.get(Router))

if __name__ == "__main__":
    # 用 VSCode/debugpy 调试时，禁用 Flask 自带 reloader：
    # 它会在子进程里重启应用，导致 SystemExit: 3 且断点不命中
    under_debugger = "debugpy" in sys.modules
    app.run(debug=True, use_reloader=not under_debugger)