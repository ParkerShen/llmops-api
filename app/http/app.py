'''
Date: 2026-08-28 15:43:00
Author: parker
FilePath: app.py
'''
from dotenv import load_dotenv
from injector import Injector

from internal.router import Router
from internal.server import Http
from config import Config

load_dotenv()

injector = Injector()

app = Http(__name__, config=Config(), router=injector.get(Router))

if __name__ == "__main__":
    app.run(debug=True)