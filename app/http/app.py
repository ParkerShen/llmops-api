'''
Date: 2026-08-28 15:43:00
Author: parker
FilePath: app.py
'''
from injector import Injector

from internal.router import Router
from internal.server import Http

injector = Injector()

app = Http(__name__, router=injector.get(Router))

if __name__ == "__main__":
    app.run(debug=True)