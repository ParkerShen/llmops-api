'''
Date: 2026-09-01
Author: parker
FilePath: test/test_app_handler.py

本文件是 AppHandler.completion()(聊天接口)的单元测试。
运行方式: .venv\Scripts\python.exe -m pytest test/test_app_handler.py -v -s
'''
import pytest
import requests  # completion() 里用 requests.post 调 DeepSeek API,测试里要把它替换成假的

from app.http.app import app  # 导入真实的 Flask 应用(import 时会执行 app/http/app.py 的初始化逻辑)


class FakeDeepSeekResp:
    """假响应:伪造 DeepSeek API 的返回,避免测试时真的联网调用"""

    def raise_for_status(self):
        # completion() 里会调 resp.raise_for_status(),这里假装"请求没出错"
        pass

    def json(self):
        # completion() 里会调 resp.json() 取 content,这里返回伪造的数据
        return {"choices": [{"message": {"content": "这是模拟的 AI 回复"}}]}


@pytest.fixture
def client():
    """fixture:搭建测试环境——Flask 测试客户端"""
    # 测试客户端不用真的启动服务器,就能像"发 HTTP 请求"一样调用 Flask 路由
    return app.test_client()


@pytest.fixture
def mock_deepseek(monkeypatch):
    """fixture:替换 requests.post,并记录每次调用的参数"""
    calls = []  # 记录"被调用了几次、传了什么参数"

    def fake_post(url, **kwargs):
        # 假 post:把调用参数记下来,返回假响应
        calls.append({"url": url, **kwargs})
        return FakeDeepSeekResp()

    # monkeypatch 是 pytest 自带的 fixture:测试期间把 requests.post 换成 fake_post,
    # 测试结束自动还原成真的,不会污染其他测试
    monkeypatch.setattr(requests, "post", fake_post)
    return calls  # 把记录返回出去,测试里可以断言"确实调用了 API"


@pytest.mark.parametrize("query", ["你好", "介绍一下你自己", "怎么学好 Python？"])
# 参数化:同一个测试逻辑,query 依次取上面 3 个值,就相当于跑了 3 遍
def test_completion_success(client, mock_deepseek, query):
    # client / mock_deepseek 是 fixture 的名字,写进参数列表,pytest 会自动"注入"进来
    # 模拟一次 POST 请求:POST /app/completion,请求体是 {"query": 你好}
    resp = client.post("/app/completion", json={"query": query})

    # —— 下面全是断言:测试的"答案",如果不满足就报错 ——
    assert resp.status_code == 200                                        # 1) HTTP 状态码是 200
    data = resp.get_json()                                                 # 2) 把返回的 JSON 解析成字典
    assert data["code"] == "success"                                       # 3) code 是 success
    assert data["data"]["content"] == "这是模拟的 AI 回复"                  # 4) content 是假响应里的内容

    # 5) 顺便确认:真的调了一次 DeepSeek API,URL 和模型参数都对
    assert len(mock_deepseek) == 1
    assert mock_deepseek[0]["url"] == "https://api.deepseek.com/chat/completions"
    assert mock_deepseek[0]["json"]["model"] == "deepseek-chat"


@pytest.mark.parametrize(
    "payload",
    [
        {},                      # 情况1:完全没传请求体
        {"query": ""},           # 情况2:query 是空字符串
        {"query": "x" * 1001},   # 情况3:query 超过 1000 字符
    ],
)
def test_completion_validation_error(client, payload):
    # 3 种非法输入,都应该被表单校验(CompletionReq)拦下来,返回 validation_error
    resp = client.post("/app/completion", json=payload)

    assert resp.status_code == 200
    data = resp.get_json()
    assert data["code"] == "validation_error"  # 关键断言:返回的是"参数校验失败"
