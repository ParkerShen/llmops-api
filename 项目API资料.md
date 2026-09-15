---
title: llmops-api
language_tabs:
  - shell: Shell
  - http: HTTP
  - javascript: JavaScript
  - ruby: Ruby
  - python: Python
  - php: PHP
  - java: Java
  - go: Go
toc_footers: []
includes: []
search: true
code_clipboard: true
highlight_theme: darkula
headingLevel: 2
generator: "@tarslib/widdershins v4.0.30"

---

# llmops-api

Base URLs:

# Authentication

# Default

## POST 创建app

POST /app

> 返回示例

> 200 Response

```json
{
  "code": "success",
  "data": null,
  "message": "\u5e94\u7528\u521b\u5efa\u6210\u529f, id={app.id}"
}

```

### 返回结果

|状态码|状态码含义|说明|数据模型|
|---|---|---|---|
|200|[OK](https://tools.ietf.org/html/rfc7231#section-6.3.1)|none|Inline|

### 返回数据结构

## GET 获取详情

GET /app/{id}

### 请求参数

|名称|位置|类型|必选|说明|
|---|---|---|---|---|
|id|path|string| 是 |none|
|id|query|string| 否 |none|

> 返回示例

> 200 Response

```json
<!doctype html>
<html lang=en>
  <head>
    <title>werkzeug.exceptions.NotFound: 404 Not Found: The requested URL was not found on the server. If you entered the URL manually please check your spelling and try again.
 // Werkzeug Debugger</title>
    <link rel="stylesheet" href="?__debugger__=yes&amp;cmd=resource&amp;f=style.css">
    <link rel="shortcut icon"
        href="?__debugger__=yes&amp;cmd=resource&amp;f=console.png">
    <script src="?__debugger__=yes&amp;cmd=resource&amp;f=debugger.js"></script>
    <script>
      var CONSOLE_MODE = false,
          EVALEX = true,
          EVALEX_TRUSTED = false,
          SECRET = "iF3XkFl7dF8Clq7Q9P8w";
    </script>
  </head>
  <body style="background-color: #fff">
    <div class="debugger">
<h1>NotFound</h1>
<div class="detail">
  <p class="errormsg">werkzeug.exceptions.NotFound: 404 Not Found: The requested URL was not found on the server. If you entered the URL manually please check your spelling and try again.
</p>
</div>
<h2 class="traceback">Traceback <em>(most recent call last)</em></h2>
<div class="traceback">
  <h3></h3>
  <ul><li><div class="frame" id="frame-1729670806128">
  <h4>File <cite class="filename">"c:\Users\96082\Desktop\code\llmops\llmops-api\.venv\lib\site-packages\flask\app.py"</cite>,
      line <em class="line">1536</em>,
      in <code class="function">__call__</code></h4>
  <div class="source library"><pre class="line before"><span class="ws">    </span>) -&gt; cabc.Iterable[bytes]:</pre>
<pre class="line before"><span class="ws">        </span>&#34;&#34;&#34;The WSGI server calls the Flask application object as the</pre>
<pre class="line before"><span class="ws">        </span>WSGI application. This calls :meth:`wsgi_app`, which can be</pre>
<pre class="line before"><span class="ws">        </span>wrapped to apply middleware.</pre>
<pre class="line before"><span class="ws">        </span>&#34;&#34;&#34;</pre>
<pre class="line current"><span class="ws">        </span>return self.wsgi_app(environ, start_response)</pre></div>
</div>

<li><div class="frame" id="frame-1729670806352">
  <h4>File <cite class="filename">"c:\Users\96082\Desktop\code\llmops\llmops-api\.venv\lib\site-packages\flask\app.py"</cite>,
      line <em class="line">1514</em>,
      in <code class="function">wsgi_app</code></h4>
  <div class="source library"><pre class="line before"><span class="ws">            </span>try:</pre>
<pre class="line before"><span class="ws">                </span>ctx.push()</pre>
<pre class="line before"><span class="ws">                </span>response = self.full_dispatch_request()</pre>
<pre class="line before"><span class="ws">            </span>except Exception as e:</pre>
<pre class="line before"><span class="ws">                </span>error = e</pre>
<pre class="line current"><span class="ws">                </span>response = self.handle_exception(e)</pre>
<pre class="line after"><span class="ws">            </span>except:</pre>
<pre class="line after"><span class="ws">                </span>error = sys.exc_info()[1]</pre>
<pre class="line after"><span class="ws">                </span>raise</pre>
<pre class="line after"><span class="ws">            </span>return response(environ, start_response)</pre>
<pre class="line after"><span class="ws">        </span>finally:</pre></div>
</div>

<li><div class="frame" id="frame-1729670805008">
  <h4>File <cite class="filename">"c:\Users\96082\Desktop\code\llmops\llmops-api\.venv\lib\site-packages\flask\app.py"</cite>,
      line <em class="line">1511</em>,
      in <code class="function">wsgi_app</code></h4>
  <div class="source library"><pre class="line before"><span class="ws">        </span>ctx = self.request_context(environ)</pre>
<pre class="line before"><span class="ws">        </span>error: BaseException | None = None</pre>
<pre class="line before"><span class="ws">        </span>try:</pre>
<pre class="line before"><span class="ws">            </span>try:</pre>
<pre class="line before"><span class="ws">                </span>ctx.push()</pre>
<pre class="line current"><span class="ws">                </span>response = self.full_dispatch_request()</pre>
<pre class="line after"><span class="ws">            </span>except Exception as e:</pre>
<pre class="line after"><span class="ws">                </span>error = e</pre>
<pre class="line after"><span class="ws">                </span>response = self.handle_exception(e)</pre>
<pre class="line after"><span class="ws">            </span>except:</pre>
<pre class="line after"><span class="ws">                </span>error = sys.exc_info()[1]</pre></div>
</div>

<li><div class="frame" id="frame-1729670806576">
  <h4>File <cite class="filename">"c:\Users\96082\Desktop\code\llmops\llmops-api\.venv\lib\site-packages\flask\app.py"</cite>,
      line <em class="line">919</em>,
      in <code class="function">full_dispatch_request</code></h4>
  <div class="source library"><pre class="line before"><span class="ws">            </span>request_started.send(self, _async_wrapper=self.ensure_sync)</pre>
<pre class="line before"><span class="ws">            </span>rv = self.preprocess_request()</pre>
<pre class="line before"><span class="ws">            </span>if rv is None:</pre>
<pre class="line before"><span class="ws">                </span>rv = self.dispatch_request()</pre>
<pre class="line before"><span class="ws">        </span>except Exception as e:</pre>
<pre class="line current"><span class="ws">            </span>rv = self.handle_user_exception(e)</pre>
<pre class="line after"><span class="ws">        </span>return self.finalize_request(rv)</pre>
<pre class="line after"><span class="ws"></span> </pre>
<pre class="line after"><span class="ws">    </span>def finalize_request(</pre>
<pre class="line after"><span class="ws">        </span>self,</pre>
<pre class="line after"><span class="ws">        </span>rv: ft.ResponseReturnValue | HTTPException,</pre></div>
</div>

<li><div class="frame" id="frame-1729670806240">
  <h4>File <cite class="filename">"c:\Users\96082\Desktop\code\llmops\llmops-api\.venv\lib\site-packages\flask\app.py"</cite>,
      line <em class="line">802</em>,
      in <code class="function">handle_user_exception</code></h4>
  <div class="source library"><pre class="line before"><span class="ws">            </span>self.debug or self.config[&#34;TRAP_BAD_REQUEST_ERRORS&#34;]</pre>
<pre class="line before"><span class="ws">        </span>):</pre>
<pre class="line before"><span class="ws">            </span>e.show_exception = True</pre>
<pre class="line before"><span class="ws"></span> </pre>
<pre class="line before"><span class="ws">        </span>if isinstance(e, HTTPException) and not self.trap_http_exception(e):</pre>
<pre class="line current"><span class="ws">            </span>return self.handle_http_exception(e)</pre>
<pre class="line after"><span class="ws"></span> </pre>
<pre class="line after"><span class="ws">        </span>handler = self._find_error_handler(e, request.blueprints)</pre>
<pre class="line after"><span class="ws"></span> </pre>
<pre class="line after"><span class="ws">        </span>if handler is None:</pre>
<pre class="line after"><span class="ws">            </span>raise</pre></div>
</div>

<li><div class="frame" id="frame-1729670806800">
  <h4>File <cite class="filename">"c:\Users\96082\Desktop\code\llmops\llmops-api\.venv\lib\site-packages\flask\app.py"</cite>,
      line <em class="line">777</em>,
      in <code class="function">handle_http_exception</code></h4>
  <div class="source library"><pre class="line before"><span class="ws">            </span>return e</pre>
<pre class="line before"><span class="ws"></span> </pre>
<pre class="line before"><span class="ws">        </span>handler = self._find_error_handler(e, request.blueprints)</pre>
<pre class="line before"><span class="ws">        </span>if handler is None:</pre>
<pre class="line before"><span class="ws">            </span>return e</pre>
<pre class="line current"><span class="ws">        </span>return self.ensure_sync(handler)(e)  # type: ignore[no-any-return]</pre>
<pre class="line after"><span class="ws"></span> </pre>
<pre class="line after"><span class="ws">    </span>def handle_user_exception(</pre>
<pre class="line after"><span class="ws">        </span>self, e: Exception</pre>
<pre class="line after"><span class="ws">    </span>) -&gt; HTTPException | ft.ResponseReturnValue:</pre>
<pre class="line after"><span class="ws">        </span>&#34;&#34;&#34;This method is called whenever an exception occurs that</pre></div>
</div>

<li><div class="frame" id="frame-1729670806912">
  <h4>File <cite class="filename">"C:\Users\96082\Desktop\code\llmops\llmops-api\internal\server\http.py"</cite>,
      line <em class="line">60</em>,
      in <code class="function">_register_error_handler</code></h4>
  <div class="source "><pre class="line before"><span class="ws">            </span>message = str(error)</pre>
<pre class="line before"><span class="ws">            </span>data = {}</pre>
<pre class="line before"><span class="ws"></span> </pre>
<pre class="line before"><span class="ws">        </span>if self.debug or os.getenv(&#34;FLASK_ENV&#34;) == &#34;development&#34;:</pre>
<pre class="line before"><span class="ws">            </span># 2.如果是开发环境，打印异常堆栈信息</pre>
<pre class="line current"><span class="ws">            </span>raise error</pre>
<pre class="line after"><span class="ws">        </span>else:</pre>
<pre class="line after"><span class="ws">            </span># 3.如果是生产环境，返回异常信息给前端</pre>
<pre class="line after"><span class="ws">            </span>return json(Response(code=code, message=message, data=data))</pre></div>
</div>

<li><div class="frame" id="frame-1729670807024">
  <h4>File <cite class="filename">"c:\Users\96082\Desktop\code\llmops\llmops-api\.venv\lib\site-packages\flask\app.py"</cite>,
      line <em class="line">917</em>,
      in <code class="function">full_dispatch_request</code></h4>
  <div class="source library"><pre class="line before"><span class="ws"></span> </pre>
<pre class="line before"><span class="ws">        </span>try:</pre>
<pre class="line before"><span class="ws">            </span>request_started.send(self, _async_wrapper=self.ensure_sync)</pre>
<pre class="line before"><span class="ws">            </span>rv = self.preprocess_request()</pre>
<pre class="line before"><span class="ws">            </span>if rv is None:</pre>
<pre class="line current"><span class="ws">                </span>rv = self.dispatch_request()</pre>
<pre class="line after"><span class="ws">        </span>except Exception as e:</pre>
<pre class="line after"><span class="ws">            </span>rv = self.handle_user_exception(e)</pre>
<pre class="line after"><span class="ws">        </span>return self.finalize_request(rv)</pre>
<pre class="line after"><span class="ws"></span> </pre>
<pre class="line after"><span class="ws">    </span>def finalize_request(</pre></div>
</div>

<li><div class="frame" id="frame-1729670807136">
  <h4>File <cite class="filename">"c:\Users\96082\Desktop\code\llmops\llmops-api\.venv\lib\site-packages\flask\app.py"</cite>,
      line <em class="line">891</em>,
      in <code class="function">dispatch_request</code></h4>
  <div class="source library"><pre class="line before"><span class="ws">           </span>This no longer does the exception handling, this code was</pre>
<pre class="line before"><span class="ws">           </span>moved to the new :meth:`full_dispatch_request`.</pre>
<pre class="line before"><span class="ws">        </span>&#34;&#34;&#34;</pre>
<pre class="line before"><span class="ws">        </span>req = request_ctx.request</pre>
<pre class="line before"><span class="ws">        </span>if req.routing_exception is not None:</pre>
<pre class="line current"><span class="ws">            </span>self.raise_routing_exception(req)</pre>
<pre class="line after"><span class="ws">        </span>rule: Rule = req.url_rule  # type: ignore[assignment]</pre>
<pre class="line after"><span class="ws">        </span># if we provide automatic options for this URL and the</pre>
<pre class="line after"><span class="ws">        </span># request came with the OPTIONS method, reply automatically</pre>
<pre class="line after"><span class="ws">        </span>if (</pre>
<pre class="line after"><span class="ws">            </span>getattr(rule, &#34;provide_automatic_options&#34;, False)</pre></div>
</div>

<li><div class="frame" id="frame-1729670807248">
  <h4>File <cite class="filename">"c:\Users\96082\Desktop\code\llmops\llmops-api\.venv\lib\site-packages\flask\app.py"</cite>,
      line <em class="line">500</em>,
      in <code class="function">raise_routing_exception</code></h4>
  <div class="source library"><pre class="line before"><span class="ws">            </span>not self.debug</pre>
<pre class="line before"><span class="ws">            </span>or not isinstance(request.routing_exception, RequestRedirect)</pre>
<pre class="line before"><span class="ws">            </span>or request.routing_exception.code in {307, 308}</pre>
<pre class="line before"><span class="ws">            </span>or request.method in {&#34;GET&#34;, &#34;HEAD&#34;, &#34;OPTIONS&#34;}</pre>
<pre class="line before"><span class="ws">        </span>):</pre>
<pre class="line current"><span class="ws">            </span>raise request.routing_exception  # type: ignore[misc]</pre>
<pre class="line after"><span class="ws"></span> </pre>
<pre class="line after"><span class="ws">        </span>from .debughelpers import FormDataRoutingRedirect</pre>
<pre class="line after"><span class="ws"></span> </pre>
<pre class="line after"><span class="ws">        </span>raise FormDataRoutingRedirect(request)</pre>
<pre class="line after"><span class="ws"></span> </pre></div>
</div>

<li><div class="frame" id="frame-1729670807360">
  <h4>File <cite class="filename">"c:\Users\96082\Desktop\code\llmops\llmops-api\.venv\lib\site-packages\flask\ctx.py"</cite>,
      line <em class="line">362</em>,
      in <code class="function">match_request</code></h4>
  <div class="source library"><pre class="line before"><span class="ws">    </span>def match_request(self) -&gt; None:</pre>
<pre class="line before"><span class="ws">        </span>&#34;&#34;&#34;Can be overridden by a subclass to hook into the matching</pre>
<pre class="line before"><span class="ws">        </span>of the request.</pre>
<pre class="line before"><span class="ws">        </span>&#34;&#34;&#34;</pre>
<pre class="line before"><span class="ws">        </span>try:</pre>
<pre class="line current"><span class="ws">            </span>result = self.url_adapter.match(return_rule=True)  # type: ignore</pre>
<pre class="line after"><span class="ws">            </span>self.request.url_rule, self.request.view_args = result  # type: ignore</pre>
<pre class="line after"><span class="ws">        </span>except HTTPException as e:</pre>
<pre class="line after"><span class="ws">            </span>self.request.routing_exception = e</pre>
<pre class="line after"><span class="ws"></span> </pre>
<pre class="line after"><span class="ws">    </span>@property</pre></div>
</div>

<li><div class="frame" id="frame-1729670971456">
  <h4>File <cite class="filename">"c:\Users\96082\Desktop\code\llmops\llmops-api\.venv\lib\site-packages\werkzeug\routing\map.py"</cite>,
      line <em class="line">629</em>,
      in <code class="function">match</code></h4>
  <div class="source library"><pre class="line before"><span class="ws">                </span>raise MethodNotAllowed(valid_methods=list(e.have_match_for)) from None</pre>
<pre class="line before"><span class="ws"></span> </pre>
<pre class="line before"><span class="ws">            </span>if e.websocket_mismatch:</pre>
<pre class="line before"><span class="ws">                </span>raise WebsocketMismatch() from None</pre>
<pre class="line before"><span class="ws"></span> </pre>
<pre class="line current"><span class="ws">            </span>raise NotFound() from None</pre>
<pre class="line after"><span class="ws">        </span>else:</pre>
<pre class="line after"><span class="ws">            </span>rule, rv = result</pre>
<pre class="line after"><span class="ws"></span> </pre>
<pre class="line after"><span class="ws">            </span>if self.map.redirect_defaults:</pre>
<pre class="line after"><span class="ws">                </span>redirect_url = self.get_default_redirect(rule, method, rv, query_args)</pre></div>
</div>
</ul>
  <blockquote>werkzeug.exceptions.NotFound: 404 Not Found: The requested URL was not found on the server. If you entered the URL manually please check your spelling and try again.
</blockquote>
</div>

<div class="plain">
    <p>
      This is the Copy/Paste friendly version of the traceback.
    </p>
    <textarea cols="50" rows="10" name="code" readonly>Traceback (most recent call last):
  File &#34;c:\Users\96082\Desktop\code\llmops\llmops-api\.venv\lib\site-packages\flask\app.py&#34;, line 1536, in __call__
    return self.wsgi_app(environ, start_response)
  File &#34;c:\Users\96082\Desktop\code\llmops\llmops-api\.venv\lib\site-packages\flask\app.py&#34;, line 1514, in wsgi_app
    response = self.handle_exception(e)
  File &#34;c:\Users\96082\Desktop\code\llmops\llmops-api\.venv\lib\site-packages\flask\app.py&#34;, line 1511, in wsgi_app
    response = self.full_dispatch_request()
  File &#34;c:\Users\96082\Desktop\code\llmops\llmops-api\.venv\lib\site-packages\flask\app.py&#34;, line 919, in full_dispatch_request
    rv = self.handle_user_exception(e)
  File &#34;c:\Users\96082\Desktop\code\llmops\llmops-api\.venv\lib\site-packages\flask\app.py&#34;, line 802, in handle_user_exception
    return self.handle_http_exception(e)
  File &#34;c:\Users\96082\Desktop\code\llmops\llmops-api\.venv\lib\site-packages\flask\app.py&#34;, line 777, in handle_http_exception
    return self.ensure_sync(handler)(e)  # type: ignore[no-any-return]
  File &#34;C:\Users\96082\Desktop\code\llmops\llmops-api\internal\server\http.py&#34;, line 60, in _register_error_handler
    raise error
  File &#34;c:\Users\96082\Desktop\code\llmops\llmops-api\.venv\lib\site-packages\flask\app.py&#34;, line 917, in full_dispatch_request
    rv = self.dispatch_request()
  File &#34;c:\Users\96082\Desktop\code\llmops\llmops-api\.venv\lib\site-packages\flask\app.py&#34;, line 891, in dispatch_request
    self.raise_routing_exception(req)
  File &#34;c:\Users\96082\Desktop\code\llmops\llmops-api\.venv\lib\site-packages\flask\app.py&#34;, line 500, in raise_routing_exception
    raise request.routing_exception  # type: ignore[misc]
  File &#34;c:\Users\96082\Desktop\code\llmops\llmops-api\.venv\lib\site-packages\flask\ctx.py&#34;, line 362, in match_request
    result = self.url_adapter.match(return_rule=True)  # type: ignore
  File &#34;c:\Users\96082\Desktop\code\llmops\llmops-api\.venv\lib\site-packages\werkzeug\routing\map.py&#34;, line 629, in match
    raise NotFound() from None
werkzeug.exceptions.NotFound: 404 Not Found: The requested URL was not found on the server. If you entered the URL manually please check your spelling and try again.
</textarea>
</div>
<div class="explanation">
  The debugger caught an exception in your WSGI application.  You can now
  look at the traceback which led to the error.  <span class="nojavascript">
  If you enable JavaScript you can also use additional features such as code
  execution (if the evalex feature is enabled), automatic pasting of the
  exceptions and much more.</span>
</div>
      <div class="footer">
        Brought to you by <strong class="arthur">DON'T PANIC</strong>, your
        friendly Werkzeug powered traceback interpreter.
      </div>
    </div>

    <div class="pin-prompt">
      <div class="inner">
        <h3>Console Locked</h3>
        <p>
          The console is locked and needs to be unlocked by entering the PIN.
          You can find the PIN printed out on the standard output of your
          shell that runs the server.
        <form>
          <p>PIN:
            <input type=text name=pin size=14>
            <input type=submit name=btn value="Confirm Pin">
        </form>
      </div>
    </div>
  </body>
</html>

<!--

Traceback (most recent call last):
  File "c:\Users\96082\Desktop\code\llmops\llmops-api\.venv\lib\site-packages\flask\app.py", line 1536, in __call__
    return self.wsgi_app(environ, start_response)
  File "c:\Users\96082\Desktop\code\llmops\llmops-api\.venv\lib\site-packages\flask\app.py", line 1514, in wsgi_app
    response = self.handle_exception(e)
  File "c:\Users\96082\Desktop\code\llmops\llmops-api\.venv\lib\site-packages\flask\app.py", line 1511, in wsgi_app
    response = self.full_dispatch_request()
  File "c:\Users\96082\Desktop\code\llmops\llmops-api\.venv\lib\site-packages\flask\app.py", line 919, in full_dispatch_request
    rv = self.handle_user_exception(e)
  File "c:\Users\96082\Desktop\code\llmops\llmops-api\.venv\lib\site-packages\flask\app.py", line 802, in handle_user_exception
    return self.handle_http_exception(e)
  File "c:\Users\96082\Desktop\code\llmops\llmops-api\.venv\lib\site-packages\flask\app.py", line 777, in handle_http_exception
    return self.ensure_sync(handler)(e)  # type: ignore[no-any-return]
  File "C:\Users\96082\Desktop\code\llmops\llmops-api\internal\server\http.py", line 60, in _register_error_handler
    raise error
  File "c:\Users\96082\Desktop\code\llmops\llmops-api\.venv\lib\site-packages\flask\app.py", line 917, in full_dispatch_request
    rv = self.dispatch_request()
  File "c:\Users\96082\Desktop\code\llmops\llmops-api\.venv\lib\site-packages\flask\app.py", line 891, in dispatch_request
    self.raise_routing_exception(req)
  File "c:\Users\96082\Desktop\code\llmops\llmops-api\.venv\lib\site-packages\flask\app.py", line 500, in raise_routing_exception
    raise request.routing_exception  # type: ignore[misc]
  File "c:\Users\96082\Desktop\code\llmops\llmops-api\.venv\lib\site-packages\flask\ctx.py", line 362, in match_request
    result = self.url_adapter.match(return_rule=True)  # type: ignore
  File "c:\Users\96082\Desktop\code\llmops\llmops-api\.venv\lib\site-packages\werkzeug\routing\map.py", line 629, in match
    raise NotFound() from None
werkzeug.exceptions.NotFound: 404 Not Found: The requested URL was not found on the server. If you entered the URL manually please check your spelling and try again.

-->

```

### 返回结果

|状态码|状态码含义|说明|数据模型|
|---|---|---|---|
|200|[OK](https://tools.ietf.org/html/rfc7231#section-6.3.1)|none|Inline|

### 返回数据结构

## POST 修改

POST /app/9abe6e36-fa3b-44b9-a0e1-e91d49701397

> 返回示例

> 200 Response

```json
{
  "code": "success",
  "data": null,
  "message": "\u5e94\u7528\u66f4\u65b0\u6210\u529f: id=9abe6e36-fa3b-44b9-a0e1-e91d49701397, name=\u5e55\u5ba2\u673a\u5668\u4eba"
}

```

### 返回结果

|状态码|状态码含义|说明|数据模型|
|---|---|---|---|
|200|[OK](https://tools.ietf.org/html/rfc7231#section-6.3.1)|none|Inline|

### 返回数据结构

## POST 删除

POST /app/{id}/delete

### 请求参数

|名称|位置|类型|必选|说明|
|---|---|---|---|---|
|id|path|string| 是 |none|

> 返回示例

> 200 Response

```json
{
  "code": "success",
  "data": null,
  "message": "\u5e94\u7528\u5220\u9664\u6210\u529f: id=ccb1dca4-8b3d-4772-b847-9732713c4da5, name=\u6d4b\u8bd5\u5e94\u7528"
}

```

### 返回结果

|状态码|状态码含义|说明|数据模型|
|---|---|---|---|
|200|[OK](https://tools.ietf.org/html/rfc7231#section-6.3.1)|none|Inline|

### 返回数据结构

# 数据模型

