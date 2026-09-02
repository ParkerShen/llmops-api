# internal 包:业务核心代码(路由/处理器/模型/服务等)
# 有了 __init__.py,internal 成为普通包,配合 app.py 里对项目根目录的
# sys.path 处理,`from internal.xxx import yyy` 才能被 import 到
