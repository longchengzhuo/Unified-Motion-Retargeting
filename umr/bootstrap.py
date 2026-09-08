"""解释器启动后、导入第三方包之前需要做的环境修正。

只依赖标准库，且必须在 ``import numpy`` / ``import mujoco`` 之前调用。
"""

from __future__ import annotations

import os
import site
import sys
import warnings


def prepare_runtime(mujoco_gl: str = "glfw") -> None:
    """屏蔽 user site 目录，并给 MuJoCo 定一个默认渲染后端。

    ``~/.local/lib`` 下常年装着一套会遮蔽 conda 环境的 numpy / mujoco。
    ``PYTHONNOUSERSITE`` 只在解释器启动时被 ``site`` 模块读取，在脚本里再设已经
    太晚，所以这里直接把 user site 目录从 ``sys.path`` 里摘掉。
    """
    # virtualenv 会换掉 site 模块，getusersitepackages 未必存在。
    get_user_site = getattr(site, "getusersitepackages", None)
    if site.ENABLE_USER_SITE and get_user_site is not None:
        user_site = get_user_site()
        sys.path[:] = [p for p in sys.path if p != user_site]

    os.environ.setdefault("MUJOCO_GL", mujoco_gl)
    # mink / mujoco / imageio 会刷一堆与结果无关的弃用警告。
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
