"""实时 MuJoCo 逐帧播放器：按住方向键播放/回退，松手暂停。

为什么不用 ``mujoco.viewer.launch_passive``
--------------------------------------------
它的 ``key_callback`` 只在 **KEYDOWN** 时触发，拿不到松手事件，因此无法实现
"按住播放、松手暂停"。这里直接用 GLFW 建窗口并注册 ``set_key_callback``，
可以拿到 ``PRESS`` / ``REPEAT`` / ``RELEASE`` 三种动作，从而精确维护按键的
按住状态。相机的旋转/平移/缩放交互按 MuJoCo 官方 simulate 的习惯实现。

按键
----
====================  ========================================
右方向键（按住）        正向播放，松手暂停
左方向键（按住）        反向回退，松手暂停
空格                   切换自动播放
``.`` / ``,``          单步前进 / 后退
Home / End             跳到首帧 / 末帧
``[`` / ``]``          播放速度减半 / 加倍
R                      回到第 0 帧
T                      切换相机跟随
P                      切换对应点显示
Esc / Q                退出
====================  ========================================
"""

from __future__ import annotations

from typing import Callable

import glfw
import mujoco
import numpy as np

MarkerFn = Callable[[int], list[tuple[np.ndarray, tuple, float]]]


class FrameScrubber:
    """按帧擦洗（scrub）播放器。

    Args:
        model: 要渲染的 MuJoCo 模型。
        n_frames: 总帧数。
        apply_frame: ``apply_frame(data, k)``，把第 k 帧写进 ``data.qpos``。
        fps: 动作帧率，决定按住方向键时的播放速度。
        markers: 可选的 ``markers(k) -> [(points, rgba, size), ...]``，用于叠加对应点。
        track_body: 相机跟随的 body 名，None 表示不跟随。
        title, width, height: 窗口参数。
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        n_frames: int,
        apply_frame: Callable[[mujoco.MjData, int], None],
        fps: float = 30.0,
        markers: MarkerFn | None = None,
        track_body: str | None = None,
        title: str = "UMR frame scrubber",
        width: int = 1280,
        height: int = 800,
        max_markers: int = 4096,
    ):
        self.model = model
        self.data = mujoco.MjData(model)
        self.n_frames = int(n_frames)
        self.apply_frame = apply_frame
        self.fps = float(fps)
        self.markers = markers
        self.show_markers = markers is not None
        self.track = track_body is not None
        self.track_bid = (
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, track_body)
            if track_body
            else -1
        )

        self.frame = 0
        self.speed = 1.0
        self.autoplay = False
        self.held_forward = False
        self.held_backward = False
        self._accumulator = 0.0

        # --- GLFW 窗口 ---
        if not glfw.init():
            raise RuntimeError("GLFW 初始化失败（需要图形界面环境）")
        # 先隐藏着建，摆好位置再显示。放任窗口管理器摆放的话，多屏环境下它常常落在
        # 某块屏的角落、并且不抢焦点地压在 IDE 底下，看起来就跟"没弹窗"一样。
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        glfw.window_hint(glfw.FOCUS_ON_SHOW, glfw.TRUE)
        self.window = glfw.create_window(width, height, title, None, None)
        glfw.default_window_hints()
        if not self.window:
            glfw.terminate()
            raise RuntimeError("GLFW 创建窗口失败")
        self._place_on_primary(width, height)
        glfw.show_window(self.window)
        glfw.focus_window(self.window)
        glfw.request_window_attention(self.window)  # 窗口管理器拒绝抢焦点时闪任务栏
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)

        x, y = glfw.get_window_pos(self.window)
        print(f'[viewer] 窗口已打开："{title}"  {width}x{height} @ ({x}, {y})')

        # --- MuJoCo 渲染资源 ---
        self.scene = mujoco.MjvScene(model, maxgeom=model.ngeom * 2 + max_markers)
        self.context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
        self.camera = mujoco.MjvCamera()
        self.option = mujoco.MjvOption()
        self.perturb = mujoco.MjvPerturb()
        mujoco.mjv_defaultCamera(self.camera)
        mujoco.mjv_defaultOption(self.option)
        mujoco.mjv_defaultPerturb(self.perturb)
        self.camera.distance = 3.6
        self.camera.elevation = -12.0
        self.camera.azimuth = 130.0

        # --- 鼠标状态 ---
        self._last_x = 0.0
        self._last_y = 0.0
        self._button_left = False
        self._button_right = False
        self._button_middle = False

        glfw.set_key_callback(self.window, self._on_key)
        glfw.set_cursor_pos_callback(self.window, self._on_mouse_move)
        glfw.set_mouse_button_callback(self.window, self._on_mouse_button)
        glfw.set_scroll_callback(self.window, self._on_scroll)

    def _place_on_primary(self, width: int, height: int) -> None:
        """把窗口摆到主显示器工作区中央。"""
        monitor = glfw.get_primary_monitor()
        if not monitor:
            return
        mx, my, mw, mh = glfw.get_monitor_workarea(monitor)
        glfw.set_window_pos(
            self.window, mx + max(0, (mw - width) // 2), my + max(0, (mh - height) // 2)
        )

    # ------------------------------------------------------------------
    # 输入回调
    # ------------------------------------------------------------------
    def _on_key(self, window, key, scancode, action, mods) -> None:
        del window, scancode, mods
        pressed = action in (glfw.PRESS, glfw.REPEAT)

        # 方向键：维护"按住"状态。GLFW 的 RELEASE 是这里的关键——
        # MuJoCo 自带 viewer 拿不到它，所以做不了松手暂停。
        if key == glfw.KEY_RIGHT:
            self.held_forward = pressed
            return
        if key == glfw.KEY_LEFT:
            self.held_backward = pressed
            return

        if action != glfw.PRESS:
            return
        if key in (glfw.KEY_ESCAPE, glfw.KEY_Q):
            glfw.set_window_should_close(self.window, True)
        elif key == glfw.KEY_SPACE:
            self.autoplay = not self.autoplay
        elif key == glfw.KEY_PERIOD:
            self._seek(self.frame + 1)
        elif key == glfw.KEY_COMMA:
            self._seek(self.frame - 1)
        elif key == glfw.KEY_HOME:
            self._seek(0)
        elif key == glfw.KEY_END:
            self._seek(self.n_frames - 1)
        elif key == glfw.KEY_R:
            self._seek(0)
            self.autoplay = False
        elif key == glfw.KEY_LEFT_BRACKET:
            self.speed = max(0.125, self.speed / 2)
        elif key == glfw.KEY_RIGHT_BRACKET:
            self.speed = min(16.0, self.speed * 2)
        elif key == glfw.KEY_T:
            self.track = not self.track
        elif key == glfw.KEY_P and self.markers is not None:
            self.show_markers = not self.show_markers

    def _on_mouse_button(self, window, button, action, mods) -> None:
        del mods
        press = action == glfw.PRESS
        if button == glfw.MOUSE_BUTTON_LEFT:
            self._button_left = press
        elif button == glfw.MOUSE_BUTTON_RIGHT:
            self._button_right = press
        elif button == glfw.MOUSE_BUTTON_MIDDLE:
            self._button_middle = press
        self._last_x, self._last_y = glfw.get_cursor_pos(window)

    def _on_mouse_move(self, window, xpos, ypos) -> None:
        dx, dy = xpos - self._last_x, ypos - self._last_y
        self._last_x, self._last_y = xpos, ypos
        if not (self._button_left or self._button_right or self._button_middle):
            return
        width, height = glfw.get_window_size(window)
        shift = (
            glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
            or glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
        )
        if self._button_right:
            action = mujoco.mjtMouse.mjMOUSE_MOVE_H if shift else mujoco.mjtMouse.mjMOUSE_MOVE_V
        elif self._button_left:
            action = mujoco.mjtMouse.mjMOUSE_ROTATE_H if shift else mujoco.mjtMouse.mjMOUSE_ROTATE_V
        else:
            action = mujoco.mjtMouse.mjMOUSE_ZOOM
        mujoco.mjv_moveCamera(
            self.model, action, dx / height, dy / height, self.scene, self.camera
        )

    def _on_scroll(self, window, xoffset, yoffset) -> None:
        del window, xoffset
        mujoco.mjv_moveCamera(
            self.model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0.0, -0.05 * yoffset,
            self.scene, self.camera,
        )

    # ------------------------------------------------------------------
    def _seek(self, k: int) -> None:
        self.frame = int(np.clip(k, 0, self.n_frames - 1))
        self._accumulator = 0.0

    def _advance(self, dt: float) -> None:
        """按住方向键时按实际时间推进帧号，保证播放速度与 fps 一致。"""
        direction = int(self.held_forward) - int(self.held_backward)
        if direction == 0 and self.autoplay:
            direction = 1
        if direction == 0:
            self._accumulator = 0.0
            return
        self._accumulator += direction * dt * self.fps * self.speed
        step = int(self._accumulator)
        if step:
            self._accumulator -= step
            self.frame = int(np.clip(self.frame + step, 0, self.n_frames - 1))

    def _status(self) -> tuple[str, str]:
        if self.held_forward:
            state = "PLAY >>"
        elif self.held_backward:
            state = "REWIND <<"
        elif self.autoplay:
            state = "AUTOPLAY"
        else:
            state = "PAUSED"
        left = (
            f"{state}\nFrame\nTime\nSpeed\n\n"
            "Right / Left\nSpace\n. / ,\nHome / End\n[ / ]\nT / P\nEsc"
        )
        right = (
            f"\n{self.frame + 1} / {self.n_frames}\n"
            f"{self.frame / self.fps:.2f} s\n{self.speed:g}x\n\n"
            "hold to play / rewind\ntoggle autoplay\nsingle step\nfirst / last frame\n"
            "slower / faster\ntrack / points\nquit"
        )
        return left, right

    def run(self, max_seconds: float | None = None) -> None:
        """进入渲染循环。``max_seconds`` 仅用于冒烟测试时自动退出。"""
        start = last_time = glfw.get_time()
        while not glfw.window_should_close(self.window):
            now = glfw.get_time()
            if max_seconds is not None and now - start > max_seconds:
                break
            dt = now - last_time
            last_time = now
            self._advance(dt)

            self.apply_frame(self.data, self.frame)
            mujoco.mj_forward(self.model, self.data)

            if self.track and self.track_bid >= 0:
                self.camera.lookat[:] = self.data.xpos[self.track_bid]

            width, height = glfw.get_framebuffer_size(self.window)
            viewport = mujoco.MjrRect(0, 0, width, height)
            mujoco.mjv_updateScene(
                self.model, self.data, self.option, self.perturb, self.camera,
                mujoco.mjtCatBit.mjCAT_ALL, self.scene,
            )
            if self.show_markers and self.markers is not None:
                from umr.sim.render import add_point_markers

                for pts, rgba, size in self.markers(self.frame):
                    add_point_markers(self.scene, pts, rgba, size)

            mujoco.mjr_render(viewport, self.scene, self.context)
            left, right = self._status()
            mujoco.mjr_overlay(
                mujoco.mjtFont.mjFONT_NORMAL,
                mujoco.mjtGridPos.mjGRID_TOPLEFT,
                viewport, left, right, self.context,
            )
            glfw.swap_buffers(self.window)
            glfw.poll_events()

        glfw.terminate()
