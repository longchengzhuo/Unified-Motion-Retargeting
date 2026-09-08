"""BVH 解析与前向运动学。

只依赖 numpy/scipy，自带解析器，不引入外部 BVH 库。

坐标系约定
----------
Xsens BVH 使用 Y-up、厘米。本模块把它转换成 MuJoCo 的约定：
X 前、Y 左、Z 上，单位米。

BVH 帧的基向量为 (X=左, Y=上, Z=前)，所以转换矩阵是一个轴的循环置换::

    R = [[0, 0, 1],
         [1, 0, 0],
         [0, 1, 0]]

即 ``new_x = old_z``、``new_y = old_x``、``new_z = old_y``，det(R) = +1。
局部旋转按相似变换 ``R_new = R @ R_old @ R.T`` 转换，使 FK 结果与整体旋转一致。
实际朝向会在 :func:`load_bvh` 中用 T-pose 自动校正（见 ``auto_face_x``）。
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from umr.bodies.skeletons import DEFAULT_HUMAN, Skeleton, get_skeleton

# BVH (X=左, Y=上, Z=前) -> MuJoCo (X=前, Y=左, Z=上)
BVH_TO_MUJOCO = np.array(
    [
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]
)

_CHANNEL_TO_AXIS = {"Xrotation": "X", "Yrotation": "Y", "Zrotation": "Z"}


@dataclass
class BvhData:
    """解析并转换到 MuJoCo 坐标系后的 BVH 动作。

    Attributes:
        names: 长度 J 的关节名列表，按层级（深度优先）顺序。
        parents: 长度 J 的父节点索引，根节点为 -1。
        offsets: (J, 3) 各关节相对父节点的静止偏移，单位米。
        end_sites: {关节索引: (3,) 末端偏移}，单位米。
        local_quat: (T, J, 4) 局部旋转四元数，wxyz。
        root_pos: (T, 3) 根节点全局位置，单位米。
        fps: 帧率。
        source_path: 原始文件路径。
    """

    names: list[str]
    parents: np.ndarray
    offsets: np.ndarray
    end_sites: dict[int, np.ndarray]
    local_quat: np.ndarray
    root_pos: np.ndarray
    fps: float
    source_path: str

    @property
    def num_joints(self) -> int:
        return len(self.names)

    @property
    def num_frames(self) -> int:
        return self.local_quat.shape[0]

    def index(self, name: str) -> int:
        return self.names.index(name)


def _parse_hierarchy(lines: list[str]) -> tuple[list[str], list[int], list[np.ndarray], dict[int, np.ndarray], list[list[str]], int]:
    names: list[str] = []
    parents: list[int] = []
    offsets: list[np.ndarray] = []
    channels: list[list[str]] = []
    end_sites: dict[int, np.ndarray] = {}

    stack: list[int] = []
    current = -1
    in_end_site = False
    idx = 0

    for idx, raw in enumerate(lines):
        line = raw.strip()
        if not line:
            continue
        if line == "MOTION":
            break

        if line.startswith("ROOT") or line.startswith("JOINT"):
            name = line.split(None, 1)[1].strip()
            names.append(name)
            parents.append(current)
            offsets.append(np.zeros(3))
            channels.append([])
            current = len(names) - 1
            in_end_site = False
        elif line.startswith("End Site"):
            in_end_site = True
        elif line.startswith("OFFSET"):
            vals = np.array([float(v) for v in line.split()[1:4]])
            if in_end_site:
                end_sites[current] = vals
            else:
                offsets[current] = vals
        elif line.startswith("CHANNELS"):
            parts = line.split()
            channels[current] = parts[2:]
        elif line == "{":
            if not in_end_site:
                stack.append(current)
        elif line == "}":
            if in_end_site:
                in_end_site = False
            else:
                stack.pop()
                current = stack[-1] if stack else -1

    return names, parents, offsets, end_sites, channels, idx


def _euler_order(rot_channels: list[str]) -> str:
    """BVH 通道顺序 -> scipy 的内旋欧拉序（大写表示 intrinsic）。

    ``CHANNELS 3 Yrotation Xrotation Zrotation`` 意味着 R = Ry @ Rx @ Rz，
    对应 scipy 的 ``from_euler("YXZ", ..., degrees=True)``。
    """
    return "".join(_CHANNEL_TO_AXIS[c] for c in rot_channels)


def read_bvh_raw(
    path: str | Path, strip_prefix: bool = False
) -> tuple[list[str], np.ndarray, np.ndarray, dict[int, np.ndarray], np.ndarray, np.ndarray, float]:
    """读取 BVH，返回 **未做坐标变换** 的原始数据（BVH 自身单位与朝向）。

    Args:
        path: BVH 文件路径。
        strip_prefix: 去掉导出器加在关节名前的前缀（``character1_Hips`` -> ``Hips``），
            取最后一个下划线之后的部分。
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.read().splitlines()

    names, parents, offsets, end_sites, channels, motion_idx = _parse_hierarchy(lines)
    if strip_prefix:
        names = [n.split("_")[-1] for n in names]
    n_joints = len(names)

    # --- MOTION 头 ---
    frames = None
    frame_time = None
    data_start = None
    for i in range(motion_idx, min(motion_idx + 10, len(lines))):
        line = lines[i].strip()
        if line.startswith("Frames:"):
            frames = int(line.split(":")[1])
        elif line.startswith("Frame Time:"):
            frame_time = float(line.split(":")[1])
            data_start = i + 1
            break
    if frames is None or frame_time is None or data_start is None:
        raise ValueError(f"BVH 缺少 Frames/Frame Time 头: {path}")

    n_channels = sum(len(c) for c in channels)
    rows = []
    for line in lines[data_start : data_start + frames]:
        line = line.strip()
        if not line:
            continue
        rows.append(np.fromstring(line, sep=" "))
    motion = np.asarray(rows, dtype=np.float64)
    if motion.ndim != 2 or motion.shape[1] != n_channels:
        raise ValueError(
            f"BVH 通道数不匹配: 头部声明 {n_channels}，实际读到 {motion.shape}"
        )
    n_frames = motion.shape[0]

    # --- 拆分通道 ---
    local_quat = np.zeros((n_frames, n_joints, 4))
    root_pos = np.zeros((n_frames, 3))
    cursor = 0
    for j in range(n_joints):
        chans = channels[j]
        pos_chans = [c for c in chans if c.endswith("position")]
        rot_chans = [c for c in chans if c.endswith("rotation")]
        block = motion[:, cursor : cursor + len(chans)]
        cursor += len(chans)

        if pos_chans:
            order = {"Xposition": 0, "Yposition": 1, "Zposition": 2}
            pos = np.zeros((n_frames, 3))
            for k, c in enumerate(chans):
                if c in order:
                    pos[:, order[c]] = block[:, k]
            if j == 0:
                root_pos = pos
        rot_idx = [k for k, c in enumerate(chans) if c.endswith("rotation")]
        angles = block[:, rot_idx]
        rot = Rotation.from_euler(_euler_order(rot_chans), angles, degrees=True)
        local_quat[:, j] = rot.as_quat(scalar_first=True)

    offsets_arr = np.asarray(offsets)
    parents_arr = np.asarray(parents, dtype=np.int64)
    return names, parents_arr, offsets_arr, end_sites, local_quat, root_pos, 1.0 / frame_time


def skeleton_signature(path: str | Path, strip_prefix: bool = False) -> str:
    """骨架签名：层级结构与骨骼 offset 的摘要，动作内容不参与。

    Stage 0 生成的人体 MJCF 与 T-pose 点云、Stage I 学到的点云对应，只取决于这些
    量，与演员后面做了什么动作无关。

    注意这是**逐文件精确**的签名：FZMotion 这类会给每条 take 单独解算骨架的系统，
    同一个演员不同 take 的 offset 能差到 1 cm，签名就会不同。要让它们共用一套对应
    关系，走 ``--share_setup dir``（见 :func:`umr.stages.setup_signature`）。

    只读到 MOTION 的第一行就停，不解析整个动作块——批处理时目录里每个文件都会调它。
    """
    head: list[str] = []
    frame_time_at = None
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            head.append(line.rstrip("\n"))
            if frame_time_at is None:
                if line.lstrip().startswith("Frame Time:"):
                    frame_time_at = i
            elif i > frame_time_at:  # 拿到第一行动作数据即可
                break

    names, parents, offsets, end_sites, _channels, _idx = _parse_hierarchy(head)
    if strip_prefix:
        names = [n.split("_")[-1] for n in names]

    h = hashlib.sha1()
    h.update("|".join(names).encode("utf-8"))
    h.update(np.asarray(parents, dtype=np.int64).tobytes())
    h.update(np.round(np.asarray(offsets, dtype=np.float64), 3).tobytes())
    for j in sorted(end_sites):
        h.update(str(j).encode("utf-8"))
        h.update(np.round(end_sites[j], 3).tobytes())
    return h.hexdigest()[:16]


def load_bvh(
    path: str | Path,
    scale: float = 0.01,
    transform: np.ndarray = BVH_TO_MUJOCO,
    auto_face_x: bool = True,
    human: str | Skeleton = DEFAULT_HUMAN,
) -> BvhData:
    """加载 BVH 并转换到 MuJoCo 坐标系（X 前 / Y 左 / Z 上，单位米）。

    Args:
        path: BVH 文件路径。
        scale: 长度缩放，Xsens / FZMotion 都是厘米故为 0.01。
        transform: 3x3 旋转矩阵，把 BVH 基向量映射到 MuJoCo 基向量。
        auto_face_x: 若为 True，用第 0 帧（T-pose）自动估计朝向并施加一个绕 Z
            的偏航修正，使角色面向 +X。对不同导出配置更鲁棒。
        human: 源骨架标识或 :class:`~umr.bodies.skeletons.Skeleton`，决定关节
            命名约定（前缀处理与朝向估计用哪些关节）。
    """
    skeleton = human if isinstance(human, Skeleton) else get_skeleton(human)
    names, parents, offsets, end_sites, local_quat, root_pos, fps = read_bvh_raw(
        path, strip_prefix=skeleton.strip_prefix
    )

    R = np.asarray(transform, dtype=np.float64)
    offsets = (offsets * scale) @ R.T
    end_sites = {k: (v * scale) @ R.T for k, v in end_sites.items()}
    root_pos = (root_pos * scale) @ R.T

    # 局部旋转的相似变换：R_new = R @ R_old @ R^T
    T, J = local_quat.shape[:2]
    mats = Rotation.from_quat(local_quat.reshape(-1, 4), scalar_first=True).as_matrix()
    mats = R @ mats @ R.T
    local_quat = Rotation.from_matrix(mats).as_quat(scalar_first=True).reshape(T, J, 4)

    data = BvhData(
        names=names,
        parents=parents,
        offsets=offsets,
        end_sites=end_sites,
        local_quat=local_quat,
        root_pos=root_pos,
        fps=fps,
        source_path=str(path),
    )

    if auto_face_x:
        yaw = _estimate_facing_yaw(data, skeleton)
        if abs(yaw) > 1e-6:
            Rz = Rotation.from_euler("z", -yaw).as_matrix()
            data = _rotate_world(data, Rz)
    return data


def _rotate_world(data: BvhData, R: np.ndarray) -> BvhData:
    """对整段动作施加一个世界系旋转（只需改根节点与根位置）。"""
    root_pos = data.root_pos @ R.T
    local_quat = data.local_quat.copy()
    root_mat = Rotation.from_quat(local_quat[:, 0], scalar_first=True).as_matrix()
    root_mat = R @ root_mat
    local_quat[:, 0] = Rotation.from_matrix(root_mat).as_quat(scalar_first=True)
    return BvhData(
        names=data.names,
        parents=data.parents,
        offsets=data.offsets,
        end_sites=data.end_sites,
        local_quat=local_quat,
        root_pos=root_pos,
        fps=data.fps,
        source_path=data.source_path,
    )


def _estimate_facing_yaw(data: BvhData, skeleton: Skeleton) -> float:
    """从第 0 帧估计角色朝向的偏航角（弧度）。

    优先用 脚踝 -> 脚趾 的水平向量，退化时用 髋部连线的法向。用哪些关节由
    ``skeleton`` 给出，因为 xsens 与 fzmotion 的命名完全不同。
    """
    pos, _ = forward_kinematics(data, frames=np.array([0]))
    pos = pos[0]

    def find(*keys: str) -> int | None:
        for k in keys:
            for i, n in enumerate(data.names):
                if n.lower() == k.lower():
                    return i
        return None

    fwd = np.zeros(3)
    for ankle, toe in skeleton.foot_chains:
        ia, it = find(ankle), find(toe)
        if ia is not None and it is not None:
            fwd += pos[it] - pos[ia]
    if np.linalg.norm(fwd[:2]) < 1e-6:
        il, ir = find(skeleton.hip_pair[0]), find(skeleton.hip_pair[1])
        if il is None or ir is None:
            return 0.0
        left = pos[il] - pos[ir]
        # 前向 = 左向 叉乘 上向 的反号：forward = left x up
        fwd = np.cross(left, np.array([0.0, 0.0, 1.0]))
    if np.linalg.norm(fwd[:2]) < 1e-6:
        return 0.0
    return float(np.arctan2(fwd[1], fwd[0]))


def forward_kinematics(
    data: BvhData, frames: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """前向运动学。

    Args:
        data: BVH 数据。
        frames: 要计算的帧索引，None 表示全部。

    Returns:
        ``(positions, rotations)``，形状分别为 (T, J, 3) 与 (T, J, 3, 3)，全局量。
    """
    quat = data.local_quat if frames is None else data.local_quat[frames]
    root = data.root_pos if frames is None else data.root_pos[frames]
    T, J = quat.shape[:2]

    local_R = Rotation.from_quat(quat.reshape(-1, 4), scalar_first=True).as_matrix()
    local_R = local_R.reshape(T, J, 3, 3)

    gpos = np.zeros((T, J, 3))
    grot = np.zeros((T, J, 3, 3))
    gpos[:, 0] = root
    grot[:, 0] = local_R[:, 0]
    for j in range(1, J):
        p = data.parents[j]
        grot[:, j] = grot[:, p] @ local_R[:, j]
        gpos[:, j] = gpos[:, p] + np.einsum("tij,j->ti", grot[:, p], data.offsets[j])
    return gpos, grot


def slice_frames(data: BvhData, start: int, stop: int) -> BvhData:
    """按帧区间 ``[start, stop)`` 裁剪，骨架部分原样保留。"""
    return BvhData(
        names=data.names,
        parents=data.parents,
        offsets=data.offsets,
        end_sites=data.end_sites,
        local_quat=data.local_quat[start:stop],
        root_pos=data.root_pos[start:stop],
        fps=data.fps,
        source_path=data.source_path,
    )


def resample(data: BvhData, target_fps: float, method: str = "slerp") -> BvhData:
    """把动作重采样到 ``target_fps``，升采样与降采样都支持。

    根位置永远走线性插值；关节旋转按 ``method`` 选择：

    ``slerp``
        对每个关节的四元数序列做球面线性插值（scipy 的 :class:`Slerp`）。角速度
        沿测地线恒定，是旋转唯一正确的插值方式。
    ``linear``
        逐分量线性插值后重新归一化。两个单位四元数逐分量 lerp 的结果模长小于 1
        （正比于 cos(夹角/2)），不归一化会引入缩放畸变；夹角大时即使归一化也会
        出现角速度不均匀。只在需要与按此法重采样的现成结果对齐时使用。

    源帧率与目标帧率相同时原样返回。
    """
    if target_fps <= 0:
        raise ValueError(f"目标帧率必须为正: {target_fps}")
    if abs(target_fps - data.fps) < 1e-9 or data.num_frames < 2:
        return data

    # 时间轴按"第 0 帧对齐 t=0"构建，保证 T-pose 标定帧原样保留。
    duration = (data.num_frames - 1) / data.fps
    src_t = np.arange(data.num_frames) / data.fps
    n_out = max(int(round(duration * target_fps)) + 1, 2)
    dst_t = np.clip(np.arange(n_out) / target_fps, src_t[0], src_t[-1])

    root_pos = np.stack(
        [np.interp(dst_t, src_t, data.root_pos[:, i]) for i in range(3)], axis=-1
    )

    J = data.local_quat.shape[1]
    local_quat = np.zeros((n_out, J, 4))
    if method == "slerp":
        for j in range(J):
            rots = Rotation.from_quat(data.local_quat[:, j], scalar_first=True)
            local_quat[:, j] = Slerp(src_t, rots)(dst_t).as_quat(scalar_first=True)
    elif method == "linear":
        for c in range(4):
            local_quat[:, :, c] = np.stack(
                [np.interp(dst_t, src_t, data.local_quat[:, j, c]) for j in range(J)], axis=-1
            )
        local_quat /= np.maximum(np.linalg.norm(local_quat, axis=-1, keepdims=True), 1e-8)
    else:
        raise ValueError(f"未知插值方法: {method}（可用 slerp / linear）")

    return BvhData(
        names=data.names,
        parents=data.parents,
        offsets=data.offsets,
        end_sites=data.end_sites,
        local_quat=local_quat,
        root_pos=root_pos,
        fps=float(target_fps),
        source_path=data.source_path,
    )


def prepare_clip(
    data: BvhData,
    *,
    tgt_fps: float | None = None,
    start: float = 0.0,
    duration: float | None = None,
    interpolation: str = "slerp",
    log: Callable[[str], None] | None = None,
) -> BvhData:
    """把整段源动作变成 Stage II 真正求解的那一段。

    顺序是「先扔标定帧、再按秒截取、最后插值」。BVH 开头那帧全零的标定帧必须在插值
    之前扔掉：它的根位置在原点而第一帧真实姿态里演员可能站在数米之外，两者一插值
    就会插出一段根本不存在的瞬移。

    这里的三个参数会一并写进 ``motion.npz``，可视化与报告脚本用同样的入参重跑本函数
    即可拿到与当初逐帧一一对应的人体动作，``frame_indices`` 因此可以直接用。

    Args:
        data: 原始帧率下的完整动作。
        tgt_fps: 输出帧率；None 或与源帧率相同则不重采样。
        start: 起始时刻（秒），相对真实动作的第一帧。
        duration: 截取时长（秒），None 表示到结尾。
        interpolation: ``slerp`` 或 ``linear``，见 :func:`resample`。
        log: 可选的日志回调。
    """
    say = log or (lambda _: None)
    src_fps = data.fps
    rest = first_motion_frame(data)
    if rest:
        say(f"跳过 BVH 开头 {rest} 个合成静止帧（真实动作从此开始）")

    f0 = max(rest, rest + int(round(start * src_fps)))
    f1 = data.num_frames if duration is None else min(
        data.num_frames, f0 + int(round(duration * src_fps))
    )
    if f1 - f0 < 2:
        raise ValueError(f"裁剪后只剩 {f1 - f0} 帧，检查 start / duration")
    clip = slice_frames(data, f0, f1)

    if tgt_fps is None or abs(tgt_fps - src_fps) < 1e-9:
        say(f"源 {src_fps:.1f} FPS x {clip.num_frames} 帧（未改帧率）")
        return clip

    out = resample(clip, tgt_fps, method=interpolation)
    say(f"源 {src_fps:.1f} FPS x {clip.num_frames} 帧 -> {out.fps:.1f} FPS x "
        f"{out.num_frames} 帧（{interpolation} 插值，{clip.num_frames / src_fps:.1f}s）")
    return out


def first_motion_frame(data: BvhData, tol: float = 1e-9) -> int:
    """跳过 BVH 开头的合成静止帧。

    Xsens 导出的文件常在第 0 帧写入一个所有通道为零的标定帧：局部旋转全为单位
    四元数、根位置在原点，而真实动作从第 1 帧才开始（且演员可能在离原点数米处）。
    如果把这一帧也拿去重定向，机器人会在第 2 帧被迫"瞬移"，产生巨大的跟踪误差。

    这一帧同时正好是我们需要的 canonical T-pose，因此保留给 Stage I 使用，只在
    Stage II 的时序求解中跳过。
    """
    ident = np.array([1.0, 0.0, 0.0, 0.0])
    deviation = np.abs(data.local_quat - ident).max(axis=(1, 2))
    moving = np.where(deviation > tol)[0]
    return int(moving[0]) if len(moving) else 0


def static_head_frames(
    data: BvhData,
    max_seconds: float = 1.0,
    pos_tol: float = 0.02,
    rot_tol_deg: float = 3.0,
) -> int:
    """片头演员站定不动的帧数。

    与 :func:`first_motion_frame` 不同：那个找的是导出器写死的合成标定帧（通道
    严格为零），这里找的是**真人站着没动**的那一段。动捕采集通常留有这么一段，
    而它是唯一可以信任「脚平放在地上」的时刻，因此适合拿来标定那些解算质量差的
    通道——踝关节的 roll 尤其容易在腾空/遮挡时跳到错误分支再也回不来。

    判据是相对第 0 帧的偏离：根平移小于 ``pos_tol``，且所有关节的四元数夹角都小于
    ``rot_tol_deg``。返回满足条件的前导帧数（已按 ``max_seconds`` 截断）。
    """
    dots = np.abs(np.einsum("tjk,jk->tj", data.local_quat, data.local_quat[0]))
    rot_dev = 2.0 * np.arccos(np.clip(dots, -1.0, 1.0)).max(axis=1)
    pos_dev = np.linalg.norm(data.root_pos - data.root_pos[0], axis=1)
    ok = (pos_dev < pos_tol) & (rot_dev < np.radians(rot_tol_deg))
    n = data.num_frames if ok.all() else int(np.argmin(ok))
    return int(min(n, round(max_seconds * data.fps)))


def bone_lengths(data: BvhData) -> dict[str, float]:
    """每个关节到其父节点的骨长（米）。"""
    return {
        name: float(np.linalg.norm(data.offsets[j]))
        for j, name in enumerate(data.names)
    }


def actor_height(data: BvhData) -> float:
    """从 T-pose 估计演员身高（脚底到头顶，米）。"""
    pos, rot = forward_kinematics(data, frames=np.array([0]))
    pos, rot = pos[0], rot[0]
    zs = [pos[:, 2].min()]
    for j, off in data.end_sites.items():
        zs.append((pos[j] + rot[j] @ off)[2])
    top = max(
        [pos[:, 2].max()] + [(pos[j] + rot[j] @ off)[2] for j, off in data.end_sites.items()]
    )
    return float(top - min(zs))
