"""对应关系的解剖学一致性评估。

论文的核心主张是：不需要人工指定骨骼映射，学到的稠密表面对应本身就会把人体
各部位对到机器人的相应部位。这里把它量化——检查每个人体分段的对应点最终落在
哪一组机器人 link 上。
"""

from __future__ import annotations

import collections

import numpy as np

#: 人体分段 -> 期望的机器人肢体组
SEGMENT_TO_LIMB: dict[str, str] = {
    "l_upperarm": "l_arm", "l_forearm": "l_arm", "l_hand": "l_arm",
    "r_upperarm": "r_arm", "r_forearm": "r_arm", "r_hand": "r_arm",
    "l_thigh": "l_leg", "l_shin": "l_leg", "l_foot": "l_leg", "l_toe": "l_leg",
    "r_thigh": "r_leg", "r_shin": "r_leg", "r_foot": "r_leg", "r_toe": "r_leg",
    "pelvis": "torso", "torso": "torso", "chest": "torso",
    "head": "head",
    # l_clavicle / r_clavicle / neck 处于分界处，不计入评分
}


def robot_link_group(name: str) -> str:
    """把机器人 link 名归到一个肢体组。"""
    arm = ("shoulder" in name) or ("elbow" in name) or ("wrist" in name) or ("hand" in name)
    if name.startswith("left_"):
        return "l_arm" if arm else "l_leg"
    if name.startswith("right_"):
        return "r_arm" if arm else "r_leg"
    if name.startswith("head"):
        return "head"
    return "torso"


def anatomical_consistency(
    segment: np.ndarray,
    segment_names: list[str],
    bound_body_ids: np.ndarray,
    body_names: list[str],
) -> tuple[float, dict[str, float], dict[str, list[tuple[str, int]]]]:
    """评估学到的对应是否把人体分段对到了正确的机器人肢体。

    Returns:
        ``(总体正确率, 各分段正确率, 各分段落点的 link 分布)``。
    """
    ok = tot = 0
    per_seg_ok: dict[str, list[int]] = collections.defaultdict(list)
    per_seg_links: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)

    for i, s in enumerate(segment):
        if s < 0:
            continue
        seg = segment_names[int(s)]
        link = body_names[int(bound_body_ids[i])]
        per_seg_links[seg][link] += 1
        expect = SEGMENT_TO_LIMB.get(seg)
        if expect is None:
            continue
        hit = int(robot_link_group(link) == expect)
        per_seg_ok[seg].append(hit)
        ok += hit
        tot += 1

    overall = float(ok / tot) if tot else 0.0
    per_seg = {k: float(np.mean(v)) for k, v in per_seg_ok.items()}
    top_links = {k: v.most_common(3) for k, v in per_seg_links.items()}
    return overall, per_seg, top_links
