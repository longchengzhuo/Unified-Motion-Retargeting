# assets/

Robot models. Each robot follows the same layout:

```
assets/robots/<FAMILY>_description/
├── xml/<VARIANT>.xml          # MuJoCo MJCF — this is what the code loads
├── urdf/                      # URDF / SRDF (reference only, unused)
└── meshes/<VARIANT>/*.STL     # meshes, referenced via the MJCF meshdir
```

| Robot | MJCF | Config | Source |
|---|---|---|---|
| Unitree G1 (29 DoF, rev 1.0) | `g1_description/xml/g1_29dof_rev_1_0.xml` | `configs/g1_29dof_rev_1_0.yaml` | [unitree_ros](https://github.com/unitreerobotics/unitree_ros/tree/master/robots/g1_description), BSD-3-Clause |

> The first run of `scripts/01_build_bodies.py` **edits the MJCF in place**: it injects a
> `T_pose` keyframe and enlarges the offscreen framebuffer to 1920x1080. This is idempotent.

## Adding a robot

The method needs no hand-authored joint mapping, so the requirements are minimal:

1. The MJCF loads with `mujoco.MjModel.from_xml_path` and has a floating base (free joint) on the root body.
2. Hinge joints carry a `range`, which `mink.ConfigurationLimit` turns into the Eq. (13) joint limits.
3. Foot links carry primitive geoms (box or sphere) at the sole. `sole_sample_points()` feeds them to
   the Eq. (14) ground clearance constraint. Mesh-only feet still work — the constraint falls back to
   the learned correspondence points near the ground — but the guarantee is weaker.

Then add a config under `configs/`. Only the `robot` block differs between robots:

```yaml
robot:
  name: my_robot
  xml: assets/robots/my_robot_description/xml/my_robot.xml
  root_body: pelvis                 # optional; defaults to the first body under world
  tpose_joints:                     # joints to offset for the canonical T-pose
    left_shoulder_roll_joint: 1.5708
    right_shoulder_roll_joint: -1.5708
  marker_body_prefixes: []          # bodies whose geoms are markers, not surface
  marker_body_suffixes: []
  foot_name_keys: ["ankle", "foot"] # substrings that identify foot links
  foot_bodies: []                   # optional; inferred from foot_name_keys if empty
```

Everything downstream — correspondence learning, residuals, constraints, solver — is unchanged.
Run with `./run_all.sh --config configs/my_robot.yaml`.
