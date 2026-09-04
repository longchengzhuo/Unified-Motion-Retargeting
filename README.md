<h1 align="center">UMR</h1> 

<p align="center">
  <b>Unified Motion Retargeting for Humanoids with Learned Point Cloud Correspondence</b><br>
  Surface-point-cloud retargeting from mocap to humanoid robots — no hand-authored joint mapping.
</p>

<p align="center">
  <a href="README.md"><img src="https://img.shields.io/badge/English-informational?style=for-the-badge" alt="English"></a>
  <a href="README.zh-CN.md"><img src="https://img.shields.io/badge/简体中文-lightgrey?style=for-the-badge" alt="简体中文"></a>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT license"></a>
  <img src="https://img.shields.io/badge/python-3.10-blue.svg" alt="Python 3.10">
  <img src="https://img.shields.io/badge/MuJoCo-3.9-orange.svg" alt="MuJoCo 3.9">
  <a href="https://arxiv.org/abs/2609.02134"><img src="https://img.shields.io/badge/arXiv-2609.02134-b31b1b.svg" alt="arXiv"></a>
</p>

> **Unofficial.** An independent reproduction inspired by the original authors' paper
> ([arXiv:2609.02134](https://arxiv.org/abs/2609.02134)). This is not the official implementation
> and is not affiliated with or endorsed by the authors.

UMR retargets a mocap clip onto a humanoid robot by matching **outer surface point clouds**
instead of skeletons, so switching robots needs no new human↔robot joint correspondence.
This repository implements both stages on top of **MuJoCo + [mink](https://github.com/kevinzakka/mink)**:
the point / normal / contact residuals are `mink.Task` subclasses, ground clearance and the trust
region are `mink.Limit` subclasses, and the QP goes to **Clarabel** via `qpsolvers`.

**Unitree G1** (29 DoF) ships with the repo and runs out of the box.

![retargeting](docs/images/walk.png)

*Stage II retargeting, mid-stride. Red dots are the selected human surface points, green dots are
their learned counterparts on the robot — same index, same body part, none of it assigned by hand.*

---

## Repository layout

```
umr/
├── environment.yml                     # the only dependency list
├── run_all.sh                          # end-to-end pipeline
├── configs/g1_29dof_rev_1_0.yaml       # Unitree G1 (default)
├── assets/robots/                      # MJCF + STL meshes, one dir per robot
├── data/                               # source BVH
├── scripts/01..05_*.py                 # the five pipeline steps
├── umr/
│   ├── bodies/       # BVH parsing, human MJCF generation, robot wrapper, surface sampler
│   ├── correspondence/  # Stage I: network, losses, geodesic graph, training, evaluation
│   ├── tasks/        # mink.Task subclasses      (Eq. 7, 8)
│   ├── limits/       # mink.Limit subclasses     (Eq. 13, 14)
│   ├── retarget/     # link binding, per-frame pipeline, pkl export
│   └── sim/          # dynamics validation, offline rendering, interactive player
├── docs/TECHNICAL.md                   # derivations and implementation notes
└── outputs/<config-name>/              # generated motions, videos, reports
```

## Method

```mermaid
flowchart LR
  bvh["Xsens BVH"] --> hmjcf["procedural human MJCF"]
  hmjcf --> hcfg["mink.Configuration (human)"]
  rmjcf["robot MJCF + T_pose key"] --> rcfg["mink.Configuration (robot)"]
  hcfg --> samp["shared surface sampler"]
  rcfg --> samp
  samp --> Xh["X^h ordered human cloud + segment labels"]
  samp --> Xr["X^r unordered robot cloud"]
  Xh --> net["PointNet encoder + MLP decoder"]
  Xr --> net
  net --> corr["X̂^r = X^h + D(E(X^r))"]
  corr --> bind["snap to links: local position + normal"]
  hcfg --> posed["per-frame human surface points"]
  bind --> tasks["mink.Task: point / normal / contact map"]
  posed --> tasks
  tasks --> ik["mink.solve_ik (Clarabel) + integrate_inplace"]
  lim["mink.Limit: joint limits / ground clearance / trust region"] --> ik
  ik --> out["robot qpos sequence"]
  out --> sim["MuJoCo dynamics check + rendering"]
```

**Stage I — correspondence learning (paper III-B).** Trained once on the aligned canonical T-pose,
producing a reusable ordered set of human↔robot surface point pairs:

$$\hat{X}^r = X^h + D_\theta(E_\theta(X^r))$$

$E_\theta$ is a PointNet-style encoder that compresses the *unordered* robot cloud into a latent
vector; $D_\theta$ is a per-point MLP predicting a deformation for each *ordered* human template
point. The loss $L_{corr} = \lambda_c L_c + \lambda_r L_r + \lambda_e L_e$ combines symmetric
Chamfer, KNN repulsion, and edge smoothing on the human geodesic graph (Eq. 2–5). Because the
indexing is inherited from the human cloud, robot points inherit human segment labels
automatically — this is where "no manual body mapping" comes from.

**Stage II — correspondence-guided retargeting (paper III-C).** Eq. (6) is solved per frame:
position and normal residuals (Eq. 7) plus a contact-map residual (Eq. 8–11), under joint limits,
ground clearance (Eq. 14) and a trust region, via damped Gauss-Newton (Eq. 12–13). The previous
frame warm-starts the next one.

## Citation

```bibtex
@article{cao2026umr,
  title   = {Unified Motion Retargeting for Humanoids with Learned Point Cloud Correspondence},
  author  = {Cao, Hanyang and Fang, Yuetong and Kwon, Taesoo and Yu, Runyi and Ma, Ji and
             Tan, Jing and Zhou, Yangchen and Du, Baoze and Gu, Yi and Gao, Yukang and
             Dai, Ruoli and Han, Lei and Xu, Renjing},
  journal = {arXiv preprint arXiv:2609.02134},
  year    = {2026}
}
```

## Installation

```bash
conda env create -f environment.yml
conda activate umr
```

Python 3.10, mujoco 3.9.0, mink 1.1.1, clarabel 0.11.1, torch 2.12.0. Verify:

```bash
python -c "import qpsolvers; assert 'clarabel' in qpsolvers.available_solvers; print('ok')"
```

A GPU is optional and only used by Stage I training (~22 s on an RTX 3090, ~11 min on 28 CPU cores).
`correspondence.device: auto` falls back to CPU on its own.

## Quick start

Robot models and the source clip ship with the repo, so this runs as-is:

```bash
./run_all.sh                    # Unitree G1, full clip
./run_all.sh --duration 10      # first 10 s only
```

Results land in `outputs/<config-name>/`. Step by step:

```bash
python scripts/01_build_bodies.py           # human + robot MJCF, T-pose surface sampling
python scripts/02_learn_correspondence.py   # Stage I: correspondence + link binding
python scripts/03_retarget.py               # Stage II: per-frame retargeting
python scripts/04_visualize.py --mode corr  # correspondence figure
python scripts/04_visualize.py --mode video --points    # side-by-side video
python scripts/04_visualize.py --mode viewer --points   # interactive player
python scripts/05_validate.py --replay      # dynamics check + metric report
```

Every script takes `--config`; set `UMR_OUTPUT_DIR` to redirect artifacts. Useful flags:

| Flag | Effect |
|---|---|
| `03_retarget.py --duration 10` | retarget the first 10 s only |
| `03_retarget.py --trust_region l2` | exact L2 trust region via Clarabel SOCP (default `box`) |
| `03_retarget.py --n_selected 1024` | larger selected set $\|I\|$ (slower; see `docs/TECHNICAL.md` §8.3) |
| `03_retarget.py --solver proxqp` | switch QP backend |
| `02_learn_correspondence.py --device cpu` | force CPU for Stage I |

All hyperparameters live in the config file.

### Interactive player

`--mode viewer` opens a live MuJoCo window that starts paused on frame 0. **Hold →** to play,
**hold ←** to rewind, release to pause. Space toggles autoplay, `.` / `,` single-step,
`[` / `]` change speed, `T` toggles camera follow, `P` toggles correspondence points, `Esc` quits.
`--human_offset 0` overlays the human and robot instead of placing them side by side;
`--robot_only` hides the human.

### Adding a robot or a motion

See [`assets/README.md`](assets/README.md) and [`data/README.md`](data/README.md).
Only the `robot` block of the config differs between robots — nothing in the method is retuned.

## Results

Unitree G1 (29 DoF, 1.32 m), full 71.2 s clip (240 Hz → 30 Hz, 2137 frames), RTX 3090 + i7.

| Stage I | |
|---|---|
| Chamfer recon→target | 19.0 mm |
| 2 cm coverage | 67.7 % |
| **Anatomical consistency** | **90.9 %** |

| Stage II | |
|---|---|
| Point error, median | **31.7 mm** |
| Normal error, mean | 38.3° |
| Peak joint torque, median | 5.8 N·m |
| Foot penetration, max | 0.57 mm |
| Foot-height tracking, corr L / R | 0.80 / 0.59 |
| Joint limit violations | 0 % |
| QP failures | 0 |

| Cost | |
|---|---|
| Setup (sampling + training + binding) | 44.9 s, once per robot |
| Retargeting throughput | **63.6 FPS** |

Anatomical consistency measures whether learned correspondences land on the anatomically right
limb, with **no manual skeleton mapping** anywhere in the pipeline. Per-segment breakdowns are
printed by `02_learn_correspondence.py`; the 9 % gap is dominated by the head, which G1's 29 DoF
MJCF folds into `torso_link` and therefore has no link name to match against. The paper reports
65.29 FPS overall throughput.

Full report: `outputs/<config-name>/report.md`. Comparison video: `outputs/<config-name>/retarget.mp4`.

## Output format

`03_retarget.py` writes `motion.npz` (used by steps 04/05) and `motion.pkl`, whose fields match
`agmr` / GMR so downstream tooling works unchanged:

```python
{
  "root_trans":  (T, 3),   # base translation
  "root_rot":    (T, 4),   # base rotation, xyzw (downstream convention)
  "dof":         (T, nj),  # joint angles
  "dof_full":    (T, nj),
  "qpos":        (T, nq),  # raw MuJoCo layout, quaternion is wxyz
  "fps": 30.0, "dof_names": [...], "body_names": [...],
  "quality_metrics": {...}, "point_error": (T,), "normal_error": (T,),
  "contact_count": (T,), "frame_indices": (T,),
  "source_file": ..., "robot_xml": ..., "scale": ..., "ground_offset": ...,
}
```

Note `root_rot` is **xyzw** while `qpos` keeps MuJoCo's **wxyz**.

## Differences from the paper

- **Source mesh** is a rigid-body human generated procedurally from the BVH skeleton, not SMPL-X
  with shape fitting. Its surface is piecewise-smooth convex primitives, so normals carry a
  systematic bias against the robot's faceted CAD meshes; the normal term is weighted low and acts
  as a soft orientation cue. Paper III-A lists rigged humanoid characters as a valid source.
- **Trust region** defaults to the inscribed box of the L2 ball
  ($\|\Delta q\|_\infty \le \eta/\sqrt{n_v}$), which satisfies the L2 constraint with any QP
  backend. `--trust_region l2` uses the Clarabel SOCP branch and matches Eq. (13) exactly.
- **Ground contact only** — no object, scene, or self-collision, since the bundled clip is walking.
  `ContactMapTask` is written against a general environment point cloud, so adding objects means
  replacing `environment`.
- **No downstream RL.** There is no BeyondMimic / SONIC / OmniRetarget comparison and no LAFAN1
  benchmark. The PD replay in `05_validate.py` is an **open-loop underactuated** simulation with no
  balance controller — a kinematic reference is expected to fall. It is a reproducible feasibility
  probe, not a stability claim.
- **ZMP** uses the standard CoM approximation that ignores angular momentum rate. Walking is
  controlled falling, so single-support ZMP excursions past the foot edge are normal.

## License

Code is released under the [MIT License](LICENSE).

Robot assets keep their own terms: the Unitree G1 description is BSD-3-Clause
(see [`assets/robots/g1_description/LICENSE`](assets/robots/g1_description/LICENSE)) and is
vendored from [unitree_ros](https://github.com/unitreerobotics/unitree_ros).
