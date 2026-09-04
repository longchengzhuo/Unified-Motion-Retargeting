#!/usr/bin/env bash
# UMR end-to-end pipeline.
#
#   ./run_all.sh                                        # default robot (Unitree G1)
#   ./run_all.sh --config configs/my_robot.yaml         # another robot
#   ./run_all.sh --duration 10                          # first 10 s only (quick check)
#
# Results go to outputs/<config-name>/. Unrecognised arguments are forwarded to
# 03_retarget.py.

set -euo pipefail
cd "$(dirname "$0")"

export PYTHONNOUSERSITE=1        # keep ~/.local packages from shadowing the conda env
export PYTHONWARNINGS=ignore
export MUJOCO_GL="${MUJOCO_GL:-glfw}"

CONFIG="configs/g1_29dof_rev_1_0.yaml"
EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done

export UMR_OUTPUT_DIR="${UMR_OUTPUT_DIR:-outputs/$(basename "$CONFIG" .yaml)}"

PY="${UMR_PYTHON:-python}"
if ! command -v "$PY" >/dev/null 2>&1; then
  echo "python not found: $PY (create the env with 'conda env create -f environment.yml')" >&2
  exit 1
fi

step() { printf '\n\033[1;36m=== %s ===\033[0m\n' "$1"; }

step "Stage 0  build human/robot MuJoCo bodies, sample T-pose surfaces"
"$PY" scripts/01_build_bodies.py --config "$CONFIG"

step "Stage I  learn point cloud correspondence (Eq. 1-5)"
"$PY" scripts/02_learn_correspondence.py --config "$CONFIG"

step "Stage II correspondence-guided retargeting (Eq. 6-14)"
"$PY" scripts/03_retarget.py --config "$CONFIG" ${EXTRA[@]+"${EXTRA[@]}"}

step "Visualise  correspondence figure + side-by-side video"
"$PY" scripts/04_visualize.py --mode corr
"$PY" scripts/04_visualize.py --mode video --points --stride 2 --max_frames 900

step "Validate   MuJoCo dynamics + metric report"
"$PY" scripts/05_validate.py --replay

printf '\n\033[1;32mDone.\033[0m Results in %s/\n' "$UMR_OUTPUT_DIR"
printf '  report.md            metric report\n'
printf '  correspondence.png   Stage I correspondence figure\n'
printf '  retarget.mp4         side-by-side comparison video\n'
printf '  motion.pkl           retargeted motion (agmr/GMR-compatible fields)\n'
printf '  motion.npz           retargeted motion (internal format)\n'
printf '\nInteractive player:\n'
printf '  UMR_OUTPUT_DIR=%s python scripts/04_visualize.py --mode viewer --points\n' "$UMR_OUTPUT_DIR"
