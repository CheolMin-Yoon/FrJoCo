#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="/home/frlab/mujoco_demo"

CONDA_ENV="/home/frlab/anaconda3/envs/mujoco_env"
MUJOCO_LIB="${CONDA_ENV}/lib/python3.10/site-packages/mujoco"

export LD_LIBRARY_PATH="${MUJOCO_LIB}:${CONDA_ENV}/lib:${LD_LIBRARY_PATH}"

# 워크스페이스 루트에서 실행 (상대경로 기준점)
cd "${WORKSPACE}"
"${SCRIPT_DIR}/build/main"
