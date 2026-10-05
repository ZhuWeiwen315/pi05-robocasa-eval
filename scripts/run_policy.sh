#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
cd -- "$ROOT"
GPU=${1:-0}
[[ $# -le 1 && "$GPU" =~ ^[0-9]+$ ]] || { echo 'Usage: bash scripts/run_policy.sh GPU_INDEX' >&2; exit 1; }
PY="$ROOT/cache/venvs/openpi-inference/bin/python"
CHECKPOINT="$ROOT/checkpoints/robocasa365/pi05_pretrain_human300/multitask_learning/75000"
[[ -x "$PY" && -d "$CHECKPOINT/params" && -f "$CHECKPOINT/assets/norm_stats.json" && -f "$ROOT/cache/openpi/big_vision/paligemma_tokenizer.model" ]] || { echo 'Missing inference environment, checkpoint, or tokenizer.' >&2; exit 1; }
if [[ -n $(ss -H -ltn 'sport = :8000') ]]; then
  echo 'Port 8000 is already in use; no process was stopped.' >&2; exit 1
fi
STATUS=$(nvidia-smi --id="$GPU" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
read -r MEM UTIL <<< "${STATUS//,/ }"
[[ "$MEM" == 0 && "$UTIL" == 0 ]] || { echo "GPU $GPU is not idle: $STATUS" >&2; exit 1; }
echo "Using GPU $GPU. Follow laboratory allocation rules; idle status is not a reservation."
mkdir -p cache/jax cache/xdg cache/tmp cache/huggingface cache/torch cache/openpi
export CUDA_VISIBLE_DEVICES="$GPU"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cuda
export PYTHONPATH="$ROOT/src:$ROOT/sources/robocasa-openpi/src:$ROOT/sources/robocasa-openpi/packages/openpi-client/src"
export OPENPI_DATA_HOME="$ROOT/cache/openpi"
export JAX_COMPILATION_CACHE_DIR="$ROOT/cache/jax"
export XDG_CACHE_HOME="$ROOT/cache/xdg"
export TMPDIR="$ROOT/cache/tmp"
export HF_HOME="$ROOT/cache/huggingface"
export TORCH_HOME="$ROOT/cache/torch"
exec "$PY" -m robocasa_eval.serve_policy --checkpoint "$CHECKPOINT" --host 127.0.0.1 --port 8000
