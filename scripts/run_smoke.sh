#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
cd -- "$ROOT"
SEED=${1:-0}
[[ $# -le 1 && "$SEED" =~ ^[0-9]+$ ]] || { echo 'Usage: bash scripts/run_smoke.sh SCENE_SEED' >&2; exit 1; }
PY="$ROOT/cache/venvs/robocasa/bin/python"
[[ -x "$PY" && -f "$ROOT/cache/rendering/lib/libOSMesa.so" ]] || { echo 'Missing simulation environment or private OSMesa.' >&2; exit 1; }
ss -H -ltn 'sport = :8000' | grep -q '127.0.0.1:8000' || { echo 'Start your policy server on 127.0.0.1:8000 first.' >&2; exit 1; }
mkdir -p cache/xdg cache/tmp
export CUDA_VISIBLE_DEVICES=""
export MUJOCO_GL=osmesa
export LIBGL_ALWAYS_SOFTWARE=1
export LD_LIBRARY_PATH="$ROOT/cache/rendering/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export XDG_CACHE_HOME="$ROOT/cache/xdg"
export TMPDIR="$ROOT/cache/tmp"
export PYTHONPATH="$ROOT/src"
exec "$PY" -m robocasa_eval.main --smoke-lightwheel-task OpenDrawer --num-trials 1 \
  --seed "$SEED" --trace-actions --log-dir "$ROOT/outputs/baseline-smoke/seed-$SEED" \
  --host 127.0.0.1 --port 8000
