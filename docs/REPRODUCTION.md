# Runtime setup and validation

This document records the existing validated EVA12 setup. It is not a one-command fresh-machine installer. The project deliberately separates dependency installation, asset acquisition, and runtime launch.

## Pinned source trees

Clone each upstream into the indicated ignored directory and check out the exact revision. Do not overwrite existing clones or modify their files.

| Directory | Repository | Revision |
| --- | --- | --- |
| `sources/robocasa-openpi` | https://github.com/robocasa-benchmark/openpi | `ca4c6d710db75e276bc7c866a57bd7e4aee5b6e8` |
| `sources/robocasa` | https://github.com/robocasa/robocasa | `456174f62b89b8fca99eaaf33949c29fec9cfc2a` |
| `sources/robosuite` | https://github.com/ARISE-Initiative/robosuite | `5ce6643f3092639d08f7b0f90ed1c6a84f50552c` |

The reference checkout `sources/openpi` at `215abfb217dbac7d5f1273282331b9b1866c0479` is not used by the runtime launch scripts.

## Environments

Both environments use Python 3.11.16. Project-local uv 0.12.19 was used. Set `UV_CACHE_DIR`, `PIP_CACHE_DIR`, `XDG_CACHE_HOME`, and `TMPDIR` beneath `cache/` before any installation.

- Simulation: `cache/venvs/robocasa`, with `requirements-sim.lock` plus editable installs of this project, robosuite, RoboCasa, and the fixed fork's `packages/openpi-client`, using `--no-deps` for the editable packages.
- Inference: `cache/venvs/openpi-inference`, installed from the fixed fork's `uv.lock` using `uv sync --frozen --no-dev` with `UV_PROJECT_ENVIRONMENT` set explicitly. The same lock supplied `pytest==8.3.5`, `iniconfig==2.1.0`, and `pluggy==1.6.0` needed by runtime imports and checks. Final observed distribution count: 207.

Do not combine the environments or upgrade MuJoCo/NumPy to resolve training-registry imports. The lightweight service avoids those imports.

Known metadata differences:

- Simulation: headless OpenCV replaces the GUI package named by robosuite and RoboCasa. RoboCasa's declared LeRobot and Tianshou dependencies are omitted from the simulation-only runtime. `python -m robocasa_eval.verify_environment` accepts only the four documented declaration differences; it is not a clean generic `pip check` result.
- Inference: the pinned LeRobot Git revision `0cf864870cf29f4738d3ade893e6fd13fbd7cdb5` reports version 0.1.0, while OpenPI declares 0.3.3. Do not silently replace that revision.

## Model files

Hugging Face repository: `robocasa/robocasa365_checkpoints`.
Revision used for download: `916d8ec4aab4e6f930ba6b5b9bb8537ad16d4bb8`.
Subdirectory: `pi05_pretrain_human300/multitask_learning/75000`.

Expected local layout:

```text
checkpoints/robocasa365/pi05_pretrain_human300/multitask_learning/75000/
  params/                 # 18 Orbax files, 12,440,860,298 bytes total
  assets/norm_stats.json  # 3,208 bytes
cache/openpi/big_vision/paligemma_tokenizer.model  # 4,264,023 bytes
```

The tokenizer source is `gs://big_vision/paligemma_tokenizer.model`. The training-state directory is unnecessary for this inference path and was not downloaded. Matching byte counts alone is not cryptographic verification. Remote file integrity metadata and local digests should be retained for a stronger reproducibility record.

## Simulation assets and rendering

Kitchen assets reside in `datasets/robocasa-kitchen-assets/`. The validated smoke setup includes the upstream bundled base assets and `tex`, `tex_generative`, `fixtures_lw`, and `objs_lw` archives. The full benchmark may require additional object libraries. `scripts/asset_inventory.py` inventories official download sources; it is not an installer.

Private CPU rendering prefix: `cache/rendering`, conda-forge `mesalib==25.0.5`, build `h57bcd07_2`. The validated installation had 25 packages. The Mesalib archive SHA-256 was `b2c88c95088db3dd3048242a48e957cf53ac852047ebaafc3a822bd083ad9858`. A full rendering-environment package lock is not included in this release.

The launch script sets `MUJOCO_GL=osmesa`, software rendering, and process-local library paths. It does not modify system libraries or shell startup files.

## Validation order

1. Verify pinned source revisions and independent environments.
2. Run offline tests in both environments; the transform tests require the inference environment.
3. Validate simulation imports and OSMesa context creation.
4. Run a small single-GPU JAX operation and checkpoint recovery.
5. Start the loopback policy, then run `bash scripts/run_smoke.sh SEED`.
6. Retain videos, traces, statistics, environment versions, and both sampling and scene seed metadata.

The existing implementation records the scene seed but does not yet provide explicit policy RNG control. First-run JAX compilation can take longer than subsequent requests. Stop only your own service using Ctrl+C after the experiment.
