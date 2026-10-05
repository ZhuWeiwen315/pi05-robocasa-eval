# π₀.₅ × RoboCasa: isolated inference and rollout diagnostics

A project-owned evaluation harness for the official RoboCasa π₀.₅ checkpoint, with separate inference and simulation environments, loopback WebSocket communication, CPU software rendering, and per-step rollout traces.

**Status:** real checkpoint recovery, single-GPU inference, three-camera simulation, and end-to-end `OpenDrawer` rollouts have run on EVA12. This is an ongoing evaluation project; no model training or fine-tuning has been performed.

## What this project adds

- A lightweight JAX policy service that bypasses the upstream training configuration registry and its incompatible simulation imports.
- A RoboCasa 1.0.1 evaluation entry point with task selection, horizon handling, finite 12D action checks, videos, and result metadata.
- Project-local asset redirection, without modifying upstream source trees.
- Optional JSONL traces of policy state, actions, target fixture, normalized drawer opening, and task success.
- Offline tests for protocol behavior, transforms, import isolation, environment declarations, asset handling, and trace capture.

The model, checkpoint, RoboCasa environment, and core policy transforms come from the upstream projects. This repository does not implement or train π₀.₅ from scratch.

## Initial smoke results

One `OpenDrawer` rollout per scene seed, using `pretrain`, Lightwheel-only objects and `generative_textures="100p"`:

| Scene seed | Outcome | Steps | Maximum normalized drawer opening |
| --- | --- | ---: | ---: |
| 0 | Success | 262 | 0.9624 |
| 1 | Success | 241 | 0.9788 |
| 2 | Failure | 750 | 0.0016 |

These three episodes are a small integration sample, **not an official benchmark estimate**. The official baseline used RoboCasa 1.0.0; this project uses 1.0.1 and a restricted smoke configuration. Policy sampling RNG was not separately controlled in these runs. See [results and limitations](docs/RESULTS.md).

[Seed 0: successful rollout](assets/rollouts/seed0_success.mp4) · [Seed 2: failed rollout](assets/rollouts/seed2_failure.mp4)

## Validated runtime

| Component | Configuration |
| --- | --- |
| Python | 3.11.16 |
| Inference | JAX/jaxlib 0.5.3, CUDA 12 plugin; NumPy 1.26.4 |
| Simulation | RoboCasa 1.0.1, robosuite 1.5.2, MuJoCo 3.3.1, NumPy 2.2.5 |
| GPU | One RTX 3090, 24 GiB; NVIDIA driver 535.179 |
| Rendering | Private OSMesa prefix, mesalib 25.0.5 `h57bcd07_2`, llvmpipe |
| Protocol | Three RGB cameras, 16D state, 50×12 action output; execute 5 actions per request |

The inference and simulation environments remain separate because their upstream dependency versions conflict. [Runtime setup and validation](docs/REPRODUCTION.md) records pinned sources, expected paths, and known dependency differences. Environment installation and large downloads are not automated by the launch scripts.

## Run in the existing project environment

From the project root, start the policy in terminal 1:

```bash
bash scripts/run_policy.sh 0
```

Wait for `server listening on 127.0.0.1:8000`, then run a single CPU-rendered rollout in terminal 2:

```bash
bash scripts/run_smoke.sh 0
```

The scripts assume the validated project-local environments, weights, tokenizer, rendering libraries, and assets already exist. They do not install or download anything. Outputs are written to an ignored timestamped directory below `outputs/`. Stop your policy server with Ctrl+C in terminal 1 after evaluation to release its GPU memory.

## Tests

```bash
mkdir -p cache/tmp
PYTHONPATH="$PWD/src:$PWD" CUDA_VISIBLE_DEVICES="" \
  cache/venvs/robocasa/bin/python -m pytest -q tests

PYTHONPATH="$PWD/src:$PWD" CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu \
  cache/venvs/openpi-inference/bin/python -m pytest -q tests/test_serve_policy.py
```

The three tests using actual OpenPI transforms are skipped in the simulation environment and should run in the inference environment. Tests use a fake tokenizer; they do not load weights or start a service.

## Laboratory boundaries

All environments, temporary files, weights, assets, and run outputs belong under this project. Runtime data and `sources/` are ignored by Git. Do not modify system drivers, CUDA, shell startup files, other users' environments, or upstream source trees. Use one available GPU and a loopback-only service. A momentary idle reading is not a GPU reservation; follow the laboratory's allocation rules. See [lab rules](docs/LAB_SAFETY.md).

## Attribution and license

This harness adapts the RoboCasa evaluation client from [robocasa-benchmark/openpi](https://github.com/robocasa-benchmark/openpi), pinned at `ca4c6d710db75e276bc7c866a57bd7e4aee5b6e8`, and reuses its π₀.₅ model and RoboCasa policy transforms. It relies on [RoboCasa](https://github.com/robocasa/robocasa) and [robosuite](https://github.com/ARISE-Initiative/robosuite).

Project code is distributed under [Apache-2.0](LICENSE), with upstream attribution retained in adapted files. Refer to the respective upstream terms for model weights, kitchen assets, and third-party software; those materials are not included here.
