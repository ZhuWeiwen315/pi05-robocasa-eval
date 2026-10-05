# Replanning interval ablation

Date: 2026-10-05.

OpenDrawer, RoboCasa 1.0.1, pretrain, Lightwheel-only objects, 100% generated textures, scene seeds 0–9, policy seed 0, one episode per scene, 750-step horizon. A fresh policy server was started before each episode.

Only the executed action interval changed from 5 to 1. The evaluation timeout increased to accommodate runtime; the simulation horizon remained unchanged.

| Scene seed | replan=5 success | Steps | replan=1 success | Steps |
| --- | --- | ---: | --- | ---: |
| 0 | Yes | 253 | Yes | 295 |
| 1 | Yes | 259 | Yes | 321 |
| 2 | Yes | 210 | Yes | 232 |
| 3 | No | 750 | No | 750 |
| 4 | Yes | 355 | No | 750 |
| 5 | No | 750 | No | 750 |
| 6 | No | 750 | Yes | 369 |
| 7 | No | 750 | No | 750 |
| 8 | Yes | 224 | No | 750 |
| 9 | No | 750 | Yes | 416 |

## Findings

- Both settings succeeded in 5/10 episodes.
- Seeds 0, 1 and 2 succeeded with both settings.
- Seeds 4 and 8 regressed with replan=1.
- Seeds 6 and 9 improved with replan=1.
- Seeds 3, 5 and 7 failed with both settings.
- The three jointly successful scenes required more steps with replan=1: 253→295, 259→321 and 210→232.

Keep replan=5 as the default. This sample shows no aggregate success improvement from replan=1; it does not establish statistical equivalence.

## Interpretation limits

More frequent replanning changes the number of policy requests and hence the progression of sampling RNG. This is an execution-interval ablation, not a noise-matched isolation of feedback frequency.

One episode per scene and one policy seed are insufficient for broad performance claims. Results belong to this restricted smoke configuration, not the official RoboCasa benchmark. No training or fine-tuning occurred.

Next: compare videos for seed 4 (regression) and seed 6 (improvement) before proposing another change.

## Reproduction

With the prepared project environment:

```bash
cache/venvs/robocasa/bin/python scripts/run_replan1_ablation.py
```

The script refuses an existing experiment output directory.
