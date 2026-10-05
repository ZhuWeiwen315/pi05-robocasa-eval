# Controlled OpenDrawer baseline

Date: 2026-10-05.

RoboCasa 1.0.1, pretrain split, Lightwheel-only objects, 100% generated textures, 224×224 policy images, 5 executed actions per request, 750-step horizon.

Scene seeds 0–9; policy seed 0; one episode per scene. The policy server was restarted before every episode.

**Result: 5/10 successful episodes (50% in this smoke configuration).** This is not an official RoboCasa benchmark result or an improvement claim.

| Scene seed | Policy seed | Success | Steps | Peak opening | Final opening |
| --- | --- | --- | ---: | ---: | ---: |
| 0 | 0 | Yes | 253 | 0.958578 | 0.958578 |
| 1 | 0 | Yes | 259 | 0.950623 | 0.950623 |
| 2 | 0 | Yes | 210 | 0.968157 | 0.968157 |
| 3 | 0 | No | 750 | 0.000473 | 0.000010 |
| 4 | 0 | Yes | 355 | 0.975965 | 0.975965 |
| 5 | 0 | No | 750 | 0.413009 | 0.413009 |
| 6 | 0 | No | 750 | 0.224809 | 0.224809 |
| 7 | 0 | No | 750 | 0.131278 | 0.026190 |
| 8 | 0 | Yes | 224 | 0.953136 | 0.953136 |
| 9 | 0 | No | 750 | 0.009329 | 0.000099 |

## Reproducibility check

Two independent scene-seed-0, policy-seed-0 episodes both succeeded after 253 steps. Recorded actions, policy states, target drawer opening and success flags were exactly identical. This check applies to the tested configuration and hardware; it does not guarantee determinism across other platforms.

## Observed failure patterns

- Seeds 3 and 9: maximum target opening below 0.01.
- Seeds 5 and 6: partial opening, ending near 0.413 and 0.225.
- Seed 7: peak opening near 0.131, falling to 0.026.

These describe trajectories, not verified failure causes.

Earlier uncontrolled smoke episodes are kept separate. No training or fine-tuning was performed.

## Run the same batch

From a prepared project root:

```bash
cache/venvs/robocasa/bin/python scripts/run_controlled_baseline.py
```

The script uses GPU 0 for inference and CPU OSMesa for simulation. It refuses an existing output directory or occupied service port. It stops only its own child processes.

The service initializer uses the verified `_rng` attribute of the pinned upstream Policy implementation. Recheck this interface when changing the upstream revision.
