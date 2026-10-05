# Initial integration results

## Protocol and evidence

These results were observed on EVA12 on 2026-10-05 with the official `pi05_pretrain_human300` checkpoint at step 75000, RoboCasa 1.0.1, `pretrain`, `OpenDrawer`, Lightwheel-only objects, 100% generated textures, 224×224 policy images, five executed actions per replanning request, and a 750-step horizon. Inference used one RTX 3090; rendering used OSMesa/llvmpipe on CPU.

The task instruction came from the environment. Success came from the upstream environment's success check, returned as `info["success"]`; no manual success override was used. For this task, the inspected upstream success condition requires all target drawer door-state values to reach at least 0.95.

| Scene seed | Instruction side | Outcome | Steps | Peak opening | Final opening |
| --- | --- | --- | ---: | ---: | ---: |
| 0 | Left | Success | 262 | 0.9624 | 0.9624 |
| 1 | Right | Success | 241 | 0.9788 | 0.9788 |
| 2 | Right | Failure | 750 | 0.0016 | 0.0000 |

Two of three episodes succeeded. This count is descriptive only: it does not establish a reliable success rate, an official benchmark reproduction, or an improvement over another policy. The table is transcribed from terminal summaries; complete raw traces are not included in this release. Example videos for seeds 0 and 2 are included under `assets/rollouts/`.

A separate diagnostic episode with scene seed 7 failed after 750 steps. For target fixture `stack_2_left_group_2`, opening started at 0.000693, peaked at 0.361490 at step 440, and ended at 0.178626. It is kept separate from the three-episode sample.

## Failure observations

Video inspection of seed 2 suggests that the gripper engages a lower or neighboring cabinet handle rather than opening the intended drawer. This is a visual hypothesis, not a verified contact-level diagnosis. Seed 7 instead shows partial opening followed by regression. Multiple failure types may be present.

`drawer_obj_pos` is an object observation and is not the drawer's opening measurement. Traces distinguish the object observations from `target_drawer.door_state`, obtained from the actual target fixture.

## Limitations

- The official checkpoint baseline was evaluated with RoboCasa 1.0.0. This harness uses 1.0.1 and avoids applying the horizon multiplier twice.
- The smoke configuration restricts object libraries and uses generated textures; it is not the full evaluation protocol.
- Scene seed does not fix the policy's sampling RNG. The persistent service advances its sampling state between requests. Exact trajectory reproducibility and paired comparisons require explicit policy RNG control and recorded reset behavior.
- Only one episode per seed is reported. No fine-tuning, adaptation, controlled ablation, or broad task sweep has been performed.
- RoboCasa's Gym wrapper declares `state.base_position` within [-1, 1], although observed positions can exceed that range. The resulting observation-space warning does not clip the actual observation. The project has not patched this upstream declaration.

## Next experiment

First record and control both scene and policy sampling seeds. Then use the same fixed protocol for a prespecified collection of seeds, save run metadata and raw traces, and summarize success counts and drawer-opening progress. Inspect target-handle contacts before attributing failures to perception or control. Any adaptation should be compared against that fixed baseline.
