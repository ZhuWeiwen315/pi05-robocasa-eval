# SPDX-License-Identifier: Apache-2.0
"""Evaluate π₀.₅ on RoboCasa 1.0.1 through the official WebSocket protocol.

Adapted from robocasa-benchmark/openpi, commit
ca4c6d710db75e276bc7c866a57bd7e4aee5b6e8,
examples/robocasa/main.py (Apache-2.0; see LICENSE in this package).
The official leaderboard checkpoint was evaluated with RoboCasa 1.0.0.
Results from this 1.0.1 protocol must be reported separately.
"""

from __future__ import annotations

import argparse
import collections
from datetime import datetime
import json
import logging
from pathlib import Path

from .assets import activate_project_assets
from .protocol import (
    ROBOCASA_VERSION,
    checked_action_chunk,
    evaluation_path,
    resolve_task_sets,
    select_horizon,
    success_stats,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATE_KEYS = (
    "state.end_effector_position_relative",
    "state.end_effector_rotation_relative",
    "state.base_position",
    "state.base_rotation",
    "state.gripper_qpos",
)
IMAGE_KEYS = (
    ("video.robot0_agentview_left", "observation/image"),
    ("video.robot0_eye_in_hand", "observation/wrist_image"),
    ("video.robot0_agentview_right", "observation/right_image"),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-set", "--task_set", nargs="+", required=True)
    parser.add_argument("--split", choices=("pretrain", "target"), default="pretrain")
    parser.add_argument("--num-trials", type=int, default=50)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--log-dir", type=Path, default=PROJECT_ROOT / "outputs")
    return parser.parse_args(argv)


def evaluate(args: argparse.Namespace) -> None:
    activate_project_assets()
    # Load simulator dependencies only when evaluation is explicitly started.
    import gymnasium as gym
    import imageio
    import numpy as np
    from openpi_client import image_tools
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    import robocasa
    from robocasa.utils.dataset_registry import TASK_SET_REGISTRY
    from robocasa.utils.dataset_registry_utils import get_task_horizon
    from robocasa.utils.env_utils import convert_action
    import tqdm

    if robocasa.__version__ != ROBOCASA_VERSION:
        raise RuntimeError(f"Expected RoboCasa {ROBOCASA_VERSION}, got {robocasa.__version__}")
    if args.num_trials <= 0 or args.resize_size <= 0:
        raise ValueError("num_trials and resize_size must be positive")
    if not 1 <= args.port <= 65535:
        raise ValueError("port must be between 1 and 65535")
    if args.replan_steps <= 0:
        raise ValueError("replan_steps must be positive")

    np.random.seed(args.seed)
    task_names = resolve_task_sets(args.task_set, TASK_SET_REGISTRY)
    for task in task_names:
        horizon = select_horizon(get_task_horizon(task), robocasa.__version__)
        log_path = evaluation_path(args.log_dir, args.split, task, datetime.now())
        if any(log_path.parent.rglob("stats.json")):
            logging.info("%s/%s: stats.json exists; skipping", task, args.split)
            continue
        log_path.mkdir(parents=True, exist_ok=False)
        client = WebsocketClientPolicy(args.host, args.port)
        env = gym.make(f"robocasa/{task}", split=args.split, seed=args.seed)
        successes = 0
        try:
            for episode_idx in tqdm.tqdm(range(args.num_trials), desc=task):
                obs, _ = env.reset()
                prompt = obs["annotation.human.task_description"]
                action_plan = collections.deque()
                frames = []
                succeeded = False

                for step in range(horizon):
                    if not action_plan:
                        request = {}
                        for observation_key, policy_key in IMAGE_KEYS:
                            image = np.ascontiguousarray(obs[observation_key])
                            request[policy_key] = image_tools.convert_to_uint8(
                                image_tools.resize_with_pad(image, args.resize_size, args.resize_size)
                            )
                        request["observation/state"] = np.concatenate(
                            tuple(obs[key] for key in STATE_KEYS), axis=0
                        )
                        request["prompt"] = prompt
                        action_chunk = client.infer(request)["actions"]
                        action_plan.extend(checked_action_chunk(action_chunk, args.replan_steps))

                    policy_action = np.asarray(action_plan.popleft())
                    if policy_action.shape != (12,) or not np.isfinite(policy_action).all():
                        raise ValueError(f"Expected a finite 12D policy action, got shape {policy_action.shape}")
                    obs, _, _, _, info = env.step(convert_action(policy_action))
                    succeeded = bool(info["success"])

                    frame = image_tools.convert_to_uint8(np.ascontiguousarray(env.render()))
                    if step % 2 == 0 or step == horizon - 1 or succeeded:
                        frames.append(frame)
                    if succeeded:
                        successes += 1
                        break

                suffix = "success" if succeeded else "failure"
                imageio.mimwrite(
                    log_path / f"rollout_{episode_idx}_{suffix}.mp4",
                    [np.asarray(frame) for frame in frames],
                    fps=20,
                )
                logging.info(
                    "%s episode %d/%d: %s; success rate %.3f",
                    task, episode_idx + 1, args.num_trials, suffix, successes / (episode_idx + 1),
                )
        finally:
            env.close()

        stats = success_stats(successes, args.num_trials)
        (log_path / "stats.json").write_text(json.dumps(stats, indent=4) + "\n", encoding="utf-8")
        logging.info("%s: %s", task, stats)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO)
    evaluate(parse_args(argv))


if __name__ == "__main__":
    main()
