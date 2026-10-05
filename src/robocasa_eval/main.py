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


def _capture_trace_state(env, obs):
    """Capture policy-visible state and cached raw drawer observations."""
    import numpy as np
    policy_state = {
        key: np.asarray(obs[key]).tolist()
        for key in STATE_KEYS
    }

    # gym.make wrappers resolve to RoboCasaGymEnv; its .env is the
    # underlying simulator. Read cached observations without forcing sensors.
    raw_env = env.unwrapped.env
    raw_obs = raw_env._get_observations(force_update=False)
    raw_keys = ("drawer_obj_pos", "drawer_obj_to_robot0_eef_pos")
    simulator_state = {
        key: np.asarray(raw_obs[key]).tolist()
        for key in raw_keys
        if key in raw_obs
    }
    drawer = getattr(raw_env, "drawer", None)
    target_drawer = None
    if drawer is not None:
        target_drawer = {
            "fixture_name": str(getattr(drawer, "name", type(drawer).__name__)),
            "door_state": {
                str(key): float(value)
                for key, value in drawer.get_door_state(env=raw_env).items()
            },
        }
    return {
        "target_drawer": target_drawer,
        "policy_observation": policy_state,
        "simulator_observation": simulator_state,
        "missing_simulator_keys": [key for key in raw_keys if key not in raw_obs],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    tasks = parser.add_mutually_exclusive_group(required=True)
    tasks.add_argument("--task-set", "--task_set", nargs="+")
    tasks.add_argument(
        "--smoke-lightwheel-task", metavar="TASK",
        help="Evaluate one task with only Lightwheel objects and 100%% generated textures",
    )
    parser.add_argument("--split", choices=("pretrain", "target"), default="pretrain")
    parser.add_argument("--num-trials", type=int)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--log-dir", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument("--trace-actions", action="store_true", help="write per-step actions and drawer state to JSONL")
    args = parser.parse_args(argv)
    if args.num_trials is None:
        args.num_trials = 1 if args.smoke_lightwheel_task is not None else 50
    return args


def resolve_evaluation_tasks(args: argparse.Namespace, registry) -> list[str]:
    if args.smoke_lightwheel_task is not None:
        if args.smoke_lightwheel_task not in registry["all_tasks"]:
            raise ValueError(f"Unknown RoboCasa task: {args.smoke_lightwheel_task}")
        return [args.smoke_lightwheel_task]
    return resolve_task_sets(args.task_set, registry)


def make_task_env(gym, task: str, args: argparse.Namespace):
    """Pass only the verified RoboCasa 1.0.1 smoke kwargs to gym.make."""
    kwargs = {"split": args.split, "seed": args.seed}
    if args.smoke_lightwheel_task is not None:
        kwargs.update(obj_registries=("lightwheel",), generative_textures="100p")
    return gym.make(f"robocasa/{task}", **kwargs)


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
    task_names = resolve_evaluation_tasks(args, TASK_SET_REGISTRY)
    for task in task_names:
        horizon = select_horizon(get_task_horizon(task), robocasa.__version__)
        log_path = evaluation_path(args.log_dir, args.split, task, datetime.now())
        log_path.mkdir(parents=True, exist_ok=False)
        client = WebsocketClientPolicy(args.host, args.port)
        env = make_task_env(gym, task, args)
        successes = 0
        try:
            for episode_idx in tqdm.tqdm(range(args.num_trials), desc=task):
                obs, _ = env.reset()
                prompt = obs["annotation.human.task_description"]
                action_plan = collections.deque()
                frames = []
                succeeded = False
                trace_path = log_path / f"rollout_{episode_idx}_trace.jsonl"
                if args.trace_actions:
                    trace_path.write_text("", encoding="utf-8")

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
                    pre_state = _capture_trace_state(env, obs) if args.trace_actions else None
                    env_action = convert_action(policy_action)
                    obs, _, _, _, info = env.step(env_action)
                    succeeded = bool(info["success"])
                    if args.trace_actions:
                        converted_action = (
                            {str(key): np.asarray(value).tolist() for key, value in env_action.items()}
                            if hasattr(env_action, "items")
                            else np.asarray(env_action).tolist()
                        )
                        post_state = _capture_trace_state(env, obs)
                        trace = {
                            "seed": args.seed,
                            "episode": episode_idx,
                            "step": step,
                            "prompt": prompt,
                            "policy_action": policy_action.tolist(),
                            "env_action": converted_action,
                            "pre": pre_state,
                            "post": post_state,
                            "success": succeeded,
                        }
                        with trace_path.open("a", encoding="utf-8") as trace_file:
                            trace_file.write(json.dumps(trace, allow_nan=False) + "\n")

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
