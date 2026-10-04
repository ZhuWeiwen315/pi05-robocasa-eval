"""Dependency-free checks for the RoboCasa 1.0.1 evaluation protocol."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import TypeVar


ROBOCASA_VERSION = "1.0.1"
OFFICIAL_BASELINE_VERSION = "1.0.0"
OFFICIAL_CHECKPOINT = (
    "robocasa/robocasa365_checkpoints/pi05_pretrain_human300/"
    "multitask_learning/75000"
)

_T = TypeVar("_T")


def resolve_task_sets(
    task_set: str | Sequence[str], registry: Mapping[str, Sequence[str]]
) -> list[str]:
    """Expand named task sets in order, removing duplicate tasks."""
    names = [task_set] if isinstance(task_set, str) else list(task_set)
    if not names:
        raise ValueError("At least one --task-set is required")
    unknown = [name for name in names if name not in registry]
    if unknown:
        raise ValueError(f"Unknown task set(s): {', '.join(unknown)}")

    tasks: list[str] = []
    seen: set[str] = set()
    for name in names:
        for task in registry[name]:
            if task not in seen:
                seen.add(task)
                tasks.append(task)
    if not tasks:
        raise ValueError("Selected task sets contain no tasks")
    return tasks


def select_horizon(registered_horizon: int, robocasa_version: str) -> int:
    """Use the 1.0.1 registry value directly; it already includes the 1.5x change."""
    if robocasa_version != ROBOCASA_VERSION:
        raise ValueError(f"Expected RoboCasa {ROBOCASA_VERSION}, got {robocasa_version}")
    if isinstance(registered_horizon, bool) or not isinstance(registered_horizon, int) or registered_horizon <= 0:
        raise ValueError("Registered horizon must be a positive integer")
    return registered_horizon


def checked_action_chunk(chunk: Sequence[_T], replan_steps: int) -> Sequence[_T]:
    """Return the next plan only when the policy supplied enough actions."""
    if isinstance(replan_steps, bool) or not isinstance(replan_steps, int) or replan_steps <= 0:
        raise ValueError("replan_steps must be a positive integer")
    if len(chunk) < replan_steps:
        raise ValueError(
            f"Replan interval is {replan_steps} steps, but policy returned {len(chunk)} actions"
        )
    return chunk[:replan_steps]


def evaluation_path(root: Path, split: str, task: str, when: datetime) -> Path:
    """Keep the official evals_1.5/split/task layout with a unique run time."""
    if split not in ("pretrain", "target"):
        raise ValueError(f"Unsupported split: {split}")
    if not task or task in (".", "..") or Path(task).name != task or "\\" in task:
        raise ValueError(f"Invalid task name: {task}")
    return root / "evals_1.5" / split / task / when.strftime("%Y-%m-%d-%H-%M-%S-%f")


def success_stats(successes: int, episodes: int) -> dict[str, int | float | str]:
    """Retain official statistic keys and identify the changed simulator protocol."""
    if episodes <= 0 or successes < 0 or successes > episodes:
        raise ValueError("Episode and success counts are inconsistent")
    return {
        "num_episodes": episodes,
        "success_rate": successes / episodes,
        "robocasa_version": ROBOCASA_VERSION,
        "official_baseline_robocasa_version": OFFICIAL_BASELINE_VERSION,
        "official_baseline_checkpoint": OFFICIAL_CHECKPOINT,
    }
