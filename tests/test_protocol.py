"""Checks that can run without simulator, model, or GPU dependencies."""

from contextlib import redirect_stderr
from datetime import datetime
from io import StringIO
from pathlib import Path
import unittest
from unittest.mock import Mock

from robocasa_eval.main import make_task_env, parse_args, resolve_evaluation_tasks
from robocasa_eval.protocol import (
    OFFICIAL_BASELINE_VERSION,
    OFFICIAL_CHECKPOINT,
    checked_action_chunk,
    evaluation_path,
    resolve_task_sets,
    select_horizon,
    success_stats,
)


class ProtocolTests(unittest.TestCase):
    def test_cli_defaults_to_loopback_without_simulator_imports(self) -> None:
        args = parse_args(["--task-set", "atomic_seen"])
        self.assertEqual(args.host, "127.0.0.1")
        self.assertEqual(args.task_set, ["atomic_seen"])
        self.assertIsNone(args.smoke_lightwheel_task)
        self.assertEqual(args.num_trials, 50)

    def test_smoke_cli_uses_verified_robocasa_values(self) -> None:
        args = parse_args(["--smoke-lightwheel-task", "OpenDrawer"])
        self.assertEqual(args.smoke_lightwheel_task, "OpenDrawer")
        self.assertIsNone(args.task_set)
        self.assertEqual(args.num_trials, 1)
        self.assertEqual(
            parse_args(["--smoke-lightwheel-task", "OpenDrawer", "--num-trials", "2"]).num_trials,
            2,
        )
        self.assertEqual(resolve_evaluation_tasks(args, {"all_tasks": ["OpenDrawer"]}), ["OpenDrawer"])
        gym = Mock()
        make_task_env(gym, "OpenDrawer", args)
        gym.make.assert_called_once_with(
            "robocasa/OpenDrawer", split="pretrain", seed=7,
            obj_registries=("lightwheel",), generative_textures="100p",
        )

    def test_default_gym_make_kwargs_remain_unchanged(self) -> None:
        args = parse_args(["--task-set", "atomic_seen"])
        self.assertEqual(
            resolve_evaluation_tasks(args, {"atomic_seen": ["A", "B"]}), ["A", "B"]
        )
        gym = Mock()
        make_task_env(gym, "OpenDrawer", args)
        gym.make.assert_called_once_with("robocasa/OpenDrawer", split="pretrain", seed=7)

    def test_smoke_requires_exactly_one_resolved_task(self) -> None:
        args = parse_args(["--smoke-lightwheel-task", "Missing"])
        with self.assertRaises(ValueError):
            resolve_evaluation_tasks(args, {"all_tasks": ["OpenDrawer"]})
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            parse_args(["--task-set", "atomic_seen", "--smoke-lightwheel-task", "OpenDrawer"])

    def test_task_sets_expand_and_deduplicate(self) -> None:
        registry = {"atomic_seen": ["A", "B"], "target50": ["B", "C"]}
        self.assertEqual(resolve_task_sets(["atomic_seen", "target50"], registry), ["A", "B", "C"])
        self.assertEqual(resolve_task_sets("atomic_seen", registry), ["A", "B"])

    def test_task_sets_reject_missing_or_unknown(self) -> None:
        for names in ([], ["missing"], ["atomic_seen", "missing"]):
            with self.subTest(names=names), self.assertRaises(ValueError):
                resolve_task_sets(names, {"atomic_seen": ["A"]})

    def test_101_horizon_is_not_scaled_again(self) -> None:
        self.assertEqual(select_horizon(450, "1.0.1"), 450)
        with self.assertRaises(ValueError):
            select_horizon(300, "1.0.0")
        with self.assertRaises(ValueError):
            select_horizon(0, "1.0.1")

    def test_action_chunk_respects_replanning_interval(self) -> None:
        self.assertEqual(checked_action_chunk([1, 2, 3], 2), [1, 2])
        with self.assertRaises(ValueError):
            checked_action_chunk([1], 2)
        with self.assertRaises(ValueError):
            checked_action_chunk([1], 0)

    def test_output_path_preserves_official_layout(self) -> None:
        result = evaluation_path(Path("outputs"), "pretrain", "A", datetime(2026, 10, 4, 12, 3, 4, 5))
        self.assertEqual(
            result,
            Path("outputs/evals_1.5/pretrain/A/2026-10-04-12-03-04-000005"),
        )
        with self.assertRaises(ValueError):
            evaluation_path(Path("outputs"), "pretrain", "../outside", datetime.now())

    def test_stats_preserve_official_fields_and_mark_version(self) -> None:
        stats = success_stats(2, 4)
        self.assertEqual(stats["num_episodes"], 4)
        self.assertEqual(stats["success_rate"], 0.5)
        self.assertEqual(stats["robocasa_version"], "1.0.1")
        self.assertEqual(stats["official_baseline_robocasa_version"], OFFICIAL_BASELINE_VERSION)
        self.assertEqual(stats["official_baseline_checkpoint"], OFFICIAL_CHECKPOINT)
        with self.assertRaises(ValueError):
            success_stats(5, 4)


if __name__ == "__main__":
    unittest.main()
