"""Checks that can run without simulator, model, or GPU dependencies."""

from datetime import datetime
from pathlib import Path
import unittest

from robocasa_eval.main import parse_args
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
