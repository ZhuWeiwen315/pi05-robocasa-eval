"""Offline checks for the slim environment's exact metadata exception set."""

import unittest

from robocasa_eval.verify_environment import (
    EXPECTED_DIFFERENCES,
    dependency_differences,
    forbidden_distributions,
    verify_differences,
)


class FakeDistribution:
    def __init__(self, version: str, requires: list[str] | None = None):
        self.version = version
        self.requires = requires or []


class EnvironmentRuleTests(unittest.TestCase):
    def test_exact_four_are_accepted(self) -> None:
        verify_differences(set(EXPECTED_DIFFERENCES))

    def test_fifth_or_changed_requirement_fails(self) -> None:
        for problems in (
            set(EXPECTED_DIFFERENCES) | {("another", "missing")},
            (set(EXPECTED_DIFFERENCES) - {("robocasa", "tianshou==0.4.10")})
            | {("robocasa", "tianshou>=0.4.10")},
            set(EXPECTED_DIFFERENCES) - {("robosuite", "opencv-python")},
        ):
            with self.subTest(problems=problems), self.assertRaises(RuntimeError):
                verify_differences(problems)

    def test_scan_checks_missing_and_wrong_versions(self) -> None:
        installed = {
            "owner": FakeDistribution("1.0", ["present>=2", "missing", "optional; extra == 'gpu'"]),
            "present": FakeDistribution("1.0"),
        }
        self.assertEqual(
            dependency_differences(installed),
            {("owner", "present>=2"), ("owner", "missing")},
        )

    def test_forbidden_package_names(self) -> None:
        names = {"numpy", "torch", "jaxlib", "lerobot", "tianshou", "nvidia-cublas-cu12", "cupy"}
        self.assertEqual(forbidden_distributions(names), names - {"numpy"})


if __name__ == "__main__":
    unittest.main()
