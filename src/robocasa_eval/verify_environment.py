"""Verify the intentionally slim RoboCasa 1.0.1 simulation environment.

Run with: python -m robocasa_eval.verify_environment
This imports packages only; it does not construct an environment or use a GPU.
"""

from __future__ import annotations

import importlib
from importlib import metadata
from pathlib import Path
import sys

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


REQUIRED_IMPORTS = (
    ("numpy", "numpy", "2.2.5"),
    ("mujoco", "mujoco", "3.3.1"),
    ("cv2", "opencv-python-headless", "5.0.0.93"),
    ("robosuite", "robosuite", "1.5.2"),
    ("robocasa", "robocasa", "1.0.1"),
    ("gymnasium", "gymnasium", "0.29.1"),
    ("openpi_client", "openpi-client", "0.1.0"),
    ("robocasa_eval", "pi05-robocasa-eval", "0.1.0"),
)

EXPECTED_DIFFERENCES = frozenset(
    {
        ("robosuite", "opencv-python"),
        ("robocasa", "opencv-python"),
        ("robocasa", "tianshou==0.4.10"),
        ("robocasa", "lerobot==0.3.3"),
    }
)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def forbidden_distributions(names: set[str]) -> set[str]:
    return {
        name
        for name in names
        if name in {"torch", "jax", "jaxlib", "lerobot", "tianshou", "cupy", "triton"}
        or name.startswith("nvidia-")
        or "cuda" in name
        or "cudnn" in name
    }


def dependency_differences(
    installed: dict[str, metadata.Distribution],
) -> set[tuple[str, str]]:
    """Return every unsatisfied active base requirement, including version drift."""
    problems: set[tuple[str, str]] = set()
    for parent, dist in installed.items():
        for raw in dist.requires or ():
            requirement = Requirement(raw)
            if requirement.marker and not requirement.marker.evaluate({"extra": ""}):
                continue
            required_name = canonicalize_name(requirement.name)
            target = installed.get(required_name)
            if target is None or not requirement.specifier.contains(target.version):
                problems.add((parent, raw))
    return problems


def verify_differences(problems: set[tuple[str, str]]) -> None:
    unexpected = problems - EXPECTED_DIFFERENCES
    missing = EXPECTED_DIFFERENCES - problems
    if unexpected or missing:
        raise RuntimeError(
            f"Dependency differences changed; unexpected={sorted(unexpected)}, "
            f"missing_expected={sorted(missing)}"
        )


def verify() -> None:
    expected_venv = PROJECT_ROOT / "cache/venvs/robocasa"
    if Path(sys.prefix).resolve() != expected_venv:
        raise RuntimeError(f"Expected project venv {expected_venv}, got {sys.prefix}")
    installed = {
        canonicalize_name(dist.metadata["Name"]): dist
        for dist in metadata.distributions()
    }
    forbidden = forbidden_distributions(set(installed))
    if forbidden:
        raise RuntimeError(f"Forbidden packages installed: {sorted(forbidden)}")
    if "opencv-python-headless" not in installed or "opencv-python" in installed:
        raise RuntimeError("Expected only opencv-python-headless")

    print(f"Installed distributions: {len(installed)}; forbidden: none")
    for module_name, dist_name, expected_version in REQUIRED_IMPORTS:
        module = importlib.import_module(module_name)
        actual_version = installed[canonicalize_name(dist_name)].version
        if actual_version != expected_version:
            raise RuntimeError(f"{dist_name}: expected {expected_version}, got {actual_version}")
        loaded_path = Path(module.__file__).resolve()
        if not loaded_path.is_relative_to(PROJECT_ROOT):
            raise RuntimeError(f"{module_name} loaded outside project: {loaded_path}")
        print(f"{module_name} {actual_version} {loaded_path}")

    problems = dependency_differences(installed)
    verify_differences(problems)
    print("Dependency differences: exactly 4 approved upstream declarations")
    for parent, requirement in sorted(problems):
        print(f"  {parent} requires {requirement}")


def main() -> None:
    verify()


if __name__ == "__main__":
    main()
