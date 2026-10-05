"""Offline checks for the lightweight pi0.5 policy service."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import copy
import importlib.util
from io import StringIO
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

from robocasa_eval.serve_policy import (
    ACTION_DIM,
    ACTION_HORIZON,
    IMAGE_SIZE,
    LOOPBACK_HOST,
    MAX_TOKEN_LEN,
    OpenPIComponents,
    build_policy_transforms,
    create_websocket_server,
    load_checkpoint_payload,
    load_openpi_components,
    make_pi05_model_config,
    parse_args,
    resolve_checkpoint_layout,
    validate_loopback_host,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_MODULES = (
    "openpi.training.config",
    "openpi.training.checkpoints",
    "openpi.groot_utils",
    "robocasa",
    "robosuite",
    "mujoco",
    "dm_control",
)


def _subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["JAX_PLATFORMS"] = "cpu"
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["TMPDIR"] = str(PROJECT_ROOT / "cache/tmp")
    return env


class ServiceIsolationTests(unittest.TestCase):
    def test_service_module_import_has_no_training_or_simulator_modules(self) -> None:
        code = (
            "import sys; import robocasa_eval.serve_policy; "
            f"forbidden={FORBIDDEN_MODULES!r}; "
            "bad=[name for name in sys.modules if any(name == x or name.startswith(x + '.') for x in forbidden)]; "
            "assert not bad, bad"
        )
        result = subprocess.run(
            [sys.executable, "-B", "-c", code],
            cwd=PROJECT_ROOT,
            env=_subprocess_env(),
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_help_does_not_load_openpi(self) -> None:
        result = subprocess.run(
            [sys.executable, "-B", "-m", "robocasa_eval.serve_policy", "--help"],
            cwd=PROJECT_ROOT,
            env=_subprocess_env(),
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--checkpoint", result.stdout)
        self.assertIn("--host", result.stdout)

    def test_cli_and_server_reject_non_loopback_hosts(self) -> None:
        for host in ("0.0.0.0", "localhost", "::1", "192.168.1.2"):
            with self.subTest(host=host), self.assertRaises(ValueError):
                validate_loopback_host(host)
            with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                parse_args(["--checkpoint", "/unused", "--host", host])

        args = parse_args(["--checkpoint", "/unused"])
        self.assertEqual(args.host, LOOPBACK_HOST)
        policy = SimpleNamespace(metadata={"name": "test"})
        factory = Mock(return_value=object())
        create_websocket_server(
            policy,
            host=args.host,
            port=args.port,
            server_factory=factory,
        )
        factory.assert_called_once_with(
            policy=policy,
            host=LOOPBACK_HOST,
            port=8000,
            metadata={"name": "test"},
        )

    def test_checkpoint_loader_uses_only_params_and_assets_norm_stats(self) -> None:
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "cache/tmp") as tmp:
            root = Path(tmp) / "75000"
            (root / "params").mkdir(parents=True)
            (root / "assets").mkdir()
            (root / "assets/norm_stats.json").write_text("{}", encoding="utf-8")
            layout = resolve_checkpoint_layout(root)

            restored = object()
            stats = {"state": object()}
            model_module = SimpleNamespace(restore_params=Mock(return_value=restored))
            normalize_module = SimpleNamespace(load=Mock(return_value=stats))
            dtype = object()
            components = OpenPIComponents(
                jnp=SimpleNamespace(bfloat16=dtype),
                model=model_module,
                normalize=normalize_module,
                pi0_config=None,
                policy=None,
                robocasa_policy=None,
                tokenizer=None,
                transforms=None,
                websocket_policy_server=None,
            )
            self.assertEqual(load_checkpoint_payload(layout, components), (restored, stats))
            model_module.restore_params.assert_called_once_with(root / "params", dtype=dtype)
            normalize_module.load.assert_called_once_with(root / "assets")

            (root / "assets/norm_stats.json").unlink()
            with self.assertRaises(FileNotFoundError):
                resolve_checkpoint_layout(root)

        with self.assertRaises(ValueError):
            resolve_checkpoint_layout("gs://bucket/checkpoint")


@unittest.skipUnless(importlib.util.find_spec("openpi"), "OpenPI inference environment required")
class UpstreamTransformParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.components = load_openpi_components()

    def test_component_loader_avoids_training_and_simulator_modules(self) -> None:
        bad = [
            name
            for name in sys.modules
            if any(name == item or name.startswith(item + ".") for item in FORBIDDEN_MODULES)
        ]
        self.assertEqual(bad, [])

    def test_three_cameras_state_padding_and_upstream_transform_order(self) -> None:
        components = self.components
        model_config = make_pi05_model_config(components)
        self.assertEqual(model_config.action_dim, ACTION_DIM)
        self.assertEqual(model_config.action_horizon, ACTION_HORIZON)
        self.assertEqual(model_config.max_token_len, MAX_TOKEN_LEN)

        norm_stats = {
            "state": components.normalize.NormStats(
                mean=np.zeros(ACTION_DIM), std=np.ones(ACTION_DIM)
            ),
            "actions": components.normalize.NormStats(
                mean=np.arange(12, dtype=np.float64), std=np.ones(12)
            ),
        }

        class FakeTokenizer:
            def __init__(self) -> None:
                self.calls = []

            def tokenize(self, prompt, state):
                self.calls.append((prompt, np.asarray(state).copy()))
                return np.arange(MAX_TOKEN_LEN, dtype=np.int32), np.ones(MAX_TOKEN_LEN, dtype=bool)

        tokenizer = FakeTokenizer()
        pipeline = build_policy_transforms(
            components=components,
            model_config=model_config,
            norm_stats=norm_stats,
            tokenizer=tokenizer,
        )
        self.assertEqual(
            [type(item) for item in pipeline.inputs],
            [
                components.transforms.InjectDefaultPrompt,
                components.robocasa_policy.RobocasaInputs,
                components.transforms.Normalize,
                components.transforms.InjectDefaultPrompt,
                components.transforms.ResizeImages,
                components.transforms.TokenizePrompt,
                components.transforms.PadStatesAndActions,
            ],
        )
        self.assertEqual(
            [type(item) for item in pipeline.outputs],
            [components.transforms.Unnormalize, components.robocasa_policy.RobocasaOutputs],
        )

        state = np.arange(16, dtype=np.float64)
        observation = {
            "observation/state": state,
            # Deliberately use three different source sizes so this also checks
            # the fixed upstream ResizeImages(224, 224) transform.
            "observation/image": np.full((64, 64, 3), 11, dtype=np.uint8),
            "observation/wrist_image": np.full((48, 48, 3), 22, dtype=np.uint8),
            "observation/right_image": np.full((32, 32, 3), 33, dtype=np.uint8),
            "prompt": "open the drawer",
        }
        transformed = components.transforms.compose(pipeline.inputs)(copy.deepcopy(observation))
        self.assertEqual(set(transformed["image"]), {"base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"})
        for image in transformed["image"].values():
            self.assertEqual(image.shape, (IMAGE_SIZE, IMAGE_SIZE, 3))
            self.assertEqual(image.dtype, np.uint8)
        self.assertTrue(np.all(transformed["image"]["base_0_rgb"] == 11))
        self.assertTrue(np.all(transformed["image"]["left_wrist_0_rgb"] == 22))
        self.assertTrue(np.all(transformed["image"]["right_wrist_0_rgb"] == 33))
        self.assertTrue(all(transformed["image_mask"].values()))
        self.assertEqual(transformed["state"].shape, (ACTION_DIM,))
        np.testing.assert_allclose(transformed["state"][:16], state / (1.0 + 1e-6))
        np.testing.assert_array_equal(transformed["state"][16:], np.zeros(ACTION_DIM - 16))
        self.assertEqual(tokenizer.calls[0][0], "open the drawer")
        np.testing.assert_allclose(tokenizer.calls[0][1], transformed["state"])

    def test_output_is_exactly_twelve_robocasa_action_dimensions(self) -> None:
        components = self.components
        model_config = make_pi05_model_config(components)
        norm_stats = {
            "state": components.normalize.NormStats(
                mean=np.zeros(ACTION_DIM), std=np.ones(ACTION_DIM)
            ),
            "actions": components.normalize.NormStats(
                mean=np.arange(12, dtype=np.float64), std=np.ones(12)
            ),
        }
        pipeline = build_policy_transforms(
            components=components,
            model_config=model_config,
            norm_stats=norm_stats,
            tokenizer=object(),
        )
        outputs = components.transforms.compose(pipeline.outputs)(
            {
                "state": np.zeros(ACTION_DIM),
                "actions": np.zeros((ACTION_HORIZON, ACTION_DIM)),
            }
        )
        self.assertEqual(outputs["actions"].shape, (ACTION_HORIZON, 12))
        np.testing.assert_allclose(
            outputs["actions"],
            np.broadcast_to(np.arange(12, dtype=np.float64), (ACTION_HORIZON, 12)),
        )


if __name__ == "__main__":
    with redirect_stdout(StringIO()):
        unittest.main()
