# SPDX-License-Identifier: Apache-2.0
"""Serve the fixed RoboCasa pi0.5 policy without importing simulation code.

This module reuses the pure inference components from robocasa-benchmark/openpi
commit ca4c6d710db75e276bc7c866a57bd7e4aee5b6e8.  It intentionally does not
import ``openpi.training.config``, ``openpi.training.checkpoints``, Groot data
loaders, RoboCasa, robosuite, or MuJoCo.

The transform order mirrors ``openpi.policies.policy_config`` and the active
``LeRobotRobocasaDataConfig`` in ``openpi.training.config`` at that commit.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
import dataclasses
import importlib
import logging
import os
from pathlib import Path
from types import ModuleType
from typing import Any
import urllib.parse


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOOPBACK_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
CONFIG_NAME = "pi05_pretrain_human300"
MAX_TOKEN_LEN = 200
ACTION_DIM = 32
ACTION_HORIZON = 50
IMAGE_SIZE = 224


@dataclasses.dataclass(frozen=True)
class CheckpointLayout:
    """The only checkpoint files read by the lightweight JAX policy loader."""

    root: Path
    params_dir: Path
    assets_dir: Path
    norm_stats_file: Path


@dataclasses.dataclass(frozen=True)
class OpenPIComponents:
    """Lazily imported upstream modules used by inference."""

    jnp: ModuleType
    model: ModuleType
    normalize: ModuleType
    pi0_config: ModuleType
    policy: ModuleType
    robocasa_policy: ModuleType
    tokenizer: ModuleType
    transforms: ModuleType
    websocket_policy_server: ModuleType


@dataclasses.dataclass(frozen=True)
class PolicyTransforms:
    inputs: Sequence[Any]
    outputs: Sequence[Any]


TokenizerFactory = Callable[[int], Any]


def load_openpi_components(
    importer: Callable[[str], ModuleType] = importlib.import_module,
) -> OpenPIComponents:
    """Import only the fixed fork's model-serving modules.

    Keeping these imports inside a function lets ``--help`` and module import
    work without importing JAX, PyTorch, OpenPI, or simulator dependencies.
    """

    return OpenPIComponents(
        jnp=importer("jax.numpy"),
        model=importer("openpi.models.model"),
        normalize=importer("openpi.shared.normalize"),
        pi0_config=importer("openpi.models.pi0_config"),
        policy=importer("openpi.policies.policy"),
        robocasa_policy=importer("openpi.policies.robocasa_policy"),
        tokenizer=importer("openpi.models.tokenizer"),
        transforms=importer("openpi.transforms"),
        websocket_policy_server=importer("openpi.serving.websocket_policy_server"),
    )


def resolve_checkpoint_layout(checkpoint: str | Path) -> CheckpointLayout:
    """Validate the local JAX checkpoint layout without any remote fallback."""

    raw = str(checkpoint)
    if urllib.parse.urlparse(raw).scheme:
        raise ValueError("Checkpoint must be an explicit local path")

    root = Path(checkpoint).expanduser().resolve()
    layout = CheckpointLayout(
        root=root,
        params_dir=root / "params",
        assets_dir=root / "assets",
        norm_stats_file=root / "assets" / "norm_stats.json",
    )
    if not layout.root.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {layout.root}")
    if not layout.params_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint params directory not found: {layout.params_dir}")
    if not layout.norm_stats_file.is_file():
        raise FileNotFoundError(f"Checkpoint norm stats not found: {layout.norm_stats_file}")
    return layout


def make_pi05_model_config(components: OpenPIComponents) -> Any:
    """Construct the exact model portion of ``pi05_pretrain_human300``."""

    config = components.pi0_config.Pi0Config(pi05=True, max_token_len=MAX_TOKEN_LEN)
    actual = (config.action_dim, config.action_horizon, config.max_token_len)
    expected = (ACTION_DIM, ACTION_HORIZON, MAX_TOKEN_LEN)
    if actual != expected or not config.discrete_state_input:
        raise RuntimeError(f"Fixed pi0.5 model defaults changed: expected {expected}, got {actual}")
    return config


def load_checkpoint_payload(
    layout: CheckpointLayout,
    components: OpenPIComponents,
) -> tuple[Any, dict[str, Any]]:
    """Load JAX params and checkpoint normalization stats from fixed paths."""

    params = components.model.restore_params(layout.params_dir, dtype=components.jnp.bfloat16)
    # normalize.load(directory) reads exactly directory / "norm_stats.json".
    norm_stats = components.normalize.load(layout.assets_dir)
    return params, norm_stats


def build_policy_transforms(
    *,
    components: OpenPIComponents,
    model_config: Any,
    norm_stats: dict[str, Any],
    tokenizer: Any,
    default_prompt: str | None = None,
) -> PolicyTransforms:
    """Build the fixed fork's RoboCasa pi0.5 inference transform sequence."""

    transforms = components.transforms
    robocasa_policy = components.robocasa_policy
    return PolicyTransforms(
        inputs=(
            # policy_config.create_trained_policy applies this before data transforms.
            transforms.InjectDefaultPrompt(default_prompt),
            robocasa_policy.RobocasaInputs(
                action_dim=model_config.action_dim,
                model_type=model_config.model_type,
            ),
            transforms.Normalize(norm_stats, use_quantiles=False),
            # ModelTransformFactory for PI05 at the fixed commit.
            transforms.InjectDefaultPrompt(None),
            transforms.ResizeImages(IMAGE_SIZE, IMAGE_SIZE),
            transforms.TokenizePrompt(
                tokenizer,
                discrete_state_input=model_config.discrete_state_input,
            ),
            transforms.PadStatesAndActions(model_config.action_dim),
        ),
        outputs=(
            transforms.Unnormalize(norm_stats, use_quantiles=False),
            robocasa_policy.RobocasaOutputs(),
        ),
    )


def build_policy(
    checkpoint: str | Path,
    *,
    default_prompt: str | None = None,
    tokenizer_factory: TokenizerFactory | None = None,
    components: OpenPIComponents | None = None,
) -> Any:
    """Restore the fixed JAX checkpoint and construct an inference-only policy."""

    components = components or load_openpi_components()
    layout = resolve_checkpoint_layout(checkpoint)
    model_config = make_pi05_model_config(components)
    params, norm_stats = load_checkpoint_payload(layout, components)
    model = model_config.load(params)

    factory = tokenizer_factory or components.tokenizer.PaligemmaTokenizer
    tokenizer = factory(model_config.max_token_len)
    pipeline = build_policy_transforms(
        components=components,
        model_config=model_config,
        norm_stats=norm_stats,
        tokenizer=tokenizer,
        default_prompt=default_prompt,
    )
    return components.policy.Policy(
        model,
        transforms=pipeline.inputs,
        output_transforms=pipeline.outputs,
        metadata=None,
        is_pytorch=False,
    )


def validate_loopback_host(host: str) -> str:
    if host != LOOPBACK_HOST:
        raise ValueError(f"Policy server must bind exactly {LOOPBACK_HOST}, got {host!r}")
    return host


def validate_port(port: int) -> int:
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
        raise ValueError("port must be an integer between 1 and 65535")
    return port


def create_websocket_server(
    policy: Any,
    *,
    host: str,
    port: int,
    server_factory: Callable[..., Any],
) -> Any:
    """Construct the upstream server only after enforcing local-only binding."""

    host = validate_loopback_host(host)
    port = validate_port(port)
    return server_factory(policy=policy, host=host, port=port, metadata=policy.metadata)


def configure_project_caches() -> None:
    """Keep runtime caches inside this project for the service process."""

    cache = PROJECT_ROOT / "cache"
    values = {
        "OPENPI_DATA_HOME": cache / "openpi",
        "JAX_COMPILATION_CACHE_DIR": cache / "jax",
        "XDG_CACHE_HOME": cache / "xdg",
        "HF_HOME": cache / "huggingface",
        "TORCH_HOME": cache / "torch",
        "TMPDIR": cache / "tmp",
    }
    for name, path in values.items():
        os.environ[name] = str(path)


def _loopback_arg(value: str) -> str:
    try:
        return validate_loopback_host(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--host", default=LOOPBACK_HOST, type=_loopback_arg)
    parser.add_argument("--port", default=DEFAULT_PORT, type=int)
    parser.add_argument("--default-prompt")
    args = parser.parse_args(argv)
    try:
        validate_port(args.port)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def serve(args: argparse.Namespace) -> None:
    configure_project_caches()
    components = load_openpi_components()
    policy = build_policy(
        args.checkpoint,
        default_prompt=args.default_prompt,
        components=components,
    )
    server = create_websocket_server(
        policy,
        host=args.host,
        port=args.port,
        server_factory=components.websocket_policy_server.WebsocketPolicyServer,
    )
    logging.info("Serving %s on ws://%s:%d", CONFIG_NAME, args.host, args.port)
    server.serve_forever()


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO)
    serve(parse_args(argv))


if __name__ == "__main__":
    main()
