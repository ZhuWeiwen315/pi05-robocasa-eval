"""Project-owned kitchen asset layout and pre-import RoboCasa redirection.

RoboCasa 1.0.1 has no kitchen asset root setting. Its object catalog is built
during import, so the root must change as ``robocasa.models`` is first loaded.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.abc
import importlib.machinery
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ASSET_ROOT = PROJECT_ROOT / "datasets" / "robocasa-kitchen-assets"
BOX_LINKS_PATH = (
    PROJECT_ROOT
    / "sources/robocasa/robocasa/models/assets/box_links/box_links_assets.json"
)

# Keys and folder arguments from RoboCasa 1.0.1 download_kitchen_assets.py.
ASSET_TYPES = (
    ("tex", "textures", "textures"),
    ("tex_generative", "generative_textures", "generative_textures"),
    ("fixtures_lw", "fixtures_lightwheel", "fixtures"),
    ("objs_objaverse", "objaverse", "objects/objaverse"),
    ("objs_aigen", "aigen_objs", "objects/aigen_objs"),
    ("objs_lw", "objects_lightwheel", "objects/lightwheel"),
)

BUNDLED_SENTINELS = (
    "arenas/empty_kitchen_arena.xml",
    "fixtures/fixture_registry/cabinet.yaml",
)


@dataclass(frozen=True)
class AssetSpec:
    name: str
    shared_url: str
    direct_url: str
    subdir: str


def direct_box_url(shared_url: str) -> str:
    """Match the fixed upstream Box shared/static URL conversion."""
    prefix = "https://utexas.box.com/s/"
    if not shared_url.startswith(prefix):
        raise ValueError("Expected an official UT Austin Box shared URL")
    shared_id = shared_url[len(prefix) :].strip("/")
    if not shared_id or "/" in shared_id or "?" in shared_id or "#" in shared_id:
        raise ValueError("Invalid Box shared ID")
    return f"https://utexas.box.com/shared/static/{shared_id}.zip"


def asset_specs(manifest: Path = BOX_LINKS_PATH) -> tuple[AssetSpec, ...]:
    links = json.loads(manifest.read_text(encoding="utf-8"))
    return tuple(
        AssetSpec(name, links[key], direct_box_url(links[key]), subdir)
        for name, key, subdir in ASSET_TYPES
    )


def missing_bundled_assets(root: Path) -> tuple[str, ...]:
    """Detect whether the package's small bundled base has been copied to root."""
    return tuple(path for path in BUNDLED_SENTINELS if not (root / path).is_file())


class _RootLoader(importlib.abc.Loader):
    def __init__(self, wrapped: importlib.abc.Loader, root: Path, module_name: str):
        self.wrapped = wrapped
        self.root = root
        self.module_name = module_name

    def create_module(self, spec):
        create = getattr(self.wrapped, "create_module", None)
        return create(spec) if create is not None else None

    def exec_module(self, module):
        self.wrapped.exec_module(module)
        if self.module_name == "robocasa.utils.texture_swap":
            module.TEXTURES_DIR = self.root / "generative_textures"
        else:
            module.assets_root = str(self.root)


class AssetRootFinder(importlib.abc.MetaPathFinder):
    """Replace the root before RoboCasa imports and caches object paths."""

    def __init__(self, root: Path, module_name: str = "robocasa.models"):
        self.root = root
        self.module_name = module_name

    def find_spec(self, fullname, path=None, target=None):
        if fullname != self.module_name:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is None or spec.loader is None:
            return spec
        spec.loader = _RootLoader(spec.loader, self.root, self.module_name)
        return spec


def activate_project_assets() -> None:
    """Activate the fixed project asset root before any RoboCasa import."""
    if ASSET_ROOT.resolve() != ASSET_ROOT:
        raise RuntimeError(f"Asset root must not resolve through a symlink: {ASSET_ROOT}")
    missing = missing_bundled_assets(ASSET_ROOT)
    if missing:
        raise FileNotFoundError(
            f"Project asset root {ASSET_ROOT} lacks bundled files: {', '.join(missing)}"
        )
    if "robocasa" in sys.modules or "robocasa.models" in sys.modules:
        raise RuntimeError("Asset root must be activated before importing RoboCasa")
    sys.meta_path.insert(0, AssetRootFinder(ASSET_ROOT, "robocasa.utils.texture_swap"))
    sys.meta_path.insert(0, AssetRootFinder(ASSET_ROOT))
