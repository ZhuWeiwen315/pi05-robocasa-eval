"""Offline checks for Box inventory and import-time asset path redirection."""

import importlib
import importlib.machinery
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from robocasa_eval.assets import (
    ASSET_TYPES,
    AssetRootFinder,
    PROJECT_ROOT,
    asset_specs,
    direct_box_url,
    missing_bundled_assets,
)
from scripts.asset_inventory import (
    _NoRedirect, _RangeRedirect, archive_size, head_size, parse_content_range,
    range_probe,
)


class AssetTests(unittest.TestCase):
    def test_six_official_types_and_extract_subdirs(self) -> None:
        links = {key: f"https://utexas.box.com/s/id{i}" for i, (_, key, _) in enumerate(ASSET_TYPES)}
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "cache") as directory:
            manifest = Path(directory) / "links.json"
            manifest.write_text(json.dumps(links), encoding="utf-8")
            specs = asset_specs(manifest)
        self.assertEqual(len(specs), 6)
        self.assertEqual(specs[0].subdir, "textures")
        self.assertEqual(specs[-1].subdir, "objects/lightwheel")
        self.assertEqual(specs[0].direct_url, "https://utexas.box.com/shared/static/id0.zip")

    def test_invalid_box_url_rejected(self) -> None:
        with self.assertRaises(ValueError):
            direct_box_url("https://example.com/s/id")

    def test_missing_bundled_base_is_detected(self) -> None:
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "cache") as directory:
            root = Path(directory)
            self.assertEqual(len(missing_bundled_assets(root)), 2)
            for relative in ("arenas/empty_kitchen_arena.xml", "fixtures/fixture_registry/cabinet.yaml"):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            self.assertEqual(missing_bundled_assets(root), ())

    def test_loader_sets_root_before_submodule_imports(self) -> None:
        class OriginalLoader:
            def exec_module(self, module):
                module.assets_root = "upstream"

        spec = importlib.machinery.ModuleSpec("fake.models", OriginalLoader())
        finder = AssetRootFinder(Path("/project/assets"), "fake.models")
        with patch.object(importlib.machinery.PathFinder, "find_spec", return_value=spec):
            wrapped = finder.find_spec("fake.models", ["/fake"])
        module = types.ModuleType("fake.models")
        wrapped.loader.exec_module(module)
        self.assertEqual(module.assets_root, "/project/assets")
        self.assertIsNone(finder.find_spec("another.module"))

    def test_generative_texture_path_is_redirected(self) -> None:
        class OriginalLoader:
            def exec_module(self, module):
                module.TEXTURES_DIR = Path("/upstream/generative_textures")

        spec = importlib.machinery.ModuleSpec("robocasa.utils.texture_swap", OriginalLoader())
        finder = AssetRootFinder(Path("/project/assets"), "robocasa.utils.texture_swap")
        with patch.object(importlib.machinery.PathFinder, "find_spec", return_value=spec):
            wrapped = finder.find_spec("robocasa.utils.texture_swap", ["/fake"])
        module = types.ModuleType("robocasa.utils.texture_swap")
        wrapped.loader.exec_module(module)
        self.assertEqual(module.TEXTURES_DIR, Path("/project/assets/generative_textures"))

    def test_import_hook_precedes_cached_object_catalog(self) -> None:
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "cache") as directory:
            package = Path(directory) / "fake_assets_package"
            models = package / "models"
            models.mkdir(parents=True)
            (package / "__init__.py").write_text("from .models import objects\n", encoding="utf-8")
            (models / "__init__.py").write_text("assets_root = 'upstream'\n", encoding="utf-8")
            (models / "objects.py").write_text("from . import assets_root\ncached = assets_root\n", encoding="utf-8")
            finder = AssetRootFinder(Path("/project/assets"), "fake_assets_package.models")
            sys.path.insert(0, directory)
            sys.meta_path.insert(0, finder)
            try:
                imported = importlib.import_module("fake_assets_package")
                self.assertEqual(imported.models.objects.cached, "/project/assets")
            finally:
                sys.meta_path.remove(finder)
                sys.path.remove(directory)
                for name in tuple(sys.modules):
                    if name == "fake_assets_package" or name.startswith("fake_assets_package."):
                        del sys.modules[name]

    def test_head_probe_never_requests_body(self) -> None:
        response = unittest.mock.MagicMock()
        response.__enter__.return_value = response
        response.status = 200
        response.headers.get.return_value = "123"
        response.headers.get.side_effect = lambda key, default=None: {
            "Content-Length": "123", "Content-Type": "application/zip"
        }.get(key, default)
        opener = unittest.mock.MagicMock()
        opener.open.return_value = response
        with patch("scripts.asset_inventory.request.build_opener", return_value=opener) as build:
            self.assertEqual(head_size("https://utexas.box.com/shared/static/id.zip"), ("200", 123))
        self.assertIsInstance(build.call_args.args[0], _NoRedirect)
        self.assertEqual(opener.open.call_args.args[0].get_method(), "HEAD")

    def test_redirect_is_not_followed_as_get(self) -> None:
        handler = _NoRedirect()
        self.assertIsNone(handler.redirect_request(None, None, 302, "Found", {}, "https://example.com"))
        opener = unittest.mock.MagicMock()
        opener.open.side_effect = HTTPError("https://example.com", 302, "Found", {}, None)
        with patch("scripts.asset_inventory.request.build_opener", return_value=opener):
            self.assertEqual(head_size("https://utexas.box.com/shared/static/id.zip"), ("302", None))

    def test_range_probe_reads_one_byte_and_closes(self) -> None:
        response = unittest.mock.MagicMock()
        response.status = 206
        response.geturl.return_value = "https://public.boxcloud.com/d/opaque-secret/download?token=secret"
        response.headers.get.side_effect = lambda key, default="": {
            "Content-Type": "application/zip",
            "Content-Length": "1",
            "Content-Range": "bytes 0-0/12345",
        }.get(key, default)
        response.read.return_value = b"P"
        opener = unittest.mock.MagicMock()
        opener.open.return_value = response
        with patch("scripts.asset_inventory.request.build_opener", return_value=opener) as build:
            result = range_probe("https://utexas.box.com/shared/static/id.zip")
        self.assertIsInstance(build.call_args.args[0], _RangeRedirect)
        req = opener.open.call_args.args[0]
        self.assertEqual(req.get_method(), "GET")
        self.assertEqual(req.get_header("Range"), "bytes=0-0")
        response.read.assert_called_once_with(1)
        response.close.assert_called_once_with()
        self.assertEqual(result.body_bytes, 1)
        self.assertEqual(result.archive_bytes, 12345)
        self.assertNotIn("secret", result.final_url)
        self.assertEqual(result.final_url, "https://public.boxcloud.com/[redacted]/download")

    def test_redirect_keeps_range(self) -> None:
        req = __import__("urllib.request", fromlist=["Request"]).Request(
            "https://utexas.box.com/file", headers={"Range": "bytes=0-0"}, method="GET"
        )
        redirected = _RangeRedirect().redirect_request(
            req, None, 302, "Found", {}, "https://dl.boxcloud.com/file"
        )
        self.assertEqual(redirected.get_header("Range"), "bytes=0-0")
        self.assertIsNone(_RangeRedirect().redirect_request(
            req, None, 302, "Found", {}, "http://example.com/file"
        ))

    def test_html_and_unknown_are_not_archives(self) -> None:
        self.assertIsNone(archive_size(200, "text/html", 'attachment; filename="x.zip"', "123", ""))
        self.assertIsNone(archive_size(200, "application/octet-stream", "", "123", ""))
        self.assertIsNone(archive_size(404, "application/zip", "", "123", ""))
        self.assertIsNone(archive_size(206, "text/html", "", "1", "bytes 0-0/100"))

    def test_content_range_total(self) -> None:
        self.assertEqual(parse_content_range("bytes 0-0/12345"), 12345)
        for value in ("bytes 0-0/*", "bytes 1-1/12345", "bytes 0-2/12345", "garbage"):
            self.assertIsNone(parse_content_range(value))


if __name__ == "__main__":
    unittest.main()
