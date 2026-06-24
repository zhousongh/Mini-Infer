from __future__ import annotations

import argparse
from pathlib import Path

import quick_chat
from mini_infer.clients import chat_client


def test_quick_chat_resolve_real_model_path_prefers_explicit_path(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    args = argparse.Namespace(real=True, model_path=str(model_dir), device="cuda:0")
    assert quick_chat.resolve_real_model_path(args) == str(model_dir)


def test_chat_client_parse_args_supports_quick_modes() -> None:
    args = chat_client.parse_args(["--quick-dry-run", "--base-url", "http://127.0.0.1:9000/v1"])
    assert args.quick_dry_run is True
    assert args.quick_model_path == ""
    assert args.base_url == "http://127.0.0.1:9000/v1"


def test_quick_chat_parse_args_supports_device() -> None:
    args, rest = quick_chat.parse_args(["--real", "--device", "cuda:1"])
    assert args.real is True
    assert args.device == "cuda:1"
    assert rest == []
