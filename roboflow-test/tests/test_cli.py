from __future__ import annotations

import json
from pathlib import Path

import pytest

from roboflow_dice_test import cli
from roboflow_dice_test.client import RoboflowConfig, RoboflowTestError


def test_cli_prints_normalized_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    image = tmp_path / "dice.jpg"
    image.write_bytes(b"image")
    captured: dict[str, object] = {}

    def fake_run(path: Path, config: RoboflowConfig) -> dict[str, object]:
        captured.update(path=path, config=config)
        return {"has_dice": True, "count": 1, "detections": []}

    monkeypatch.setattr(cli, "run_inference", fake_run)
    monkeypatch.setenv("ROBOFLOW_API_KEY", "key")

    exit_code = cli.main([str(image), "--confidence", "0.8"])

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out)["has_dice"] is True
    assert captured["path"] == image
    config = captured["config"]
    assert isinstance(config, RoboflowConfig)
    assert config.confidence == pytest.approx(0.8)


def test_cli_returns_error_without_leaking_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    image = tmp_path / "dice.jpg"
    image.write_bytes(b"image")
    monkeypatch.setenv("ROBOFLOW_API_KEY", "top-secret")
    monkeypatch.setattr(
        cli,
        "run_inference",
        lambda *_: (_ for _ in ()).throw(RoboflowTestError("request failed")),
    )

    assert cli.main([str(image)]) == 2
    error = capsys.readouterr().err
    assert "request failed" in error
    assert "top-secret" not in error
