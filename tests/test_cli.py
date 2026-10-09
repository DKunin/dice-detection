import json
import sys
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from dice_detector.cli import main
from dice_detector.types import Box, Detection, DetectionResult


class FakeDetector:
    def detect(self, image: Image.Image) -> DetectionResult:
        return DetectionResult(
            has_dice=True,
            count=1,
            detections=(Detection("d20", 0.9, Box(1, 2, 3, 4)),),
            image_width=image.width,
            image_height=image.height,
            inference_ms=1.5,
        )


def test_download_model_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    model_path = tmp_path / "model.onnx"
    monkeypatch.setattr(
        "dice_detector.cli.download_artifacts",
        lambda directory, accept_license: (model_path,),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["dice-detector", "download-model", "--model-dir", str(tmp_path), "--accept-license"],
    )

    main()

    assert capsys.readouterr().out.strip() == str(model_path)


def test_detect_command_outputs_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    image_path = tmp_path / "image.png"
    Image.new("RGB", (10, 8), "red").save(image_path)
    monkeypatch.setattr(
        "dice_detector.cli.DiceDetector.from_files",
        lambda *args, **kwargs: FakeDetector(),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["dice-detector", "detect", str(image_path), "--model-dir", str(tmp_path)],
    )

    main()

    output = json.loads(capsys.readouterr().out)
    assert output["has_dice"] is True
    assert output["detections"][0]["label"] == "d20"


def test_serve_command_starts_uvicorn(monkeypatch: pytest.MonkeyPatch) -> None:
    received: dict[str, Any] = {}

    def run(app: object, *, host: str, port: int) -> None:
        received.update(app=app, host=host, port=port)

    monkeypatch.setattr("uvicorn.run", run)
    monkeypatch.setattr(
        sys,
        "argv",
        ["dice-detector", "serve", "--host", "0.0.0.0", "--port", "9090"],
    )

    main()

    assert received["host"] == "0.0.0.0"
    assert received["port"] == 9090


def test_serve_command_uses_server_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    received: dict[str, Any] = {}

    def create_app(*, settings: object) -> object:
        received["settings"] = settings
        return object()

    def run(app: object, *, host: str, port: int) -> None:
        received.update(app=app, host=host, port=port)

    monkeypatch.setattr("dice_detector.cli.create_app", create_app)
    monkeypatch.setattr("uvicorn.run", run)
    monkeypatch.setattr(sys, "argv", ["dice-detector", "serve"])

    main()

    assert received["host"] == "127.0.0.1"
    assert received["port"] == 18080
    assert received["settings"].confidence_threshold == 0.85
