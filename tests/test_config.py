from pathlib import Path

import pytest

from dice_detector.config import Settings


def test_settings_use_default_confidence_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DICE_DETECTOR_CONFIDENCE", raising=False)

    assert Settings().confidence_threshold == 0.85
    assert Settings.from_env().confidence_threshold == 0.85


def test_settings_load_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DICE_DETECTOR_MODEL_DIR", "/tmp/dice-models")
    monkeypatch.setenv("DICE_DETECTOR_CONFIDENCE", "0.4")
    monkeypatch.setenv("DICE_DETECTOR_IOU", "0.5")
    monkeypatch.setenv("DICE_DETECTOR_MAX_UPLOAD_BYTES", "123")
    monkeypatch.setenv("DICE_DETECTOR_MAX_SOURCE_PIXELS", "789")
    monkeypatch.setenv("DICE_DETECTOR_MAX_IMAGE_PIXELS", "456")
    monkeypatch.setenv("DICE_DETECTOR_HIGH_CONFIDENCE_FALLBACK", "false")
    monkeypatch.setenv("DICE_DETECTOR_ONNX_THREADS", "3")

    settings = Settings.from_env()

    assert settings.model_path == Path("/tmp/dice-models/dice-shape.onnx")
    assert settings.classes_path == Path("/tmp/dice-models/dice-shape.classes.json")
    assert settings.confidence_threshold == 0.4
    assert settings.iou_threshold == 0.5
    assert settings.max_upload_bytes == 123
    assert settings.max_source_pixels == 789
    assert settings.max_image_pixels == 456
    assert settings.high_confidence_fallback is False
    assert settings.onnx_threads == 3


def test_settings_reject_invalid_environment_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DICE_DETECTOR_HIGH_CONFIDENCE_FALLBACK", "sometimes")

    with pytest.raises(ValueError, match="boolean flag"):
        Settings.from_env()


@pytest.mark.parametrize(
    "settings,message",
    [
        (Settings(confidence_threshold=1.1), "confidence"),
        (Settings(iou_threshold=-0.1), "IoU"),
        (Settings(max_upload_bytes=0), "upload"),
        (Settings(max_source_pixels=0), "source pixels"),
        (Settings(max_image_pixels=0), "pixels"),
        (Settings(max_source_pixels=100, max_image_pixels=101), "cannot exceed"),
        (Settings(onnx_threads=-1), "thread"),
    ],
)
def test_settings_validation(settings: Settings, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        settings.validate()
