from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from roboflow_dice_test.client import (
    RoboflowConfig,
    RoboflowResponseError,
    RoboflowTestError,
    create_client,
    parse_response,
    run_inference,
)


def test_config_reads_environment_and_keeps_key_out_of_repr() -> None:
    config = RoboflowConfig.from_env(
        {
            "ROBOFLOW_API_KEY": " secret ",
            "ROBOFLOW_API_URL": "https://example.test",
            "ROBOFLOW_MODEL_ID": "workspace/model/7",
            "ROBOFLOW_CONFIDENCE": "0.61",
        }
    )

    assert config.api_key == "secret"
    assert config.api_url == "https://example.test"
    assert config.model_id == "workspace/model/7"
    assert config.confidence == pytest.approx(0.61)
    assert "secret" not in repr(config)


@pytest.mark.parametrize(
    ("environment", "message"),
    [
        ({}, "ROBOFLOW_API_KEY"),
        ({"ROBOFLOW_API_KEY": "key", "ROBOFLOW_CONFIDENCE": "not-a-number"}, "confidence"),
        ({"ROBOFLOW_API_KEY": "key", "ROBOFLOW_CONFIDENCE": "1.1"}, "between"),
        (
            {"ROBOFLOW_API_KEY": "key", "ROBOFLOW_API_URL": "http://example.test"},
            "HTTPS",
        ),
    ],
)
def test_config_rejects_invalid_environment(environment: dict[str, str], message: str) -> None:
    with pytest.raises(RoboflowTestError, match=message):
        RoboflowConfig.from_env(environment)


def test_parse_response_returns_binary_result_and_converts_boxes() -> None:
    result = parse_response(
        {
            "time": 0.123,
            "image": {"width": 200, "height": 100},
            "predictions": [
                {
                    "class": "d20",
                    "confidence": 0.92,
                    "x": 50,
                    "y": 40,
                    "width": 30,
                    "height": 20,
                },
                {
                    "class": "d6",
                    "confidence": 0.73,
                    "x": 195,
                    "y": 95,
                    "width": 30,
                    "height": 20,
                },
            ],
        },
        model_id="dnd-dices-ycaiz/2",
        confidence=0.7,
    )

    assert result == {
        "has_dice": True,
        "count": 2,
        "model_id": "dnd-dices-ycaiz/2",
        "image": {"width": 200, "height": 100},
        "inference_time_seconds": 0.123,
        "detections": [
            {
                "label": "d20",
                "confidence": 0.92,
                "box": {"x_min": 35.0, "y_min": 30.0, "x_max": 65.0, "y_max": 50.0},
            },
            {
                "label": "d6",
                "confidence": 0.73,
                "box": {"x_min": 180.0, "y_min": 85.0, "x_max": 200.0, "y_max": 100.0},
            },
        ],
    }


def test_parse_response_filters_low_confidence_and_unknown_classes() -> None:
    result = parse_response(
        {
            "image": {"width": 80, "height": 60},
            "predictions": [
                {"class": "d4", "confidence": 0.49, "x": 20, "y": 20, "width": 10, "height": 10},
                {"class": "coin", "confidence": 0.99, "x": 20, "y": 20, "width": 10, "height": 10},
            ],
        },
        model_id="model/1",
        confidence=0.5,
    )

    assert result["has_dice"] is False
    assert result["count"] == 0
    assert result["detections"] == []
    assert result["inference_time_seconds"] is None


@pytest.mark.parametrize(
    "response",
    [
        [],
        {},
        {"predictions": "not-a-list"},
        {"predictions": [{"class": "d20"}]},
        {
            "predictions": [
                {
                    "class": "d20",
                    "confidence": True,
                    "x": 1,
                    "y": 1,
                    "width": 1,
                    "height": 1,
                }
            ]
        },
    ],
)
def test_parse_response_rejects_malformed_payload(response: Any) -> None:
    with pytest.raises(RoboflowResponseError):
        parse_response(response, model_id="model/1", confidence=0.5)


def test_run_inference_passes_model_and_wraps_transport_errors(tmp_path: Path) -> None:
    image = tmp_path / "dice.jpg"
    image.write_bytes(b"image")
    config = RoboflowConfig(api_key="key", api_url="https://example.test", model_id="model/2")
    calls: list[tuple[str, str]] = []

    class FakeClient:
        def infer(self, image_path: str, *, model_id: str) -> object:
            calls.append((image_path, model_id))
            return {"image": {"width": 1, "height": 1}, "predictions": []}

    result = run_inference(image, config, client_factory=lambda _: FakeClient())
    assert result["has_dice"] is False
    assert calls == [(str(image), "model/2")]

    class BrokenClient:
        def infer(self, image_path: str, *, model_id: str) -> object:
            raise RuntimeError("network unavailable for key")

    with pytest.raises(RoboflowTestError, match="Roboflow request failed") as error:
        run_inference(image, config, client_factory=lambda _: BrokenClient())
    assert "key" not in str(error.value)
    assert "[redacted]" in str(error.value)


def test_run_inference_rejects_missing_file(tmp_path: Path) -> None:
    config = RoboflowConfig(api_key="key")
    with pytest.raises(RoboflowTestError, match="does not exist"):
        run_inference(tmp_path / "missing.jpg", config)


def test_create_client_uses_header_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    fake_module = ModuleType("inference_sdk")

    class FakeConfiguration:
        def __init__(self, **kwargs: Any) -> None:
            captured["configuration"] = kwargs

    class FakeHTTPClient:
        def __init__(self, **kwargs: Any) -> None:
            captured["client"] = kwargs

        def configure(self, configuration: object) -> FakeHTTPClient:
            captured["configured"] = configuration
            return self

    fake_module.InferenceConfiguration = FakeConfiguration  # type: ignore[attr-defined]
    fake_module.InferenceHTTPClient = FakeHTTPClient  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "inference_sdk", fake_module)

    config = RoboflowConfig(
        api_key="top-secret",
        api_url="https://example.test",
        confidence=0.72,
    )
    create_client(config)

    assert captured["client"] == {"api_url": "https://example.test", "api_key": "top-secret"}
    assert captured["configuration"] == {
        "api_key_transport": "header",
        "confidence_threshold": 0.72,
    }
