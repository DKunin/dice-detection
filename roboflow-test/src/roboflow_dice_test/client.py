"""Typed boundary around the Roboflow hosted-inference SDK."""

from __future__ import annotations

import math
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, cast

DEFAULT_API_URL = "https://serverless.roboflow.com"
DEFAULT_MODEL_ID = "dnd-dices-ycaiz/2"
DEFAULT_CONFIDENCE = 0.35
DICE_CLASSES = frozenset({"d4", "d6", "d8", "d10", "d12", "d20"})


class RoboflowTestError(RuntimeError):
    """Base error with a safe message suitable for the CLI."""


class RoboflowResponseError(RoboflowTestError):
    """Raised when Roboflow returns a response with an unexpected shape."""


class InferenceClient(Protocol):
    """Small SDK surface used by the application and fakes."""

    def infer(self, image_path: str, *, model_id: str) -> object: ...


ClientFactory = Callable[["RoboflowConfig"], InferenceClient]


@dataclass(frozen=True, slots=True)
class RoboflowConfig:
    """Configuration loaded from explicit environment variables."""

    api_key: str = field(repr=False)
    api_url: str = DEFAULT_API_URL
    model_id: str = DEFAULT_MODEL_ID
    confidence: float = DEFAULT_CONFIDENCE

    def __post_init__(self) -> None:
        if not self.api_key.strip():
            raise RoboflowTestError("ROBOFLOW_API_KEY is required")
        if not self.api_url.strip():
            raise RoboflowTestError("ROBOFLOW_API_URL must not be empty")
        if not self.api_url.startswith("https://"):
            raise RoboflowTestError("ROBOFLOW_API_URL must use HTTPS")
        if not self.model_id.strip():
            raise RoboflowTestError("ROBOFLOW_MODEL_ID must not be empty")
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise RoboflowTestError("confidence must be between 0 and 1")

    @classmethod
    def from_env(cls, environment: Mapping[str, str] | None = None) -> RoboflowConfig:
        """Build configuration without ever including the API key in an error."""
        source = os.environ if environment is None else environment
        raw_confidence = source.get("ROBOFLOW_CONFIDENCE", str(DEFAULT_CONFIDENCE))
        try:
            confidence = float(raw_confidence)
        except ValueError as exc:
            raise RoboflowTestError("confidence must be a number") from exc

        return cls(
            api_key=source.get("ROBOFLOW_API_KEY", "").strip(),
            api_url=source.get("ROBOFLOW_API_URL", DEFAULT_API_URL).strip().rstrip("/"),
            model_id=source.get("ROBOFLOW_MODEL_ID", DEFAULT_MODEL_ID).strip(),
            confidence=confidence,
        )


def create_client(config: RoboflowConfig) -> InferenceClient:
    """Create the real SDK client with credential transport restricted to a header."""
    from inference_sdk import InferenceConfiguration, InferenceHTTPClient

    client = InferenceHTTPClient(api_url=config.api_url, api_key=config.api_key)
    configured = client.configure(
        InferenceConfiguration(
            api_key_transport="header",
            confidence_threshold=config.confidence,
        )
    )
    return cast(InferenceClient, configured)


def run_inference(
    image_path: Path,
    config: RoboflowConfig,
    *,
    client_factory: ClientFactory | None = None,
) -> dict[str, object]:
    """Run hosted inference and normalize its response into the local binary contract."""
    if not image_path.is_file():
        raise RoboflowTestError(f"image does not exist or is not a file: {image_path}")

    factory = create_client if client_factory is None else client_factory
    try:
        response = factory(config).infer(str(image_path), model_id=config.model_id)
    except RoboflowTestError:
        raise
    except Exception as exc:
        detail = str(exc).replace(config.api_key, "[redacted]")
        raise RoboflowTestError(f"Roboflow request failed: {detail}") from exc

    return parse_response(response, model_id=config.model_id, confidence=config.confidence)


def parse_response(
    response: object,
    *,
    model_id: str,
    confidence: float,
) -> dict[str, object]:
    """Validate and normalize an object-detection response from Roboflow."""
    payload = _mapping(response, "response")
    if "predictions" not in payload:
        raise RoboflowResponseError("response.predictions is missing")
    predictions = payload["predictions"]
    if not isinstance(predictions, Sequence) or isinstance(predictions, (str, bytes)):
        raise RoboflowResponseError("response.predictions must be a list")

    image = _mapping(payload.get("image"), "response.image")
    image_width = _positive_number(image.get("width"), "response.image.width")
    image_height = _positive_number(image.get("height"), "response.image.height")

    detections: list[dict[str, object]] = []
    for index, raw_prediction in enumerate(predictions):
        prediction = _mapping(raw_prediction, f"response.predictions[{index}]")
        label = prediction.get("class")
        if not isinstance(label, str):
            raise RoboflowResponseError(f"response.predictions[{index}].class must be a string")

        score = _number(prediction.get("confidence"), f"response.predictions[{index}].confidence")
        x = _number(prediction.get("x"), f"response.predictions[{index}].x")
        y = _number(prediction.get("y"), f"response.predictions[{index}].y")
        width = _positive_number(prediction.get("width"), f"response.predictions[{index}].width")
        height = _positive_number(prediction.get("height"), f"response.predictions[{index}].height")

        if label not in DICE_CLASSES or score < confidence:
            continue

        detections.append(
            {
                "label": label,
                "confidence": score,
                "box": {
                    "x_min": max(0.0, x - width / 2),
                    "y_min": max(0.0, y - height / 2),
                    "x_max": min(image_width, x + width / 2),
                    "y_max": min(image_height, y + height / 2),
                },
            }
        )

    detections.sort(key=lambda detection: cast(float, detection["confidence"]), reverse=True)
    inference_time = payload.get("time")
    normalized_time = None if inference_time is None else _number(inference_time, "response.time")

    return {
        "has_dice": bool(detections),
        "count": len(detections),
        "model_id": model_id,
        "image": {"width": int(image_width), "height": int(image_height)},
        "inference_time_seconds": normalized_time,
        "detections": detections,
    }


def _mapping(value: object, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise RoboflowResponseError(f"{field_name} must be an object")
    return cast(Mapping[str, object], value)


def _number(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RoboflowResponseError(f"{field_name} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise RoboflowResponseError(f"{field_name} must be a finite number")
    return number


def _positive_number(value: object, field_name: str) -> float:
    number = _number(value, field_name)
    if number <= 0:
        raise RoboflowResponseError(f"{field_name} must be greater than zero")
    return number
