from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Box:
    """Bounding box in original-image pixel coordinates."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float


@dataclass(frozen=True, slots=True)
class Detection:
    """One detected die."""

    label: str
    confidence: float
    box: Box


@dataclass(frozen=True, slots=True)
class DetectionResult:
    """Binary result plus optional per-die details."""

    has_dice: bool
    count: int
    detections: tuple[Detection, ...]
    image_width: int
    image_height: int
    inference_ms: float
