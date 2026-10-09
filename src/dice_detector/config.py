import os
from dataclasses import dataclass, field
from pathlib import Path


def _default_threads() -> int:
    return max(1, min(os.cpu_count() or 1, 6))


def _environment_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean flag")


@dataclass(frozen=True, slots=True)
class Settings:
    model_dir: Path = Path("models")
    confidence_threshold: float = 0.85
    iou_threshold: float = 0.45
    max_upload_bytes: int = 10 * 1024 * 1024
    max_source_pixels: int = 150_000_000
    max_image_pixels: int = 4_000_000
    high_confidence_fallback: bool = True
    onnx_threads: int = field(default_factory=_default_threads)

    @property
    def model_path(self) -> Path:
        return self.model_dir / "dice-shape.onnx"

    @property
    def classes_path(self) -> Path:
        return self.model_dir / "dice-shape.classes.json"

    @classmethod
    def from_env(cls) -> "Settings":
        settings = cls(
            model_dir=Path(os.getenv("DICE_DETECTOR_MODEL_DIR", "models")),
            confidence_threshold=float(os.getenv("DICE_DETECTOR_CONFIDENCE", "0.85")),
            iou_threshold=float(os.getenv("DICE_DETECTOR_IOU", "0.45")),
            max_upload_bytes=int(
                os.getenv("DICE_DETECTOR_MAX_UPLOAD_BYTES", str(10 * 1024 * 1024))
            ),
            max_source_pixels=int(
                os.getenv("DICE_DETECTOR_MAX_SOURCE_PIXELS", "150000000")
            ),
            max_image_pixels=int(os.getenv("DICE_DETECTOR_MAX_IMAGE_PIXELS", "4000000")),
            high_confidence_fallback=_environment_flag(
                "DICE_DETECTOR_HIGH_CONFIDENCE_FALLBACK",
                True,
            ),
            onnx_threads=int(os.getenv("DICE_DETECTOR_ONNX_THREADS", str(_default_threads()))),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        if not 0 <= self.confidence_threshold <= 1:
            raise ValueError("confidence threshold must be between 0 and 1")
        if not 0 <= self.iou_threshold <= 1:
            raise ValueError("IoU threshold must be between 0 and 1")
        if self.max_upload_bytes <= 0:
            raise ValueError("max upload bytes must be positive")
        if self.max_source_pixels <= 0:
            raise ValueError("max source pixels must be positive")
        if self.max_image_pixels <= 0:
            raise ValueError("max image pixels must be positive")
        if self.max_image_pixels > self.max_source_pixels:
            raise ValueError("max image pixels cannot exceed max source pixels")
        if self.onnx_threads < 0:
            raise ValueError("ONNX thread count cannot be negative")
