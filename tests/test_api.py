from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from dice_detector.api import create_app
from dice_detector.config import Settings
from dice_detector.types import Box, Detection, DetectionResult


class FakeDetector:
    def detect(self, image: Image.Image) -> DetectionResult:
        return DetectionResult(
            has_dice=True,
            count=1,
            detections=(Detection("d20", 0.91, Box(1, 2, 10, 12)),),
            image_width=image.width,
            image_height=image.height,
            inference_ms=4.2,
        )


def png_bytes(width: int = 16, height: int = 8) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (width, height), "blue").save(buffer, format="PNG")
    return buffer.getvalue()


def test_health_ready_and_detect() -> None:
    app = create_app(detector=FakeDetector())

    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/ready").json() == {"ready": True}
        response = client.post(
            "/v1/detect",
            files={"file": ("dice.png", png_bytes(), "image/png")},
        )

    assert response.status_code == 200
    assert response.json()["has_dice"] is True
    assert response.json()["detections"][0]["label"] == "d20"


def test_detect_rejects_oversized_upload() -> None:
    settings = Settings(max_upload_bytes=8)
    app = create_app(settings=settings, detector=FakeDetector())

    with TestClient(app) as client:
        response = client.post(
            "/v1/detect",
            files={"file": ("dice.png", b"123456789", "image/png")},
        )

    assert response.status_code == 413


def test_detect_downscales_large_image_before_inference() -> None:
    settings = Settings(max_source_pixels=1_000, max_image_pixels=100)
    app = create_app(settings=settings, detector=FakeDetector())

    with TestClient(app) as client:
        response = client.post(
            "/v1/detect",
            files={"file": ("dice.png", png_bytes(20, 20), "image/png")},
        )

    assert response.status_code == 200
    assert response.json()["image_width"] == 10
    assert response.json()["image_height"] == 10


def test_ready_reports_missing_model() -> None:
    app = create_app(model_loader=lambda: (_ for _ in ()).throw(FileNotFoundError("missing")))

    with TestClient(app) as client:
        response = client.get("/ready")

    assert response.status_code == 503
    assert response.json() == {"ready": False, "detail": "missing"}
