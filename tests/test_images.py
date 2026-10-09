from io import BytesIO

import pytest
from PIL import Image

from dice_detector.images import InvalidImageError, decode_image, source_image_size


def png_bytes(width: int = 16, height: int = 8) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (width, height), "blue").save(buffer, format="PNG")
    return buffer.getvalue()


def jpeg_bytes(width: int = 16, height: int = 8) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (width, height), "blue").save(buffer, format="JPEG")
    return buffer.getvalue()


def test_decode_image_returns_exif_normalized_rgb() -> None:
    image = decode_image(png_bytes())

    assert image.mode == "RGB"
    assert image.size == (16, 8)


def test_decode_image_rejects_invalid_content() -> None:
    with pytest.raises(InvalidImageError, match="valid image"):
        decode_image(b"not an image")


def test_decode_image_downscales_for_inference() -> None:
    image = decode_image(png_bytes(20, 20), max_pixels=100)

    assert image.size == (10, 10)
    assert source_image_size(image) == (20, 20)


def test_decode_image_uses_decoder_downscaling_before_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("dice_detector.images._MAX_SAFE_DECODE_PIXELS", 100)

    image = decode_image(jpeg_bytes(40, 40), max_pixels=25, max_source_pixels=2_000)

    assert image.width * image.height <= 25


def test_decode_image_rejects_source_above_safety_limit() -> None:
    with pytest.raises(InvalidImageError, match="safety limit"):
        decode_image(png_bytes(20, 20), max_pixels=100, max_source_pixels=300)
