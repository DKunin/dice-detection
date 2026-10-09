import pytest
from PIL import Image

from dice_detector.preprocess import LetterboxTransform, letterbox


def test_letterbox_centers_and_normalizes_image() -> None:
    image = Image.new("RGB", (200, 100), (255, 0, 0))

    tensor, transform = letterbox(image, input_size=640)

    assert tensor.shape == (1, 3, 640, 640)
    assert tensor.dtype.name == "float32"
    assert transform == LetterboxTransform(
        original_width=200,
        original_height=100,
        input_size=640,
        scale=3.2,
        pad_x=0,
        pad_y=160,
    )
    assert tensor[0, :, 0, 0] == pytest.approx([114 / 255, 114 / 255, 114 / 255])
    assert tensor[0, :, 320, 320] == pytest.approx([1.0, 0.0, 0.0])


def test_letterbox_rejects_invalid_size() -> None:
    with pytest.raises(ValueError, match="positive"):
        letterbox(Image.new("RGB", (10, 10)), input_size=0)
