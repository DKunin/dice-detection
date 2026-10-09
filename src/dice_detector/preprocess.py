from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from PIL import Image


@dataclass(frozen=True, slots=True)
class LetterboxTransform:
    """Information required to map model boxes back to the source image."""

    original_width: int
    original_height: int
    input_size: int
    scale: float
    pad_x: int
    pad_y: int


def letterbox(
    image: Image.Image,
    *,
    input_size: int = 640,
    padding_value: int = 114,
) -> tuple[NDArray[np.float32], LetterboxTransform]:
    """Resize an image without distortion and return an NCHW float tensor."""
    if input_size <= 0:
        raise ValueError("input_size must be positive")
    if image.width <= 0 or image.height <= 0:
        raise ValueError("image dimensions must be positive")
    if not 0 <= padding_value <= 255:
        raise ValueError("padding_value must be between 0 and 255")

    rgb_image = image.convert("RGB")
    scale = min(input_size / rgb_image.width, input_size / rgb_image.height)
    resized_width = max(1, round(rgb_image.width * scale))
    resized_height = max(1, round(rgb_image.height * scale))
    resized = rgb_image.resize(
        (resized_width, resized_height),
        resample=Image.Resampling.BILINEAR,
    )

    pad_x = (input_size - resized_width) // 2
    pad_y = (input_size - resized_height) // 2
    canvas = Image.new(
        "RGB",
        (input_size, input_size),
        (padding_value, padding_value, padding_value),
    )
    canvas.paste(resized, (pad_x, pad_y))

    array = np.asarray(canvas, dtype=np.float32) / np.float32(255.0)
    tensor = np.ascontiguousarray(array.transpose(2, 0, 1)[None, ...])
    transform = LetterboxTransform(
        original_width=rgb_image.width,
        original_height=rgb_image.height,
        input_size=input_size,
        scale=scale,
        pad_x=pad_x,
        pad_y=pad_y,
    )
    return tensor, transform
