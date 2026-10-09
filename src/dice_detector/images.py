import warnings
from io import BytesIO
from math import sqrt

from PIL import Image, ImageOps, UnidentifiedImageError

_MAX_SAFE_DECODE_PIXELS = 40_000_000
_SOURCE_SIZE_KEY = "dice_detector_source_size"


class InvalidImageError(ValueError):
    """Raised when an upload cannot be safely decoded as an image."""


def source_image_size(image: Image.Image) -> tuple[int, int]:
    """Return source dimensions retained across inference-only downscaling."""
    value = image.info.get(_SOURCE_SIZE_KEY)
    if (
        isinstance(value, tuple)
        and len(value) == 2
        and all(isinstance(dimension, int) and dimension > 0 for dimension in value)
    ):
        return value
    return image.size


def _size_within_pixels(width: int, height: int, max_pixels: int) -> tuple[int, int]:
    if width * height <= max_pixels:
        return width, height
    scale = sqrt(max_pixels / (width * height))
    return max(1, int(width * scale)), max(1, int(height * scale))


def decode_image(
    payload: bytes,
    *,
    max_pixels: int = 4_000_000,
    max_source_pixels: int = 150_000_000,
) -> Image.Image:
    """Decode an upload, safely downscale it for inference, and normalize to RGB."""
    if not payload:
        raise InvalidImageError("upload is not a valid image")
    if max_pixels <= 0:
        raise ValueError("max_pixels must be positive")
    if max_source_pixels <= 0:
        raise ValueError("max_source_pixels must be positive")
    if max_pixels > max_source_pixels:
        raise ValueError("max_pixels cannot exceed max_source_pixels")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", Image.DecompressionBombWarning)
            with Image.open(BytesIO(payload)) as opened:
                source_width, source_height = opened.size
                source_pixels = source_width * source_height
                if source_pixels > max_source_pixels:
                    raise InvalidImageError(
                        f"source image exceeds the {max_source_pixels:,}-pixel safety limit"
                    )

                decode_size = _size_within_pixels(
                    source_width,
                    source_height,
                    min(max_pixels, _MAX_SAFE_DECODE_PIXELS),
                )
                if decode_size != opened.size:
                    opened.draft("RGB", decode_size)
                if opened.width * opened.height > _MAX_SAFE_DECODE_PIXELS:
                    raise InvalidImageError("image format cannot be safely downscaled")

                opened.load()
                normalized = ImageOps.exif_transpose(opened)
                image = normalized.convert("RGB")
                target_size = _size_within_pixels(image.width, image.height, max_pixels)
                if target_size != image.size:
                    image = image.resize(target_size, resample=Image.Resampling.LANCZOS)
                image.info[_SOURCE_SIZE_KEY] = (source_width, source_height)
                image.load()
                return image
    except InvalidImageError:
        raise
    except (Image.DecompressionBombError, OSError, UnidentifiedImageError) as error:
        raise InvalidImageError("upload is not a valid image") from error
