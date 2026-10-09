import numpy as np
import pytest

from dice_detector.postprocess import postprocess
from dice_detector.preprocess import LetterboxTransform

LABELS = ("d4", "d6", "d8", "d10", "d12", "d20")


def prediction(
    *,
    cx: float,
    cy: float,
    width: float,
    height: float,
    objectness: float,
    class_index: int,
    class_probability: float,
) -> list[float]:
    row = [cx, cy, width, height, objectness, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
    row[5 + class_index] = class_probability
    return row


def test_postprocess_multiplies_scores_and_suppresses_overlaps() -> None:
    output = np.array(
        [
            [
                prediction(
                    cx=320,
                    cy=320,
                    width=100,
                    height=80,
                    objectness=0.9,
                    class_index=5,
                    class_probability=0.8,
                ),
                prediction(
                    cx=322,
                    cy=319,
                    width=100,
                    height=80,
                    objectness=0.8,
                    class_index=5,
                    class_probability=0.8,
                ),
                prediction(
                    cx=100,
                    cy=100,
                    width=20,
                    height=20,
                    objectness=0.4,
                    class_index=1,
                    class_probability=0.5,
                ),
            ]
        ],
        dtype=np.float32,
    )
    transform = LetterboxTransform(640, 640, 640, 1.0, 0, 0)

    detections = postprocess(
        output,
        transform=transform,
        labels=LABELS,
        confidence_threshold=0.35,
        iou_threshold=0.45,
    )

    assert len(detections) == 1
    detection = detections[0]
    assert detection.label == "d20"
    assert detection.confidence == pytest.approx(0.72)
    assert detection.box.x_min == pytest.approx(270)
    assert detection.box.y_min == pytest.approx(280)
    assert detection.box.x_max == pytest.approx(370)
    assert detection.box.y_max == pytest.approx(360)


def test_postprocess_maps_letterbox_coordinates_to_original_image() -> None:
    output = np.array(
        [
            [
                prediction(
                    cx=320,
                    cy=320,
                    width=320,
                    height=160,
                    objectness=1,
                    class_index=0,
                    class_probability=1,
                )
            ]
        ],
        dtype=np.float32,
    )
    transform = LetterboxTransform(200, 100, 640, 3.2, 0, 160)

    detection = postprocess(
        output,
        transform=transform,
        labels=LABELS,
        confidence_threshold=0.5,
        iou_threshold=0.45,
    )[0]

    assert detection.box.x_min == pytest.approx(50)
    assert detection.box.x_max == pytest.approx(150)
    assert detection.box.y_min == pytest.approx(25)
    assert detection.box.y_max == pytest.approx(75)


def test_postprocess_filters_detections_below_high_threshold() -> None:
    output = np.array(
        [
            [
                prediction(
                    cx=100,
                    cy=100,
                    width=40,
                    height=40,
                    objectness=1,
                    class_index=0,
                    class_probability=0.84,
                ),
                prediction(
                    cx=500,
                    cy=500,
                    width=40,
                    height=40,
                    objectness=1,
                    class_index=5,
                    class_probability=0.86,
                ),
            ]
        ],
        dtype=np.float32,
    )
    transform = LetterboxTransform(640, 640, 640, 1.0, 0, 0)

    detections = postprocess(
        output,
        transform=transform,
        labels=LABELS,
        confidence_threshold=0.85,
        iou_threshold=0.45,
    )

    assert len(detections) == 1
    assert detections[0].label == "d20"
    assert detections[0].confidence == pytest.approx(0.86)


def test_postprocess_rejects_unexpected_output_shape() -> None:
    transform = LetterboxTransform(640, 640, 640, 1.0, 0, 0)

    with pytest.raises(ValueError, match="output shape"):
        postprocess(
            np.zeros((1, 10, 10), dtype=np.float32),
            transform=transform,
            labels=LABELS,
            confidence_threshold=0.5,
            iou_threshold=0.45,
        )
