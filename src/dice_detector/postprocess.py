import numpy as np
from numpy.typing import NDArray

from dice_detector.preprocess import LetterboxTransform
from dice_detector.types import Box, Detection


def postprocess(
    output: NDArray[np.floating],
    *,
    transform: LetterboxTransform,
    labels: tuple[str, ...],
    confidence_threshold: float,
    iou_threshold: float,
    max_detections: int = 100,
) -> tuple[Detection, ...]:
    """Decode YOLOX-style rows and apply class-agnostic non-max suppression."""
    _validate_threshold("confidence_threshold", confidence_threshold)
    _validate_threshold("iou_threshold", iou_threshold)
    if max_detections <= 0:
        raise ValueError("max_detections must be positive")

    predictions = np.asarray(output)
    expected_features = 5 + len(labels)
    if (
        predictions.ndim != 3
        or predictions.shape[0] != 1
        or predictions.shape[2] != expected_features
    ):
        raise ValueError(
            "unexpected model output shape: "
            f"{predictions.shape}; expected [1, N, {expected_features}]"
        )

    rows = predictions[0]
    class_indices = np.argmax(rows[:, 5:], axis=1)
    class_probabilities = rows[np.arange(rows.shape[0]), class_indices + 5]
    scores = rows[:, 4] * class_probabilities
    valid = np.isfinite(rows).all(axis=1) & (scores >= confidence_threshold)
    if not np.any(valid):
        return ()

    rows = rows[valid]
    scores = scores[valid]
    class_indices = class_indices[valid]
    boxes = _center_to_corners(rows[:, :4])
    positive_area = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    boxes = boxes[positive_area]
    scores = scores[positive_area]
    class_indices = class_indices[positive_area]
    if boxes.shape[0] == 0:
        return ()

    keep = _nms(boxes, scores, iou_threshold=iou_threshold, limit=max_detections)
    detections: list[Detection] = []
    for index in keep:
        mapped_box = _map_to_original(boxes[index], transform)
        if mapped_box.x_max <= mapped_box.x_min or mapped_box.y_max <= mapped_box.y_min:
            continue
        detections.append(
            Detection(
                label=labels[int(class_indices[index])],
                confidence=float(scores[index]),
                box=mapped_box,
            )
        )
    return tuple(detections)


def _validate_threshold(name: str, value: float) -> None:
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be between 0 and 1")


def _center_to_corners(boxes: NDArray[np.floating]) -> NDArray[np.floating]:
    corners = np.empty_like(boxes)
    corners[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    corners[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    corners[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    corners[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return corners


def _nms(
    boxes: NDArray[np.floating],
    scores: NDArray[np.floating],
    *,
    iou_threshold: float,
    limit: int,
) -> list[int]:
    order = np.argsort(scores)[::-1]
    keep: list[int] = []
    while order.size and len(keep) < limit:
        current = int(order[0])
        keep.append(current)
        remaining = order[1:]
        if remaining.size == 0:
            break
        overlaps = _intersection_over_union(boxes[current], boxes[remaining])
        order = remaining[overlaps < iou_threshold]
    return keep


def _intersection_over_union(
    box: NDArray[np.floating],
    boxes: NDArray[np.floating],
) -> NDArray[np.float64]:
    x_min = np.maximum(box[0], boxes[:, 0])
    y_min = np.maximum(box[1], boxes[:, 1])
    x_max = np.minimum(box[2], boxes[:, 2])
    y_max = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(0, x_max - x_min) * np.maximum(0, y_max - y_min)
    box_area = max(0, float(box[2] - box[0])) * max(0, float(box[3] - box[1]))
    other_areas = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(
        0, boxes[:, 3] - boxes[:, 1]
    )
    union = box_area + other_areas - intersection
    result = np.zeros(intersection.shape, dtype=np.float64)
    np.divide(
        intersection,
        union,
        out=result,
        where=union > 0,
    )
    return result


def _map_to_original(
    box: NDArray[np.floating],
    transform: LetterboxTransform,
) -> Box:
    x_min = (float(box[0]) - transform.pad_x) / transform.scale
    y_min = (float(box[1]) - transform.pad_y) / transform.scale
    x_max = (float(box[2]) - transform.pad_x) / transform.scale
    y_max = (float(box[3]) - transform.pad_y) / transform.scale
    return Box(
        x_min=min(max(x_min, 0.0), float(transform.original_width)),
        y_min=min(max(y_min, 0.0), float(transform.original_height)),
        x_max=min(max(x_max, 0.0), float(transform.original_width)),
        y_max=min(max(y_max, 0.0), float(transform.original_height)),
    )
