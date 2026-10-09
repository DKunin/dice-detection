import json
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageEnhance, ImageOps

from dice_detector.images import source_image_size
from dice_detector.postprocess import postprocess
from dice_detector.preprocess import letterbox
from dice_detector.types import Box, Detection, DetectionResult


class ModelContractError(RuntimeError):
    """Raised when model files do not match the expected detector contract."""


_FALLBACK_CANDIDATE_THRESHOLD = 0.35
_FALLBACK_CONTEXT_MARGIN = 0.04
_FALLBACK_MATCH_IOU = 0.25
_FALLBACK_TILE_MARGIN = 0.04


class SessionNode(Protocol):
    name: str
    shape: list[Any]


class Session(Protocol):
    def get_inputs(self) -> list[SessionNode]: ...

    def get_outputs(self) -> list[SessionNode]: ...

    def run(
        self,
        output_names: None,
        input_feed: dict[str, NDArray[np.float32]],
    ) -> list[NDArray[np.float32]]: ...


def load_labels(path: Path) -> tuple[str, ...]:
    """Load and validate the class order shipped alongside the ONNX model."""
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ModelContractError(f"cannot read class metadata: {path}") from error

    if metadata.get("family") != "yolox":
        raise ModelContractError("unsupported model family; expected 'yolox'")
    labels = metadata.get("classes")
    if (
        not isinstance(labels, list)
        or not labels
        or not all(isinstance(item, str) for item in labels)
    ):
        raise ModelContractError("class metadata must contain a non-empty string list")
    if len(set(labels)) != len(labels):
        raise ModelContractError("class labels must be unique")
    return tuple(labels)


class DiceDetector:
    """Run the pinned polyhedral-dice model with ONNX Runtime."""

    input_size = 640

    def __init__(
        self,
        *,
        session: Session,
        labels: tuple[str, ...],
        confidence_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        high_confidence_fallback: bool = False,
    ) -> None:
        self._session = session
        self.labels = labels
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.high_confidence_fallback = high_confidence_fallback
        self._input_name = self._validate_contract()

    @classmethod
    def from_files(
        cls,
        model_path: Path,
        classes_path: Path,
        *,
        confidence_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        threads: int = 0,
        high_confidence_fallback: bool = False,
    ) -> "DiceDetector":
        """Create a CPU inference session from downloaded model artifacts."""
        if not model_path.is_file():
            raise FileNotFoundError(f"model not found: {model_path}")
        if not classes_path.is_file():
            raise FileNotFoundError(f"class metadata not found: {classes_path}")

        import onnxruntime as ort

        options = ort.SessionOptions()
        if threads > 0:
            options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        return cls(
            session=session,
            labels=load_labels(classes_path),
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            high_confidence_fallback=high_confidence_fallback,
        )

    def detect(self, image: Image.Image) -> DetectionResult:
        """Detect dice and return both a binary answer and detailed boxes."""
        candidate_threshold = self.confidence_threshold
        if self.high_confidence_fallback:
            candidate_threshold = min(
                candidate_threshold,
                _FALLBACK_CANDIDATE_THRESHOLD,
            )
        candidates, inference_ms = self._infer(
            image,
            confidence_threshold=candidate_threshold,
        )
        detections = tuple(
            detection
            for detection in candidates
            if detection.confidence >= self.confidence_threshold
        )
        if self.high_confidence_fallback and not detections:
            detections, fallback_ms = self._detect_with_fallback(image, candidates)
            inference_ms += fallback_ms

        source_width, source_height = source_image_size(image)
        detections = self._map_to_source(
            detections,
            working_size=image.size,
            source_size=(source_width, source_height),
        )
        return DetectionResult(
            has_dice=bool(detections),
            count=len(detections),
            detections=detections,
            image_width=source_width,
            image_height=source_height,
            inference_ms=inference_ms,
        )

    def _infer(
        self,
        image: Image.Image,
        *,
        confidence_threshold: float,
    ) -> tuple[tuple[Detection, ...], float]:
        tensor, transform = letterbox(image, input_size=self.input_size)
        started = perf_counter()
        output = self._session.run(None, {self._input_name: tensor})[0]
        inference_ms = (perf_counter() - started) * 1_000
        detections = postprocess(
            output,
            transform=transform,
            labels=self.labels,
            confidence_threshold=confidence_threshold,
            iou_threshold=self.iou_threshold,
        )
        return detections, inference_ms

    def _detect_with_fallback(
        self,
        image: Image.Image,
        candidates: tuple[Detection, ...],
    ) -> tuple[tuple[Detection, ...], float]:
        inference_ms = 0.0
        augmented_groups: list[tuple[Detection, ...]] = []
        variants = (
            (ImageOps.mirror(image), _map_from_horizontal_flip),
            (image.transpose(Image.Transpose.ROTATE_90), _map_from_rotate_90),
            (ImageEnhance.Contrast(image).enhance(1.25), _identity_detection),
        )
        for variant, map_detection in variants:
            variant_detections, variant_ms = self._infer(
                variant,
                confidence_threshold=self.confidence_threshold,
            )
            inference_ms += variant_ms
            augmented_groups.append(
                tuple(
                    map_detection(detection, image.width, image.height)
                    for detection in variant_detections
                )
            )
            if len(augmented_groups) >= 2:
                confirmed = _confirmed_detections(candidates, augmented_groups)
                if confirmed:
                    return confirmed, inference_ms

        context_threshold = max(
            _FALLBACK_CANDIDATE_THRESHOLD,
            self.confidence_threshold - _FALLBACK_CONTEXT_MARGIN,
        )
        if max((item.confidence for item in candidates), default=0.0) < context_threshold:
            return (), inference_ms

        for x_min, y_min, x_max, y_max in _tile_boxes(image.width, image.height):
            tile_detections, tile_ms = self._infer(
                image.crop((x_min, y_min, x_max, y_max)),
                confidence_threshold=self.confidence_threshold,
            )
            inference_ms += tile_ms
            if tile_detections:
                return (
                    tuple(
                        Detection(
                            label=detection.label,
                            confidence=detection.confidence,
                            box=Box(
                                x_min=detection.box.x_min + x_min,
                                y_min=detection.box.y_min + y_min,
                                x_max=detection.box.x_max + x_min,
                                y_max=detection.box.y_max + y_min,
                            ),
                        )
                        for detection in tile_detections
                    ),
                    inference_ms,
                )
        return (), inference_ms

    @staticmethod
    def _map_to_source(
        detections: tuple[Detection, ...],
        *,
        working_size: tuple[int, int],
        source_size: tuple[int, int],
    ) -> tuple[Detection, ...]:
        if working_size == source_size:
            return detections
        scale_x = source_size[0] / working_size[0]
        scale_y = source_size[1] / working_size[1]
        return tuple(
            Detection(
                label=detection.label,
                confidence=detection.confidence,
                box=Box(
                    x_min=detection.box.x_min * scale_x,
                    y_min=detection.box.y_min * scale_y,
                    x_max=detection.box.x_max * scale_x,
                    y_max=detection.box.y_max * scale_y,
                ),
            )
            for detection in detections
        )

    def _validate_contract(self) -> str:
        inputs = self._session.get_inputs()
        outputs = self._session.get_outputs()
        if len(inputs) != 1 or list(inputs[0].shape) != [1, 3, self.input_size, self.input_size]:
            observed = [list(item.shape) for item in inputs]
            raise ModelContractError(
                f"unexpected model input contract: {observed}; expected [1, 3, 640, 640]"
            )
        expected_output_features = 5 + len(self.labels)
        if (
            len(outputs) != 1
            or len(outputs[0].shape) != 3
            or outputs[0].shape[0] != 1
            or outputs[0].shape[2] != expected_output_features
        ):
            observed = [list(item.shape) for item in outputs]
            raise ModelContractError(
                "unexpected model output contract: "
                f"{observed}; expected [1, N, {expected_output_features}]"
            )
        return inputs[0].name


def _identity_detection(
    detection: Detection,
    width: int,
    height: int,
) -> Detection:
    del width, height
    return detection


def _map_from_horizontal_flip(
    detection: Detection,
    width: int,
    height: int,
) -> Detection:
    del height
    return Detection(
        label=detection.label,
        confidence=detection.confidence,
        box=Box(
            x_min=width - detection.box.x_max,
            y_min=detection.box.y_min,
            x_max=width - detection.box.x_min,
            y_max=detection.box.y_max,
        ),
    )


def _map_from_rotate_90(
    detection: Detection,
    width: int,
    height: int,
) -> Detection:
    del height
    return Detection(
        label=detection.label,
        confidence=detection.confidence,
        box=Box(
            x_min=width - detection.box.y_max,
            y_min=detection.box.x_min,
            x_max=width - detection.box.y_min,
            y_max=detection.box.x_max,
        ),
    )


def _confirmed_detections(
    candidates: tuple[Detection, ...],
    augmented_groups: list[tuple[Detection, ...]],
) -> tuple[Detection, ...]:
    confirmed: list[Detection] = []
    for candidate in candidates:
        matches: list[Detection] = []
        for group in augmented_groups:
            group_matches = (
                detection
                for detection in group
                if detection.label == candidate.label
                and _box_iou(detection.box, candidate.box) >= _FALLBACK_MATCH_IOU
            )
            best_match = max(group_matches, key=lambda item: item.confidence, default=None)
            if best_match is not None:
                matches.append(best_match)
        if len(matches) >= 2:
            confirmed.append(max(matches, key=lambda item: item.confidence))
    return tuple(confirmed)


def _box_iou(left: Box, right: Box) -> float:
    intersection_width = max(
        0.0,
        min(left.x_max, right.x_max) - max(left.x_min, right.x_min),
    )
    intersection_height = max(
        0.0,
        min(left.y_max, right.y_max) - max(left.y_min, right.y_min),
    )
    intersection = intersection_width * intersection_height
    left_area = max(0.0, left.x_max - left.x_min) * max(
        0.0, left.y_max - left.y_min
    )
    right_area = max(0.0, right.x_max - right.x_min) * max(
        0.0, right.y_max - right.y_min
    )
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def _tile_boxes(width: int, height: int) -> tuple[tuple[int, int, int, int], ...]:
    boxes: list[tuple[int, int, int, int]] = []
    for row in range(3):
        for column in range(3):
            x_min = round(width * max(0.0, column / 3 - _FALLBACK_TILE_MARGIN))
            y_min = round(height * max(0.0, row / 3 - _FALLBACK_TILE_MARGIN))
            x_max = round(width * min(1.0, (column + 1) / 3 + _FALLBACK_TILE_MARGIN))
            y_max = round(height * min(1.0, (row + 1) / 3 + _FALLBACK_TILE_MARGIN))
            boxes.append((x_min, y_min, x_max, y_max))
    return tuple(boxes)
