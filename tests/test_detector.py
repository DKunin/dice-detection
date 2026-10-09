import json
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from dice_detector.detector import DiceDetector, ModelContractError, load_labels
from dice_detector.images import decode_image
from dice_detector.types import Box


class Node:
    def __init__(self, name: str, shape: list[int]) -> None:
        self.name = name
        self.shape = shape


class FakeSession:
    def __init__(self, output: np.ndarray) -> None:
        self.output = output
        self.received: dict[str, np.ndarray] | None = None

    def get_inputs(self) -> list[Node]:
        return [Node("images__rgb_0_1", [1, 3, 640, 640])]

    def get_outputs(self) -> list[Node]:
        return [Node("output", [1, 8400, 11])]

    def run(self, output_names: None, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        self.received = inputs
        return [self.output]


class SequenceSession(FakeSession):
    def __init__(self, outputs: list[np.ndarray]) -> None:
        super().__init__(outputs[0])
        self.outputs = outputs
        self.calls = 0

    def run(self, output_names: None, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        self.received = inputs
        output = self.outputs[self.calls]
        self.calls += 1
        return [output]


def model_output(
    *detections: tuple[float, float, float, float, float, int],
) -> np.ndarray:
    output = np.zeros((1, 8400, 11), dtype=np.float32)
    for index, (x, y, width, height, confidence, class_index) in enumerate(detections):
        output[0, index, :5] = [x, y, width, height, confidence]
        output[0, index, 5 + class_index] = 1
    return output


def test_detector_returns_binary_result_and_detections() -> None:
    output = np.zeros((1, 8400, 11), dtype=np.float32)
    output[0, 0, :5] = [320, 320, 100, 100, 0.9]
    output[0, 0, 10] = 0.8
    session = FakeSession(output)
    detector = DiceDetector(session=session, labels=("d4", "d6", "d8", "d10", "d12", "d20"))

    result = detector.detect(Image.new("RGB", (640, 640), "white"))

    assert result.has_dice is True
    assert result.count == 1
    assert result.detections[0].label == "d20"
    assert result.image_width == 640
    assert result.image_height == 640
    assert result.inference_ms >= 0
    assert session.received is not None
    assert session.received["images__rgb_0_1"].shape == (1, 3, 640, 640)


def test_detector_maps_downscaled_boxes_to_source_coordinates() -> None:
    output = np.zeros((1, 8400, 11), dtype=np.float32)
    output[0, 0, :5] = [320, 320, 320, 320, 1]
    output[0, 0, 5] = 1
    buffer = BytesIO()
    Image.new("RGB", (20, 20), "white").save(buffer, format="PNG")
    image = decode_image(buffer.getvalue(), max_pixels=100)
    detector = DiceDetector(
        session=FakeSession(output),
        labels=("d4", "d6", "d8", "d10", "d12", "d20"),
    )

    result = detector.detect(image)

    assert (result.image_width, result.image_height) == (20, 20)
    assert result.detections[0].box == Box(5, 5, 15, 15)


def test_high_confidence_fallback_stops_after_base_detection() -> None:
    session = SequenceSession([model_output((320, 320, 100, 100, 0.9, 5))])
    detector = DiceDetector(
        session=session,
        labels=("d4", "d6", "d8", "d10", "d12", "d20"),
        confidence_threshold=0.85,
        high_confidence_fallback=True,
    )

    result = detector.detect(Image.new("RGB", (640, 640), "white"))

    assert result.has_dice is True
    assert result.detections[0].confidence == pytest.approx(0.9)
    assert session.calls == 1


def test_high_confidence_fallback_requires_two_matching_transformed_views() -> None:
    empty = model_output()
    session = SequenceSession(
        [
            model_output((100, 100, 40, 40, 0.82, 5)),
            model_output((540, 100, 40, 40, 0.9, 5)),
            model_output((100, 540, 40, 40, 0.91, 5)),
            empty,
        ]
    )
    detector = DiceDetector(
        session=session,
        labels=("d4", "d6", "d8", "d10", "d12", "d20"),
        confidence_threshold=0.85,
        high_confidence_fallback=True,
    )

    result = detector.detect(Image.new("RGB", (640, 640), "white"))

    assert result.has_dice is True
    assert result.count == 1
    assert result.detections[0].confidence == pytest.approx(0.91)
    assert result.detections[0].box == pytest.approx(Box(80, 80, 120, 120))
    assert session.calls == 3


def test_high_confidence_fallback_rejects_one_transformed_view_without_context() -> None:
    empty = model_output()
    session = SequenceSession(
        [
            model_output((100, 100, 40, 40, 0.7, 5)),
            model_output((540, 100, 40, 40, 0.9, 5)),
            empty,
            empty,
        ]
    )
    detector = DiceDetector(
        session=session,
        labels=("d4", "d6", "d8", "d10", "d12", "d20"),
        confidence_threshold=0.85,
        high_confidence_fallback=True,
    )

    result = detector.detect(Image.new("RGB", (640, 640), "white"))

    assert result.has_dice is False
    assert session.calls == 4


def test_high_confidence_fallback_uses_tiles_for_near_threshold_context() -> None:
    empty = model_output()
    session = SequenceSession(
        [
            model_output((100, 100, 40, 40, 0.82, 5)),
            empty,
            empty,
            empty,
            model_output((320, 320, 100, 100, 0.88, 1)),
        ]
    )
    detector = DiceDetector(
        session=session,
        labels=("d4", "d6", "d8", "d10", "d12", "d20"),
        confidence_threshold=0.85,
        high_confidence_fallback=True,
    )

    result = detector.detect(Image.new("RGB", (640, 640), "white"))

    assert result.has_dice is True
    assert result.detections[0].label == "d6"
    assert result.detections[0].confidence == pytest.approx(0.88)
    assert session.calls == 5


def test_detector_validates_model_contract() -> None:
    session = FakeSession(np.zeros((1, 8400, 11), dtype=np.float32))
    session.get_outputs = lambda: [Node("output", [1, 8400, 10])]  # type: ignore[method-assign]

    with pytest.raises(ModelContractError, match="output"):
        DiceDetector(session=session, labels=("d4", "d6", "d8", "d10", "d12", "d20"))


def test_load_labels_validates_metadata(tmp_path: Path) -> None:
    classes_path = tmp_path / "classes.json"
    classes_path.write_text(
        json.dumps({"classes": ["d4", "d6"], "family": "unexpected"}),
        encoding="utf-8",
    )

    with pytest.raises(ModelContractError, match="family"):
        load_labels(classes_path)
