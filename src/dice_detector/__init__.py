"""ONNX detector for standard polyhedral RPG dice."""

from dice_detector.detector import DiceDetector
from dice_detector.types import Box, Detection, DetectionResult

__all__ = ["Box", "Detection", "DetectionResult", "DiceDetector"]
__version__ = "0.1.0"
