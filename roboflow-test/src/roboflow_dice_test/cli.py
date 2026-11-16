"""Command-line entry point for a single Roboflow model test."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from roboflow_dice_test.client import RoboflowConfig, RoboflowTestError, run_inference


def _confidence(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number between 0 and 1") from exc
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Test an RPG dice photo with Roboflow")
    parser.add_argument("image", type=Path, help="path to a JPEG, PNG, or WebP image")
    parser.add_argument(
        "--confidence",
        type=_confidence,
        help="minimum confidence from 0 to 1 (default: ROBOFLOW_CONFIDENCE or 0.35)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = RoboflowConfig.from_env()
        if args.confidence is not None:
            config = replace(config, confidence=args.confidence)
        result = run_inference(args.image, config)
    except RoboflowTestError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
