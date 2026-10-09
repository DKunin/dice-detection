import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import NoReturn

from dice_detector.api import create_app
from dice_detector.config import Settings
from dice_detector.detector import DiceDetector
from dice_detector.download import LicenseNotAcceptedError, download_artifacts
from dice_detector.images import decode_image


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dice-detector")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download_parser = subparsers.add_parser("download-model", help="download pinned model files")
    download_parser.add_argument("--model-dir", type=Path, default=Path("models"))
    download_parser.add_argument("--accept-license", action="store_true")

    detect_parser = subparsers.add_parser("detect", help="detect dice in one image")
    detect_parser.add_argument("image", type=Path)
    detect_parser.add_argument("--model-dir", type=Path, default=Path("models"))
    detect_parser.add_argument("--confidence", type=float, default=0.35)
    detect_parser.add_argument("--iou", type=float, default=0.45)
    detect_parser.add_argument("--threads", type=int, default=max(1, min(os.cpu_count() or 1, 6)))

    serve_parser = subparsers.add_parser("serve", help="start the HTTP API")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=18080)
    serve_parser.add_argument("--model-dir", type=Path, default=Path("models"))
    serve_parser.add_argument("--confidence", type=float, default=0.85)
    serve_parser.add_argument("--iou", type=float, default=0.45)
    return parser


def main() -> None:
    parser = build_parser()
    arguments = parser.parse_args()
    try:
        if arguments.command == "download-model":
            paths = download_artifacts(
                arguments.model_dir,
                accept_license=arguments.accept_license,
            )
            for path in paths:
                print(path)
            return
        if arguments.command == "detect":
            detector = DiceDetector.from_files(
                arguments.model_dir / "dice-shape.onnx",
                arguments.model_dir / "dice-shape.classes.json",
                confidence_threshold=arguments.confidence,
                iou_threshold=arguments.iou,
                threads=arguments.threads,
            )
            image = decode_image(arguments.image.read_bytes())
            print(json.dumps(asdict(detector.detect(image)), ensure_ascii=False, indent=2))
            return
        if arguments.command == "serve":
            import uvicorn

            settings = Settings(
                model_dir=arguments.model_dir,
                confidence_threshold=arguments.confidence,
                iou_threshold=arguments.iou,
            )
            uvicorn.run(create_app(settings=settings), host=arguments.host, port=arguments.port)
            return
        _fail(parser, f"unknown command: {arguments.command}")
    except (FileNotFoundError, LicenseNotAcceptedError, RuntimeError, ValueError) as error:
        _fail(parser, str(error))


def _fail(parser: argparse.ArgumentParser, message: str) -> NoReturn:
    parser.error(message)


if __name__ == "__main__":
    main()
