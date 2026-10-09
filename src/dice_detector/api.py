from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Annotated, Protocol

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from PIL import Image
from pydantic import BaseModel
from starlette.responses import JSONResponse

from dice_detector.config import Settings
from dice_detector.detector import DiceDetector
from dice_detector.images import InvalidImageError, decode_image
from dice_detector.types import DetectionResult


class Detector(Protocol):
    def detect(self, image: Image.Image) -> DetectionResult: ...


class BoxResponse(BaseModel):
    x_min: float
    y_min: float
    x_max: float
    y_max: float


class DetectionResponse(BaseModel):
    label: str
    confidence: float
    box: BoxResponse


class ResultResponse(BaseModel):
    has_dice: bool
    count: int
    detections: list[DetectionResponse]
    image_width: int
    image_height: int
    inference_ms: float


def create_app(
    *,
    settings: Settings | None = None,
    detector: Detector | None = None,
    model_loader: Callable[[], Detector] | None = None,
) -> FastAPI:
    """Create an API app without loading the model during module import."""
    resolved_settings = settings or Settings.from_env()
    resolved_settings.validate()

    def default_model_loader() -> Detector:
        return DiceDetector.from_files(
            resolved_settings.model_path,
            resolved_settings.classes_path,
            confidence_threshold=resolved_settings.confidence_threshold,
            iou_threshold=resolved_settings.iou_threshold,
            threads=resolved_settings.onnx_threads,
            high_confidence_fallback=resolved_settings.high_confidence_fallback,
        )

    load_model = model_loader or default_model_loader

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        application.state.detector = detector
        application.state.model_error = None
        if detector is None:
            try:
                application.state.detector = await run_in_threadpool(load_model)
            except (FileNotFoundError, RuntimeError, ValueError) as error:
                application.state.model_error = str(error)
        yield

    application = FastAPI(
        title="Dice Detector",
        version="0.1.0",
        description="Binary and per-object RPG dice detection with ONNX Runtime.",
        lifespan=lifespan,
    )

    @application.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @application.get("/ready", response_model=None)
    async def ready() -> dict[str, bool] | JSONResponse:
        if application.state.detector is None:
            return JSONResponse(
                status_code=503,
                content={
                    "ready": False,
                    "detail": application.state.model_error or "model is not loaded",
                },
            )
        return {"ready": True}

    @application.post("/v1/detect", response_model=ResultResponse)
    async def detect(file: Annotated[UploadFile, File(...)]) -> ResultResponse:
        active_detector: Detector | None = application.state.detector
        if active_detector is None:
            raise HTTPException(
                status_code=503,
                detail=application.state.model_error or "model is not loaded",
            )

        payload = await file.read(resolved_settings.max_upload_bytes + 1)
        if len(payload) > resolved_settings.max_upload_bytes:
            raise HTTPException(status_code=413, detail="uploaded file is too large")
        try:
            image = decode_image(
                payload,
                max_pixels=resolved_settings.max_image_pixels,
                max_source_pixels=resolved_settings.max_source_pixels,
            )
        except InvalidImageError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

        result = await run_in_threadpool(active_detector.detect, image)
        return ResultResponse.model_validate(asdict(result))

    return application


app = create_app()
