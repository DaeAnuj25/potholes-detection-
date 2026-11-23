"""YOLO inference adapter that guarantees grayscale input to the model."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str

    @property
    def closest_edge_y(self) -> float:
        """The lower box edge approximates the nearest visible ground point."""
        return self.y2


@dataclass(frozen=True)
class InferenceResult:
    # Three-channel BGR image where all channels contain the same grayscale data.
    # This keeps the Ultralytics input shape valid while preserving grayscale-only
    # appearance for the trained model.
    gray_bgr: Any
    detections: tuple[Detection, ...]
    inference_ms: float


def grayscale_bgr(frame: Any) -> Any:
    """Convert a captured BGR/BGRA/grayscale frame to repeated grayscale BGR.

    Ultralytics expects three-channel image arrays. Passing a grayscale image
    replicated into B, G and R means the model receives grayscale pixel values
    without changing its expected input channel count.
    """
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - depends on the user's install
        raise RuntimeError(
            "OpenCV is required. Install project dependencies with `pip install -e .`."
        ) from exc

    shape = getattr(frame, "shape", None)
    if shape is None or len(shape) not in (2, 3):
        raise ValueError("Camera frame must be a 2-D or 3-D image array")

    if len(shape) == 2:
        gray = frame
    elif shape[2] == 1:
        gray = frame[:, :, 0]
    elif shape[2] == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    elif shape[2] == 4:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
    else:
        raise ValueError(f"Unsupported camera frame channel count: {shape[2]}")

    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


class UltralyticsPotholeDetector:
    """Load the repository's Ultralytics checkpoint once and reuse it per frame."""

    def __init__(
        self,
        weights: str | Path,
        image_size: int = 640,
        confidence: float = 0.25,
        iou: float = 0.45,
        device: str | None = None,
    ) -> None:
        weights_path = Path(weights).expanduser()
        if not weights_path.is_file():
            raise FileNotFoundError(
                f"Model weights not found: {weights_path}. The repository's supplied "
                "checkpoint is new_model.pt; use --weights to select another .pt file."
            )
        if image_size < 32:
            raise ValueError("image_size must be at least 32 pixels")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if not 0.0 <= iou <= 1.0:
            raise ValueError("iou must be between 0 and 1")

        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - depends on the user's install
            raise RuntimeError(
                "Ultralytics is required to load the .pt model. Install project "
                "dependencies with `pip install -e .`."
            ) from exc

        self.weights_path = weights_path
        self.image_size = int(image_size)
        self.confidence = float(confidence)
        self.iou = float(iou)
        self.device = device
        self.model = YOLO(str(weights_path))

    def predict(self, frame: Any) -> InferenceResult:
        """Run one frame through grayscale conversion and the YOLO model."""
        started = time.perf_counter()
        model_input = grayscale_bgr(frame)
        options: dict[str, Any] = {
            "source": model_input,
            "imgsz": self.image_size,
            "conf": self.confidence,
            "iou": self.iou,
            "verbose": False,
        }
        if self.device is not None:
            options["device"] = self.device

        results = self.model.predict(**options)
        if not results:
            return InferenceResult(
                gray_bgr=model_input,
                detections=(),
                inference_ms=(time.perf_counter() - started) * 1000.0,
            )

        result = results[0]
        detections: list[Detection] = []
        boxes = getattr(result, "boxes", None)
        if boxes is not None and len(boxes):
            coordinates = boxes.xyxy.detach().cpu().tolist()
            confidences = boxes.conf.detach().cpu().tolist()
            class_ids = boxes.cls.detach().cpu().tolist()
            names = getattr(result, "names", getattr(self.model, "names", {}))

            for coords, confidence, raw_class_id in zip(
                coordinates, confidences, class_ids
            ):
                class_id = int(raw_class_id)
                if isinstance(names, dict):
                    class_name = str(names.get(class_id, class_id))
                elif isinstance(names, (list, tuple)) and class_id < len(names):
                    class_name = str(names[class_id])
                else:
                    class_name = str(class_id)
                detections.append(
                    Detection(
                        x1=float(coords[0]),
                        y1=float(coords[1]),
                        x2=float(coords[2]),
                        y2=float(coords[3]),
                        confidence=float(confidence),
                        class_id=class_id,
                        class_name=class_name,
                    )
                )

        return InferenceResult(
            gray_bgr=model_input,
            detections=tuple(detections),
            inference_ms=(time.perf_counter() - started) * 1000.0,
        )
