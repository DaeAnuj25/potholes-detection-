"""Camera/video loop, frame-rate sampling, overlays and JSONL telemetry."""

from __future__ import annotations

import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

from .calibration import DistanceCalibration, DistanceEstimate
from .detector import Detection, InferenceResult, UltralyticsPotholeDetector


class FrameRateGate:
    """Permit at most ``target_fps`` processing starts per wall-clock second."""

    def __init__(self, target_fps: float) -> None:
        if not math.isfinite(target_fps) or target_fps <= 0:
            raise ValueError("target_fps must be a positive finite number")
        self.interval_s = 1.0 / float(target_fps)
        self._next_due_s: float | None = None

    def should_process(self, now_s: float) -> bool:
        """Return whether a live-camera frame is due; excess frames are dropped."""
        if self._next_due_s is None:
            self._next_due_s = now_s + self.interval_s
            return True
        if now_s < self._next_due_s:
            return False

        late_by_s = now_s - self._next_due_s
        missed_slots = math.floor(late_by_s / self.interval_s)
        self._next_due_s += (missed_slots + 1) * self.interval_s
        return True


def parse_source(source: str) -> int | str:
    """Interpret a numeric source as a camera index and everything else as a URI/path."""
    value = source.strip()
    if value.isdecimal():
        return int(value)
    return value


def default_weights_path() -> Path:
    """Prefer best.pt if supplied, otherwise use the tracked model in this repo."""
    repository_root = Path(__file__).resolve().parents[1]
    for filename in ("best.pt", "new_model.pt"):
        candidate = repository_root / filename
        if candidate.is_file():
            return candidate
    return repository_root / "new_model.pt"


def _require_cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - depends on the user's install
        raise RuntimeError(
            "OpenCV is required. Install project dependencies with `pip install -e .`."
        ) from exc
    return cv2


def _is_local_file_source(source_text: str, source: int | str) -> bool:
    if isinstance(source, int):
        return False
    lowered = source_text.lower()
    if lowered.startswith(("rtsp://", "http://", "https://")):
        return False
    return Path(source_text).expanduser().is_file()


def _range_for_detection(
    detection: Detection,
    image_height: int,
    calibration: DistanceCalibration | None,
) -> DistanceEstimate | None:
    if calibration is None:
        return None
    closest_y = min(max(detection.closest_edge_y, 0.0), float(image_height - 1))
    return calibration.estimate(closest_y, image_height)


def _make_event(
    *,
    source_text: str,
    frame_index: int,
    source_time_ms: float | None,
    frame_width: int,
    frame_height: int,
    inference: InferenceResult,
    calibration: DistanceCalibration | None,
) -> dict[str, Any]:
    detections: list[dict[str, Any]] = []
    for detection in inference.detections:
        estimate = _range_for_detection(detection, frame_height, calibration)
        record: dict[str, Any] = {
            "class_id": detection.class_id,
            "class_name": detection.class_name,
            "confidence": detection.confidence,
            "box_xyxy_px": [detection.x1, detection.y1, detection.x2, detection.y2],
            "closest_edge_pixel_y": detection.closest_edge_y,
            "distance_from_rover_front_m": None,
            "distance_from_camera_ground_m": None,
            "distance_from_camera_center_m": None,
            "distance_status": "uncalibrated",
        }
        if estimate is not None:
            record.update(estimate.to_dict())
        detections.append(record)

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source": source_text,
        "frame_index": frame_index,
        "source_time_ms": source_time_ms,
        "image_width_px": frame_width,
        "image_height_px": frame_height,
        "inference_ms": inference.inference_ms,
        "detections": detections,
    }


def _distance_text(estimate: DistanceEstimate | None) -> str:
    if estimate is None:
        return "distance: uncalibrated"
    parts = [estimate.label]
    if estimate.distance_from_camera_center_m is not None:
        parts.append(f"camera slant {estimate.distance_from_camera_center_m:.2f} m")
    elif estimate.distance_from_camera_ground_m is not None:
        parts.append(f"camera ground {estimate.distance_from_camera_ground_m:.2f} m")
    return " | ".join(parts)


def _draw_overlay(
    cv2: Any,
    inference: InferenceResult,
    calibration: DistanceCalibration | None,
    target_fps: float,
    actual_fps: float,
    is_video_file: bool,
) -> Any:
    output = inference.gray_bgr.copy()
    height, width = output.shape[:2]

    for detection in inference.detections:
        x1 = max(0, min(width - 1, round(detection.x1)))
        y1 = max(0, min(height - 1, round(detection.y1)))
        x2 = max(0, min(width - 1, round(detection.x2)))
        y2 = max(0, min(height - 1, round(detection.y2)))
        estimate = _range_for_detection(detection, height, calibration)
        color = (40, 220, 70)
        if estimate is not None and estimate.status == "beyond_near_range":
            color = (0, 80, 255)

        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        label = (
            f"{detection.class_name} {detection.confidence:.2f} | "
            f"{_distance_text(estimate)}"
        )
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.48
        thickness = 1
        (text_width, text_height), baseline = cv2.getTextSize(
            label, font, font_scale, thickness
        )
        text_y = max(text_height + baseline + 4, y1)
        cv2.rectangle(
            output,
            (x1, text_y - text_height - baseline - 4),
            (min(width - 1, x1 + text_width + 6), text_y + 2),
            color,
            -1,
        )
        cv2.putText(
            output,
            label,
            (x1 + 3, text_y - baseline),
            font,
            font_scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA,
        )

    rate_description = (
        f"source sample <= {target_fps:g} fps"
        if is_video_file
        else f"target <= {target_fps:g} fps | actual {actual_fps:.1f} fps"
    )
    hud = (
        f"GRAYSCALE  {rate_description}  inference {inference.inference_ms:.0f} ms  "
        f"{width}x{height}"
    )
    cv2.rectangle(output, (0, 0), (width, 30), (20, 20, 20), -1)
    cv2.putText(
        output,
        hud,
        (8, 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )
    if not inference.detections:
        message = "No pothole detected in this frame"
        cv2.putText(
            output,
            message,
            (12, min(height - 12, 55)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (235, 235, 235),
            2,
            cv2.LINE_AA,
        )
    if calibration is None:
        message = "DISTANCE UNCALIBRATED - run the calibration tool"
        cv2.putText(
            output,
            message,
            (8, max(48, height - 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 190, 255),
            1,
            cv2.LINE_AA,
        )
    return output


def _write_video_frame(
    cv2: Any,
    writer: Any,
    output_path: Path,
    frame: Any,
    fps: float,
) -> Any:
    if writer is None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        height, width = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        if not writer.isOpened():
            writer.release()
            raise RuntimeError(f"Could not create output video: {output_path}")
    writer.write(frame)
    return writer


def _write_event(stream: TextIO, event: dict[str, Any]) -> None:
    stream.write(json.dumps(event, ensure_ascii=True, separators=(",", ":")) + "\n")
    stream.flush()


def run(args: Any) -> int:
    """Run a camera, stream, still image or video until it ends or Q is pressed."""
    cv2 = _require_cv2()
    source = parse_source(args.source)
    local_file = _is_local_file_source(args.source, source)
    source_path = Path(args.source).expanduser() if local_file else None
    if local_file and source_path is not None:
        source = str(source_path)
    image_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    single_image = (
        source_path is not None and source_path.suffix.lower() in image_suffixes
    )
    if args.save_image is not None and not single_image:
        raise ValueError("--save-image can only be used when --source is a still image")

    calibration = (
        DistanceCalibration.load(args.calibration)
        if args.calibration is not None
        else None
    )
    detector = UltralyticsPotholeDetector(
        weights=args.weights,
        image_size=args.image_size,
        confidence=args.confidence,
        iou=args.iou,
        device=args.device,
    )

    capture = None
    still_frame = None
    if single_image:
        still_frame = cv2.imread(str(source_path))
        if still_frame is None:
            raise RuntimeError(f"Could not read input image: {source_path}")
        source_fps = 0.0
    else:
        capture = cv2.VideoCapture(source)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(
                f"Could not open input source {args.source!r}. Check the camera index, "
                "video path or stream URL."
            )

        if not local_file:
            if args.width is not None:
                capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
            if args.height is not None:
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
            # Best effort: avoid processing a backlog of old camera frames when
            # inference is slower than capture. Some capture backends ignore this.
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if not math.isfinite(source_fps) or source_fps <= 0:
            source_fps = 0.0

    if local_file and not single_image and source_fps > 0:
        video_step_frames = source_fps / args.target_fps
    else:
        video_step_frames = 1.0

    jsonl_stream: TextIO | None = None
    close_jsonl = False
    if args.jsonl is not None:
        if args.jsonl == "-":
            jsonl_stream = sys.stdout
        else:
            jsonl_path = Path(args.jsonl).expanduser()
            jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            jsonl_stream = jsonl_path.open("w", encoding="utf-8")
            close_jsonl = True

    log_stream = sys.stderr if args.jsonl == "-" else sys.stdout

    def log(message: str) -> None:
        print(message, file=log_stream, flush=True)

    log(f"Weights: {detector.weights_path}")
    log(f"Input: {args.source} | grayscale preprocessing: on")
    log(f"Target processing rate: {args.target_fps:g} FPS")
    if calibration is None:
        log("Distance: uncalibrated (detections will still run)")
    else:
        log(
            f"Distance calibration: {len(calibration.points)} points, "
            f"reference={calibration.image_height_px}px high, rover-front origin"
        )
    if not args.headless:
        log("Press Q or Esc in the preview window to stop.")

    writer = None
    rate_gate = FrameRateGate(args.target_fps)
    frame_index = 0
    processed_frames = 0
    started_s = time.monotonic()
    next_video_frame = 0
    window_name = "Pothole rover - grayscale detection"

    try:
        while True:
            if single_image:
                if frame_index > 0:
                    break
                frame = still_frame
                ok = frame is not None
            else:
                assert capture is not None
                ok, frame = capture.read()
            if not ok or frame is None:
                break

            this_frame_index = frame_index
            frame_index += 1
            source_time_ms: float | None = 0.0 if single_image else None

            if local_file and not single_image:
                if source_fps > 0:
                    source_time_ms = this_frame_index * 1000.0 / source_fps
                    if this_frame_index + 1e-9 < next_video_frame:
                        continue
                    next_video_frame = this_frame_index + video_step_frames
                elif not rate_gate.should_process(time.monotonic()):
                    # Some video containers do not expose FPS metadata; use a
                    # wall-clock limit rather than silently processing unbounded.
                    continue
            elif (
                not local_file
                and not single_image
                and not rate_gate.should_process(time.monotonic())
            ):
                continue

            inference = detector.predict(frame)
            height, width = inference.gray_bgr.shape[:2]
            processed_frames += 1
            elapsed = max(time.monotonic() - started_s, 1e-9)
            actual_fps = processed_frames / elapsed
            output_frame = _draw_overlay(
                cv2=cv2,
                inference=inference,
                calibration=calibration,
                target_fps=args.target_fps,
                actual_fps=actual_fps,
                is_video_file=local_file and not single_image,
            )

            if args.save_video is not None:
                output_fps = (
                    min(source_fps, args.target_fps)
                    if local_file and not single_image and source_fps
                    else args.target_fps
                )
                writer = _write_video_frame(
                    cv2,
                    writer,
                    Path(args.save_video).expanduser(),
                    output_frame,
                    output_fps,
                )
            if args.save_image is not None:
                image_output = Path(args.save_image).expanduser()
                image_output.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(image_output), output_frame):
                    raise RuntimeError(
                        f"Could not write annotated image: {image_output}"
                    )

            event = _make_event(
                source_text=args.source,
                frame_index=this_frame_index,
                source_time_ms=source_time_ms,
                frame_width=width,
                frame_height=height,
                inference=inference,
                calibration=calibration,
            )
            if jsonl_stream is not None:
                _write_event(jsonl_stream, event)

            if inference.detections:
                details = "; ".join(
                    f"{d.class_name} {d.confidence:.2f}, "
                    f"{_distance_text(_range_for_detection(d, height, calibration))}"
                    for d in inference.detections
                )
            else:
                details = "no potholes"
            log(
                f"frame {this_frame_index}: {details} | "
                f"inference {inference.inference_ms:.0f} ms | "
                f"actual {actual_fps:.1f} FPS"
            )

            if not args.headless:
                cv2.imshow(window_name, output_frame)
                key = cv2.waitKey(0 if single_image else 1) & 0xFF
                if key in (ord("q"), 27):
                    break

        log(f"Stopped. Read {frame_index} frames; processed {processed_frames}.")
    finally:
        if capture is not None:
            capture.release()
        if writer is not None:
            writer.release()
        if close_jsonl and jsonl_stream is not None:
            jsonl_stream.close()
        if not args.headless:
            cv2.destroyAllWindows()

    return 0
