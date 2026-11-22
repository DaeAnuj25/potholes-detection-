"""Command-line entry point for rover camera inference."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from .runtime import default_weights_path, run


def _positive_float(value: str) -> float:
    try:
        result = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return result


def _positive_int(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pothole-rover",
        description=(
            "Detect potholes from a camera/video after grayscale conversion. "
            "The default model is the new_model.pt checkpoint in this repository."
        ),
    )
    parser.add_argument(
        "--source",
        default="0",
        help="camera index (default: 0), video/image path, or RTSP/HTTP stream URL",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=default_weights_path(),
        help="Ultralytics .pt checkpoint (default: best.pt if present, else new_model.pt)",
    )
    parser.add_argument(
        "--target-fps",
        type=_positive_float,
        default=5.0,
        help="maximum live-camera processing rate and video sampling rate (default: 5)",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=None,
        help="JSON ground-plane calibration created by python -m pothole_rover.calibrate",
    )
    parser.add_argument(
        "--image-size",
        type=_positive_int,
        default=640,
        help="YOLO inference image size (default: 640)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="minimum detection confidence between 0 and 1 (default: 0.25)",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="NMS IoU threshold between 0 and 1 (default: 0.45)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Ultralytics device, e.g. cpu, 0, or 0,1 (default: automatic)",
    )
    parser.add_argument(
        "--width",
        type=_positive_int,
        default=None,
        help="request camera capture width in pixels (keep this fixed after calibration)",
    )
    parser.add_argument(
        "--height",
        type=_positive_int,
        default=None,
        help="request camera capture height in pixels (keep this fixed after calibration)",
    )
    parser.add_argument(
        "--save-video",
        type=Path,
        default=None,
        help="optional path for an annotated MP4 output",
    )
    parser.add_argument(
        "--save-image",
        type=Path,
        default=None,
        help="save an annotated still image (only valid with an image --source)",
    )
    parser.add_argument(
        "--jsonl",
        default=None,
        help="optional JSONL detections output path; use '-' to write JSONL to stdout",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="do not open an OpenCV preview window (useful on a rover without a display)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not 0.0 <= args.confidence <= 1.0:
        parser.error("--confidence must be between 0 and 1")
    if not 0.0 <= args.iou <= 1.0:
        parser.error("--iou must be between 0 and 1")

    try:
        return run(args)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    except (FileNotFoundError, RuntimeError, TypeError, ValueError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":  # pragma: no cover - covered by module entry point
    raise SystemExit(main())
