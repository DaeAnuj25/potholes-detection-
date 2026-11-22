"""Interactively collect image-row/known-distance pairs for range estimation."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from .calibration import CalibrationPoint, DistanceCalibration


def _parse_sample(value: str) -> tuple[Path, float]:
    try:
        image_text, distance_text = value.rsplit("=", 1)
        distance_m = float(distance_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "use IMAGE_PATH=DISTANCE_METERS, for example samples/near.jpg=0.5"
        ) from exc
    if not image_text:
        raise argparse.ArgumentTypeError("sample image path cannot be empty")
    if not math.isfinite(distance_m) or distance_m < 0:
        raise argparse.ArgumentTypeError(
            "sample distance must be a finite, non-negative number"
        )
    return Path(image_text).expanduser(), distance_m


def _choose_image_row(
    cv2: object, image_path: Path, distance_m: float
) -> tuple[int, int]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read calibration image: {image_path}")
    image_height, image_width = image.shape[:2]
    scale = min(1.0, 1280.0 / image_width, 800.0 / image_height)
    if scale < 1.0:
        display_image = cv2.resize(
            image,
            (round(image_width * scale), round(image_height * scale)),
            interpolation=cv2.INTER_AREA,
        )
    else:
        display_image = image.copy()

    state: dict[str, int | None] = {"pixel_y": None}
    window_name = f"Range calibration - {image_path.name}"

    def on_mouse(event: int, x: int, y: int, flags: int, userdata: object) -> None:
        del x, flags, userdata
        if event == cv2.EVENT_LBUTTONDOWN:
            state["pixel_y"] = max(0, min(image_height - 1, round(y / scale)))

    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, on_mouse)
    while state["pixel_y"] is None:
        preview = display_image.copy()
        cv2.putText(
            preview,
            f"Measured from rover front: {distance_m:.2f} m - click the ground point",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            preview,
            "For a pothole, click its nearest visible edge. Press Q to cancel.",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.imshow(window_name, preview)
        key = cv2.waitKey(20) & 0xFF
        if key in (ord("q"), 27):
            cv2.destroyWindow(window_name)
            raise RuntimeError(f"Calibration cancelled while choosing {image_path}")

    pixel_y = int(state["pixel_y"])
    cv2.destroyWindow(window_name)
    return image_height, pixel_y


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m pothole_rover.calibrate",
        description=(
            "Click known ground points in fixed-camera images and save a distance "
            "calibration. Each sample distance is measured from the rover's front edge."
        ),
    )
    parser.add_argument(
        "--sample",
        action="append",
        type=_parse_sample,
        required=True,
        metavar="IMAGE=DISTANCE_M",
        help="repeat once per known distance, e.g. --sample near.jpg=0.5",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("calibration.json"),
        help="output calibration JSON (default: calibration.json)",
    )
    parser.add_argument(
        "--camera-to-rover-front-m",
        type=float,
        default=None,
        help=(
            "horizontal distance from camera ground projection to rover front; positive "
            "if the camera is behind the front edge"
        ),
    )
    parser.add_argument(
        "--camera-height-m",
        type=float,
        default=None,
        help="camera optical-center height above the flat ground (enables slant range)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        import cv2
    except ImportError:  # pragma: no cover - depends on the user's install
        parser.exit(
            2, "error: OpenCV is required; install the project dependencies first.\n"
        )

    if args.camera_height_m is not None and args.camera_to_rover_front_m is None:
        parser.error("--camera-height-m also needs --camera-to-rover-front-m")

    points: list[CalibrationPoint] = []
    reference_size: tuple[int, int] | None = None
    for image_path, distance_m in args.sample:
        if not image_path.is_file():
            parser.error(f"calibration image not found: {image_path}")
        image_height, pixel_y = _choose_image_row(cv2, image_path, distance_m)
        image = cv2.imread(str(image_path))
        image_width = int(image.shape[1])
        size = (image_width, image_height)
        if reference_size is None:
            reference_size = size
        elif size != reference_size:
            parser.error(
                "all calibration images must have the same resolution; "
                f"expected {reference_size[0]}x{reference_size[1]}, got "
                f"{image_width}x{image_height} in {image_path}"
            )
        points.append(CalibrationPoint(pixel_y=pixel_y, distance_m=distance_m))
        print(f"Captured {image_path}: y={pixel_y}px, distance={distance_m:.3f}m")

    assert reference_size is not None
    try:
        calibration = DistanceCalibration(
            image_height_px=reference_size[1],
            points=points,
            camera_to_rover_front_m=args.camera_to_rover_front_m,
            camera_height_m=args.camera_height_m,
        )
        calibration.save(args.output)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    print(f"Saved {len(points)} calibration points to {args.output.expanduser()}")
    if len(points) < 4:
        print(
            "Tip: 4-8 points spanning the rover's useful range usually calibrate better."
        )
    print(
        "Keep camera position, lens/FOV, crop and capture resolution unchanged at run time."
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised with -m
    raise SystemExit(main())
