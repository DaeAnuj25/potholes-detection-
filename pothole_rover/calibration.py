"""Ground-plane distance calibration for a fixed, forward-facing camera.

A pothole's apparent width/area is not a reliable range cue because holes have
very different real-world sizes. This module maps the image row of the nearest
visible edge to a measured ground distance instead.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path


@dataclass(frozen=True)
class CalibrationPoint:
    """One measured image-row / ground-distance pair."""

    pixel_y: float
    distance_m: float


@dataclass(frozen=True)
class DistanceEstimate:
    """A range estimate and the limits of the calibration that produced it."""

    distance_from_rover_front_m: float
    status: str
    distance_from_camera_ground_m: float | None = None
    distance_from_camera_center_m: float | None = None

    @property
    def label(self) -> str:
        """Short, ASCII-only label suitable for OpenCV overlays."""
        value = self.distance_from_rover_front_m
        if self.status == "beyond_near_range":
            return f"<= {value:.2f} m from rover front"
        if self.status == "beyond_far_range":
            return f">= {value:.2f} m from rover front"
        return f"~ {value:.2f} m from rover front"

    def to_dict(self) -> dict[str, float | str | None]:
        return {
            "distance_from_rover_front_m": self.distance_from_rover_front_m,
            "distance_from_camera_ground_m": self.distance_from_camera_ground_m,
            "distance_from_camera_center_m": self.distance_from_camera_center_m,
            "distance_status": self.status,
        }


class DistanceCalibration:
    """Piecewise-linear image-row to ground-distance calibration.

    Distances in ``points`` are measured forward from the rover's front edge.
    ``camera_to_rover_front_m`` is the horizontal ground distance from the
    camera's vertical ground projection to the rover front; it is positive when
    the camera is behind the rover front and negative when it is ahead.
    """

    VERSION = 1

    def __init__(
        self,
        image_height_px: int,
        points: Iterable[CalibrationPoint],
        camera_to_rover_front_m: float | None = None,
        camera_height_m: float | None = None,
    ) -> None:
        if isinstance(image_height_px, bool) or int(image_height_px) < 2:
            raise ValueError("image_height_px must be an integer of at least 2")
        self.image_height_px = int(image_height_px)

        normalized_points = tuple(
            CalibrationPoint(float(point.pixel_y), float(point.distance_m))
            for point in points
        )
        if len(normalized_points) < 2:
            raise ValueError("At least two calibration points are required")
        if any(
            not math.isfinite(point.pixel_y) or not math.isfinite(point.distance_m)
            for point in normalized_points
        ):
            raise ValueError("Calibration points must contain finite numbers")
        if any(
            point.pixel_y < 0 or point.pixel_y > self.image_height_px - 1
            for point in normalized_points
        ):
            raise ValueError(
                f"pixel_y must be between 0 and {self.image_height_px - 1} "
                "for the calibration image height"
            )
        if any(point.distance_m < 0 for point in normalized_points):
            raise ValueError("Calibration distances cannot be negative")

        ordered = tuple(sorted(normalized_points, key=lambda point: point.pixel_y))
        for upper_row, lower_row in pairwise(ordered):
            if math.isclose(upper_row.pixel_y, lower_row.pixel_y, abs_tol=1e-9):
                raise ValueError("Each calibration point must use a distinct image row")
            # On a normal forward-facing camera, lower image rows correspond to
            # nearer points on the ground plane, so their measured distance is less.
            if lower_row.distance_m >= upper_row.distance_m:
                raise ValueError(
                    "Distances must decrease as pixel_y increases. Check that "
                    "the clicked points and measured distances are in the right order."
                )

        self.points = ordered
        self.camera_to_rover_front_m = self._optional_finite(
            camera_to_rover_front_m, "camera_to_rover_front_m"
        )
        self.camera_height_m = self._optional_finite(camera_height_m, "camera_height_m")
        if self.camera_height_m is not None and self.camera_height_m < 0:
            raise ValueError("camera_height_m cannot be negative")

    @staticmethod
    def _optional_finite(value: float | None, name: str) -> float | None:
        if value is None:
            return None
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"{name} must be a finite number")
        return result

    def estimate(
        self, pixel_y: float, current_image_height_px: int
    ) -> DistanceEstimate:
        """Estimate distance for the closest (bottom) point of a detection box.

        The input row is scaled to the calibration image height. Keep the same
        camera, lens/FOV, crop and aspect ratio at run time as during calibration.
        Out-of-range rows return a conservative bound instead of extrapolating an
        unvalidated distance.
        """
        if (
            isinstance(current_image_height_px, bool)
            or int(current_image_height_px) < 2
        ):
            raise ValueError("current_image_height_px must be an integer of at least 2")
        height = int(current_image_height_px)
        row = float(pixel_y)
        if not math.isfinite(row) or row < 0 or row > height - 1:
            raise ValueError("pixel_y must be a finite row inside the current image")

        row *= (self.image_height_px - 1) / (height - 1)
        first = self.points[0]
        last = self.points[-1]

        if row < first.pixel_y:
            return self._make_estimate(first.distance_m, "beyond_far_range")
        if row > last.pixel_y:
            return self._make_estimate(last.distance_m, "beyond_near_range")

        for left, right in zip(self.points, self.points[1:]):
            if left.pixel_y <= row <= right.pixel_y:
                fraction = (row - left.pixel_y) / (right.pixel_y - left.pixel_y)
                distance = left.distance_m + fraction * (
                    right.distance_m - left.distance_m
                )
                return self._make_estimate(distance, "within_calibration_range")

        # The only remaining cases are exact matches at the first/last point.
        if math.isclose(row, first.pixel_y, abs_tol=1e-9):
            return self._make_estimate(first.distance_m, "within_calibration_range")
        return self._make_estimate(last.distance_m, "within_calibration_range")

    def _make_estimate(self, rover_distance_m: float, status: str) -> DistanceEstimate:
        camera_ground_distance = None
        camera_center_distance = None
        if self.camera_to_rover_front_m is not None:
            camera_ground_distance = rover_distance_m + self.camera_to_rover_front_m
            if camera_ground_distance < 0:
                camera_ground_distance = None
            elif self.camera_height_m is not None:
                camera_center_distance = math.hypot(
                    camera_ground_distance, self.camera_height_m
                )
        return DistanceEstimate(
            distance_from_rover_front_m=float(rover_distance_m),
            status=status,
            distance_from_camera_ground_m=camera_ground_distance,
            distance_from_camera_center_m=camera_center_distance,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.VERSION,
            "distance_reference": "rover_front",
            "image_height_px": self.image_height_px,
            "camera_to_rover_front_m": self.camera_to_rover_front_m,
            "camera_height_m": self.camera_height_m,
            "points": [
                {"pixel_y": point.pixel_y, "distance_m": point.distance_m}
                for point in self.points
            ],
            "notes": (
                "Distances are measured along the flat ground plane from the rover's "
                "front edge. Runtime range uses the detection box's bottom-center row."
            ),
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> DistanceCalibration:
        version = data.get("version", cls.VERSION)
        if version != cls.VERSION:
            raise ValueError(f"Unsupported calibration version: {version}")
        points_data = data.get("points")
        if not isinstance(points_data, list):
            raise TypeError("Calibration JSON must contain a 'points' list")
        points: list[CalibrationPoint] = []
        for item in points_data:
            if (
                not isinstance(item, dict)
                or "pixel_y" not in item
                or "distance_m" not in item
            ):
                raise ValueError("Each point must contain 'pixel_y' and 'distance_m'")
            points.append(CalibrationPoint(item["pixel_y"], item["distance_m"]))

        image_height_px = data.get("image_height_px")
        if image_height_px is None:
            raise ValueError("Calibration JSON must contain 'image_height_px'")
        return cls(
            image_height_px=image_height_px,
            points=points,
            camera_to_rover_front_m=data.get("camera_to_rover_front_m"),
            camera_height_m=data.get("camera_height_m"),
        )

    @classmethod
    def load(cls, path: str | Path) -> DistanceCalibration:
        calibration_path = Path(path).expanduser()
        try:
            with calibration_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Calibration file not found: {calibration_path}"
            ) from exc
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid calibration JSON in {calibration_path}: {exc}"
            ) from exc
        if not isinstance(data, dict):
            raise TypeError("Calibration file must contain a JSON object")
        return cls.from_dict(data)

    def save(self, path: str | Path) -> None:
        calibration_path = Path(path).expanduser()
        calibration_path.parent.mkdir(parents=True, exist_ok=True)
        with calibration_path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)
            handle.write("\n")
