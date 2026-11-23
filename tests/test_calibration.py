import json
import math

import pytest

from pothole_rover.calibration import CalibrationPoint, DistanceCalibration


def make_calibration(**kwargs):
    return DistanceCalibration(
        image_height_px=101,
        points=[
            CalibrationPoint(pixel_y=20, distance_m=4.0),
            CalibrationPoint(pixel_y=40, distance_m=3.0),
            CalibrationPoint(pixel_y=60, distance_m=2.0),
            CalibrationPoint(pixel_y=80, distance_m=1.0),
        ],
        **kwargs,
    )


def test_interpolates_pixel_row_to_rover_front_distance():
    calibration = make_calibration()

    estimate = calibration.estimate(pixel_y=30, current_image_height_px=101)

    assert estimate.distance_from_rover_front_m == pytest.approx(3.5)
    assert estimate.status == "within_calibration_range"
    assert estimate.label == "~ 3.50 m from rover front"


def test_scales_rows_when_capture_height_changes():
    calibration = make_calibration()

    # y=60 in a 201-row image maps to y=30 in the 101-row calibration image.
    estimate = calibration.estimate(pixel_y=60, current_image_height_px=201)

    assert estimate.distance_from_rover_front_m == pytest.approx(3.5)


def test_outside_range_returns_a_bound_instead_of_extrapolating():
    calibration = make_calibration()

    far = calibration.estimate(pixel_y=10, current_image_height_px=101)
    near = calibration.estimate(pixel_y=90, current_image_height_px=101)

    assert far.distance_from_rover_front_m == pytest.approx(4.0)
    assert far.status == "beyond_far_range"
    assert far.label == ">= 4.00 m from rover front"
    assert near.distance_from_rover_front_m == pytest.approx(1.0)
    assert near.status == "beyond_near_range"
    assert near.label == "<= 1.00 m from rover front"


def test_can_report_camera_ground_and_slant_distances_when_offsets_are_known():
    calibration = make_calibration(
        camera_to_rover_front_m=0.2,
        camera_height_m=0.5,
    )

    estimate = calibration.estimate(pixel_y=60, current_image_height_px=101)

    assert estimate.distance_from_rover_front_m == pytest.approx(2.0)
    assert estimate.distance_from_camera_ground_m == pytest.approx(2.2)
    assert estimate.distance_from_camera_center_m == pytest.approx(math.hypot(2.2, 0.5))


def test_rejects_non_monotonic_or_duplicate_calibration_rows():
    with pytest.raises(ValueError, match="Distances must decrease"):
        DistanceCalibration(
            image_height_px=100,
            points=[CalibrationPoint(20, 1), CalibrationPoint(40, 2)],
        )

    with pytest.raises(ValueError, match="distinct image row"):
        DistanceCalibration(
            image_height_px=100,
            points=[CalibrationPoint(20, 2), CalibrationPoint(20, 1)],
        )


def test_save_and_load_round_trip(tmp_path):
    calibration = make_calibration(
        camera_to_rover_front_m=0.15,
        camera_height_m=0.4,
    )
    target = tmp_path / "calibration.json"

    calibration.save(target)
    loaded = DistanceCalibration.load(target)

    assert loaded.to_dict() == calibration.to_dict()
    assert json.loads(target.read_text())["distance_reference"] == "rover_front"
