from pothole_rover.calibration import CalibrationPoint, DistanceCalibration
from pothole_rover.detector import Detection, InferenceResult
from pothole_rover.runtime import _make_event


def test_telemetry_includes_rover_and_optional_camera_ranges():
    calibration = DistanceCalibration(
        image_height_px=101,
        points=[CalibrationPoint(20, 4.0), CalibrationPoint(80, 1.0)],
        camera_to_rover_front_m=0.2,
        camera_height_m=0.5,
    )
    inference = InferenceResult(
        gray_bgr=None,
        detections=(Detection(10, 20, 30, 50, 0.9, 0, "pothole"),),
        inference_ms=25.0,
    )

    event = _make_event(
        source_text="0",
        frame_index=12,
        source_time_ms=None,
        frame_width=101,
        frame_height=101,
        inference=inference,
        calibration=calibration,
    )

    detection = event["detections"][0]
    assert detection["distance_from_rover_front_m"] == 2.5
    assert detection["distance_from_camera_ground_m"] == 2.7
    assert detection["distance_status"] == "within_calibration_range"
    assert detection["distance_from_camera_center_m"] is not None


def test_uncalibrated_telemetry_does_not_invent_a_range():
    inference = InferenceResult(
        gray_bgr=None,
        detections=(Detection(10, 20, 30, 50, 0.9, 0, "pothole"),),
        inference_ms=25.0,
    )

    event = _make_event(
        source_text="0",
        frame_index=0,
        source_time_ms=None,
        frame_width=100,
        frame_height=100,
        inference=inference,
        calibration=None,
    )

    detection = event["detections"][0]
    assert detection["distance_from_rover_front_m"] is None
    assert detection["distance_status"] == "uncalibrated"
