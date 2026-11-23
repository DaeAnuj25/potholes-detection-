import pytest

from pothole_rover.runtime import FrameRateGate, parse_source


def test_frame_rate_gate_limits_live_processing_to_target_rate():
    gate = FrameRateGate(5)

    assert gate.should_process(10.0)
    assert not gate.should_process(10.1)
    assert not gate.should_process(10.199)
    assert gate.should_process(10.2)
    assert not gate.should_process(10.39)
    assert gate.should_process(10.4)


def test_frame_rate_gate_skips_missed_slots_when_inference_is_slow():
    gate = FrameRateGate(5)

    assert gate.should_process(0.0)
    assert gate.should_process(0.65)
    assert not gate.should_process(0.79)
    assert gate.should_process(0.8)


def test_source_parser_treats_digits_as_camera_indices():
    assert parse_source("0") == 0
    assert parse_source(" 2 ") == 2
    assert parse_source("road-test.mp4") == "road-test.mp4"
    assert parse_source("rtsp://camera/stream") == "rtsp://camera/stream"


def test_frame_rate_gate_requires_positive_finite_rate():
    with pytest.raises(ValueError):
        FrameRateGate(0)
    with pytest.raises(ValueError):
        FrameRateGate(float("inf"))
