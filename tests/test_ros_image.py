import cv2
import numpy as np
import pytest

from pothole_rover.ros_image import decode_compressed_image, decode_raw_image


def test_decode_bgr8_honors_padded_ros_row_step():
    # One padding byte follows each 1-pixel row.
    data = bytes([1, 2, 3, 99, 4, 5, 6, 99])

    frame = decode_raw_image(
        height=2,
        width=1,
        encoding="bgr8",
        step=4,
        data=data,
    )

    assert frame.tolist() == [[[1, 2, 3]], [[4, 5, 6]]]


def test_decode_rgb8_converts_to_opencv_bgr_order():
    frame = decode_raw_image(1, 1, "rgb8", 3, bytes([255, 0, 0]))

    assert frame.tolist() == [[[0, 0, 255]]]


def test_decode_mono8_returns_a_2d_frame():
    frame = decode_raw_image(1, 3, "mono8", 3, bytes([0, 127, 255]))

    assert frame.shape == (1, 3)
    assert frame.tolist() == [[0, 127, 255]]


def test_decode_compressed_image_payload():
    source = np.array([[[10, 20, 30], [80, 90, 100]]], dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", source)
    assert ok

    decoded = decode_compressed_image(encoded.tobytes())

    assert np.array_equal(decoded, source)


def test_rejects_unknown_encoding_or_short_payload():
    with pytest.raises(ValueError, match="Unsupported Image encoding"):
        decode_raw_image(1, 1, "yuv422", 2, b"\x00\x00")
    with pytest.raises(ValueError, match="shorter than"):
        decode_raw_image(2, 2, "bgr8", 6, b"\x00\x00")
