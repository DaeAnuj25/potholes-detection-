"""Conversion helpers for the ROS 2 sensor image encodings used by the node."""

from __future__ import annotations

import cv2
import numpy as np


def decode_raw_image(
    height: int,
    width: int,
    encoding: str,
    step: int,
    data: bytes | bytearray | memoryview,
) -> np.ndarray:
    """Decode common 8-bit ROS Image encodings, honoring padded row strides."""
    height = int(height)
    width = int(width)
    encoding = str(encoding).lower()
    channels = {
        "bgr8": 3,
        "rgb8": 3,
        "bgra8": 4,
        "rgba8": 4,
        "mono8": 1,
        "8uc1": 1,
    }.get(encoding)
    if channels is None:
        raise ValueError(
            f"Unsupported Image encoding {encoding!r}; use bgr8, rgb8, bgra8, rgba8 or mono8"
        )
    row_step = int(step) or width * channels
    if height <= 0 or width <= 0 or row_step < width * channels:
        raise ValueError("Image dimensions or row step are invalid")

    raw = np.frombuffer(data, dtype=np.uint8)
    required = height * row_step
    if raw.size < required:
        raise ValueError("Image data buffer is shorter than height * step")
    rows = raw[:required].reshape(height, row_step)
    pixels = rows[:, : width * channels]
    if channels == 1:
        return pixels.reshape(height, width)

    pixels = pixels.reshape(height, width, channels)
    if encoding == "rgb8":
        return cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
    if encoding == "rgba8":
        return cv2.cvtColor(pixels, cv2.COLOR_RGBA2BGR)
    if encoding == "bgra8":
        return cv2.cvtColor(pixels, cv2.COLOR_BGRA2BGR)
    return np.ascontiguousarray(pixels)


def decode_compressed_image(data: bytes | bytearray | memoryview) -> np.ndarray:
    """Decode a ROS CompressedImage payload into a BGR frame."""
    encoded = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Could not decode CompressedImage payload")
    return frame
