import json
from argparse import Namespace

import cv2
import numpy as np

from pothole_rover import runtime
from pothole_rover.detector import Detection, InferenceResult, grayscale_bgr


def test_still_image_pipeline_writes_annotation_and_jsonl(tmp_path, monkeypatch):
    source = tmp_path / "input.png"
    annotated = tmp_path / "annotated.png"
    telemetry = tmp_path / "detections.jsonl"
    cv2.imwrite(str(source), np.full((40, 60, 3), 120, dtype=np.uint8))

    class StubDetector:
        def __init__(self, **kwargs):
            self.weights_path = kwargs["weights"]

        def predict(self, frame):
            return InferenceResult(
                gray_bgr=grayscale_bgr(frame),
                detections=(Detection(5, 6, 30, 31, 0.88, 0, "pothole"),),
                inference_ms=12.0,
            )

    monkeypatch.setattr(runtime, "UltralyticsPotholeDetector", StubDetector)
    args = Namespace(
        source=str(source),
        weights=tmp_path / "unused.pt",
        image_size=640,
        confidence=0.25,
        iou=0.45,
        device=None,
        calibration=None,
        target_fps=5.0,
        width=None,
        height=None,
        jsonl=str(telemetry),
        headless=True,
        save_video=None,
        save_image=annotated,
    )

    assert runtime.run(args) == 0

    event = json.loads(telemetry.read_text().strip())
    assert event["frame_index"] == 0
    assert event["image_width_px"] == 60
    detection = event["detections"][0]
    assert detection["class_name"] == "pothole"
    assert detection["confidence"] == 0.88
    assert detection["distance_from_rover_front_m"] is None
    assert detection["distance_status"] == "uncalibrated"
    assert annotated.is_file()
