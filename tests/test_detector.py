import sys
from types import SimpleNamespace

import numpy as np

from pothole_rover.detector import UltralyticsPotholeDetector, grayscale_bgr


def test_grayscale_frame_is_replicated_across_three_channels():
    frame = np.array([[[0, 0, 255], [255, 0, 0]]], dtype=np.uint8)

    prepared = grayscale_bgr(frame)

    assert prepared.shape == (1, 2, 3)
    assert np.array_equal(prepared[:, :, 0], prepared[:, :, 1])
    assert np.array_equal(prepared[:, :, 1], prepared[:, :, 2])


def test_already_grayscale_frame_keeps_its_intensity():
    frame = np.array([[0, 127, 255]], dtype=np.uint8)

    prepared = grayscale_bgr(frame)

    assert np.array_equal(prepared[0, :, 0], frame[0])
    assert np.array_equal(prepared[0, :, 1], frame[0])
    assert np.array_equal(prepared[0, :, 2], frame[0])


class FakeTensor:
    def __init__(self, values):
        self.values = values

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.values


class FakeBoxes:
    def __init__(self):
        self.xyxy = FakeTensor([[2, 3, 18, 19]])
        self.conf = FakeTensor([0.92])
        self.cls = FakeTensor([0])

    def __len__(self):
        return 1


def test_detector_passes_grayscale_to_model_and_extracts_boxes(tmp_path, monkeypatch):
    captured = {}

    class FakeYOLO:
        def __init__(self, path):
            self.names = {0: "pothole"}
            captured["weights"] = path

        def predict(self, **options):
            captured["input"] = options["source"]
            return [SimpleNamespace(boxes=FakeBoxes(), names={0: "pothole"})]

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=FakeYOLO))
    weights = tmp_path / "model.pt"
    weights.write_bytes(b"test weights")
    original = np.array([[[0, 0, 255]] * 20] * 20, dtype=np.uint8)

    detector = UltralyticsPotholeDetector(weights=weights)
    result = detector.predict(original)

    model_input = captured["input"]
    assert model_input.shape == (20, 20, 3)
    assert np.array_equal(model_input[:, :, 0], model_input[:, :, 1])
    assert np.array_equal(model_input[:, :, 1], model_input[:, :, 2])
    assert len(result.detections) == 1
    assert result.detections[0].class_name == "pothole"
    assert result.detections[0].confidence == 0.92
    assert result.detections[0].closest_edge_y == 19
