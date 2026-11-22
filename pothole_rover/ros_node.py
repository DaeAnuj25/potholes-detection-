"""ROS 2 image subscriber and pothole detection publisher.

The output is a JSON document in ``std_msgs/msg/String`` so consumers do not
need this package's own message definitions. Input images may be raw
``sensor_msgs/msg/Image`` or ``CompressedImage``.
"""

from __future__ import annotations

import json
import math
import queue
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import String

from .calibration import DistanceCalibration
from .detector import UltralyticsPotholeDetector
from .ros_image import decode_compressed_image, decode_raw_image
from .runtime import FrameRateGate, _draw_overlay, _make_event, default_weights_path

_IMAGE_TYPE_TO_ROS = {
    "image": "sensor_msgs/msg/Image",
    "compressed": "sensor_msgs/msg/CompressedImage",
}
_ROS_TYPE_TO_IMAGE = {value: key for key, value in _IMAGE_TYPE_TO_ROS.items()}
_TOPIC_PATTERN = re.compile(r"/(?:[A-Za-z_][A-Za-z0-9_]*)(?:/[A-Za-z_][A-Za-z0-9_]*)*")


def _default_ros_weights() -> str:
    """Find the model copied into the ament share directory, or use the checkout."""
    try:
        from ament_index_python.packages import get_package_share_directory

        share = Path(get_package_share_directory("pothole_rover"))
        for filename in ("best.pt", "new_model.pt"):
            candidate = share / filename
            if candidate.is_file():
                return str(candidate)
    except (ImportError, LookupError):
        pass
    return str(default_weights_path())


@dataclass(frozen=True)
class TopicConfiguration:
    image_topic: str
    image_type: str
    detections_topic: str
    target_fps: float
    confidence: float
    calibration_file: str


class PotholeDetectorNode(Node):
    """Subscribe to a ROS image topic and publish detections as JSON strings."""

    def __init__(self, node_name: str = "pothole_detector") -> None:
        super().__init__(node_name)
        self.declare_parameter("weights", _default_ros_weights())
        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("image_type", "image")
        self.declare_parameter("detections_topic", "/pothole/detections")
        self.declare_parameter("annotated_topic", "/pothole/annotated")
        self.declare_parameter("target_fps", 5.0)
        self.declare_parameter("confidence", 0.25)
        self.declare_parameter("iou", 0.45)
        self.declare_parameter("image_size", 640)
        self.declare_parameter("device", "")
        self.declare_parameter("calibration_file", "")
        self.declare_parameter("dashboard_host", "127.0.0.1")
        self.declare_parameter("dashboard_port", 8080)

        self._lock = threading.RLock()
        self._configuration_queue: queue.Queue[tuple[int, TopicConfiguration]] = (
            queue.Queue()
        )
        self._configuration_id = 0
        self._image_subscription: Any = None
        self._detections_publisher: Any = None
        self._annotated_publisher: Any = None
        self._active_image_topic = ""
        self._active_image_type = "image"
        self._active_detections_topic = ""
        self._annotated_topic = str(self.get_parameter("annotated_topic").value).strip()
        self._calibration: DistanceCalibration | None = None
        self._calibration_file = ""
        self._target_fps = float(self.get_parameter("target_fps").value)
        self._rate_gate = FrameRateGate(self._target_fps)
        self._frame_index = 0
        self._processing_times: list[float] = []
        self._last_image_received_s: float | None = None
        self._last_preview_jpeg: bytes | None = None
        self._available_image_topics: list[dict[str, str]] = []
        self._available_string_topics: list[str] = []
        self._input_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        self._output_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        self._preview_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        if self._annotated_topic:
            try:
                self._validate_topic_type(
                    self._annotated_topic, "sensor_msgs/msg/Image"
                )
                self._annotated_publisher = self.create_publisher(
                    Image, self._annotated_topic, self._preview_qos
                )
            except Exception as exc:  # noqa: BLE001 - optional ROS topic failure should not abort node startup
                self.get_logger().warning(f"Annotated image publisher disabled: {exc}")
                self._annotated_topic = ""

        weights = str(self.get_parameter("weights").value)
        self.detector = UltralyticsPotholeDetector(
            weights=weights,
            image_size=int(self.get_parameter("image_size").value),
            confidence=float(self.get_parameter("confidence").value),
            iou=float(self.get_parameter("iou").value),
            device=str(self.get_parameter("device").value).strip() or None,
        )
        self._state: dict[str, Any] = {
            "node_name": node_name,
            "status": "starting",
            "weights": str(self.detector.weights_path),
            "image_topic": "",
            "image_type": "",
            "detections_topic": "",
            "annotated_topic": self._annotated_topic,
            "target_fps": self._target_fps,
            "confidence": self.detector.confidence,
            "calibration_file": "",
            "actual_fps": 0.0,
            "inference_ms": None,
            "processed_frames": 0,
            "input_frames": 0,
            "dropped_frames": 0,
            "detection_count": 0,
            "last_event": None,
            "last_error": None,
            "configuration": {"status": "starting", "message": "Loading model"},
        }

        initial = TopicConfiguration(
            image_topic=str(self.get_parameter("image_topic").value),
            image_type=str(self.get_parameter("image_type").value),
            detections_topic=str(self.get_parameter("detections_topic").value),
            target_fps=self._target_fps,
            confidence=float(self.get_parameter("confidence").value),
            calibration_file=str(self.get_parameter("calibration_file").value),
        )
        self._apply_configuration(initial, request_id=None)
        self._configuration_timer = self.create_timer(
            0.1, self._apply_queued_configuration
        )
        self._topic_timer = self.create_timer(1.0, self._refresh_topics)
        self._refresh_topics()

    @staticmethod
    def _validate_topic(topic: str, label: str) -> str:
        value = topic.strip()
        if not _TOPIC_PATTERN.fullmatch(value):
            raise ValueError(
                f"{label} must be an absolute ROS topic name, for example /camera/image_raw"
            )
        return value

    def request_configuration(self, payload: dict[str, Any]) -> int:
        """Queue dashboard settings; topic changes happen on the ROS executor."""
        image_topic = self._validate_topic(
            str(payload.get("image_topic", "")), "Image topic"
        )
        detections_topic = self._validate_topic(
            str(payload.get("detections_topic", "")), "Detections topic"
        )
        image_type = str(payload.get("image_type", "image"))
        if image_type not in _IMAGE_TYPE_TO_ROS:
            raise ValueError("image_type must be 'image' or 'compressed'")

        try:
            target_fps = float(payload.get("target_fps", self._target_fps))
            confidence = float(payload.get("confidence", self.detector.confidence))
        except (TypeError, ValueError) as exc:
            raise ValueError("FPS and confidence must be numbers") from exc
        if not math.isfinite(target_fps) or target_fps <= 0 or target_fps > 60:
            raise ValueError("target_fps must be greater than 0 and no more than 60")
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")

        calibration_file = str(payload.get("calibration_file", "")).strip()
        config = TopicConfiguration(
            image_topic=image_topic,
            image_type=image_type,
            detections_topic=detections_topic,
            target_fps=target_fps,
            confidence=confidence,
            calibration_file=calibration_file,
        )
        with self._lock:
            self._configuration_id += 1
            request_id = self._configuration_id
            self._state["configuration"] = {
                "id": request_id,
                "status": "pending",
                "message": "Applying topic and processing settings",
            }
            self._state["status"] = "reconfiguring"
        self._configuration_queue.put((request_id, config))
        return request_id

    def _apply_queued_configuration(self) -> None:
        request: tuple[int, TopicConfiguration] | None = None
        while True:
            try:
                request = self._configuration_queue.get_nowait()
            except queue.Empty:
                break
        if request is None:
            return
        request_id, config = request
        try:
            self._apply_configuration(config, request_id=request_id)
        except Exception as exc:  # noqa: BLE001 - report all ROS/config errors to the dashboard
            message = str(exc)
            with self._lock:
                self._state["status"] = "configuration_error"
                self._state["last_error"] = message
                self._state["configuration"] = {
                    "id": request_id,
                    "status": "error",
                    "message": message,
                }
            self.get_logger().error(
                f"Could not apply dashboard configuration: {message}"
            )

    def _validate_topic_type(self, topic: str, ros_type: str) -> None:
        advertised = dict(self.get_topic_names_and_types()).get(topic, [])
        if advertised and ros_type not in advertised:
            raise ValueError(
                f"{topic} currently advertises {', '.join(advertised)}, not {ros_type}"
            )

    def _apply_configuration(
        self, config: TopicConfiguration, request_id: int | None
    ) -> None:
        self._validate_topic(config.image_topic, "Image topic")
        self._validate_topic(config.detections_topic, "Detections topic")
        if self._annotated_topic and config.image_topic == self._annotated_topic:
            raise ValueError(
                "Input topic cannot be the annotated output topic (feedback loop)"
            )
        if config.image_type not in _IMAGE_TYPE_TO_ROS:
            raise ValueError("image_type must be 'image' or 'compressed'")
        self._validate_topic_type(
            config.image_topic, _IMAGE_TYPE_TO_ROS[config.image_type]
        )
        self._validate_topic_type(config.detections_topic, "std_msgs/msg/String")

        calibration = (
            DistanceCalibration.load(config.calibration_file)
            if config.calibration_file
            else None
        )
        input_changed = (
            config.image_topic != self._active_image_topic
            or config.image_type != self._active_image_type
            or self._image_subscription is None
        )
        output_changed = (
            config.detections_topic != self._active_detections_topic
            or self._detections_publisher is None
        )
        new_subscription = self._image_subscription
        new_publisher = self._detections_publisher
        created_subscription = None
        created_publisher = None

        try:
            if input_changed:
                message_type = (
                    Image if config.image_type == "image" else CompressedImage
                )
                created_subscription = self.create_subscription(
                    message_type,
                    config.image_topic,
                    self._on_image,
                    self._input_qos,
                )
                new_subscription = created_subscription
            if output_changed:
                created_publisher = self.create_publisher(
                    String, config.detections_topic, self._output_qos
                )
                new_publisher = created_publisher
        except Exception:
            if created_subscription is not None:
                self.destroy_subscription(created_subscription)
            if created_publisher is not None:
                self.destroy_publisher(created_publisher)
            raise

        old_subscription = self._image_subscription
        old_publisher = self._detections_publisher
        self._image_subscription = new_subscription
        self._detections_publisher = new_publisher
        self._active_image_topic = config.image_topic
        self._active_image_type = config.image_type
        self._active_detections_topic = config.detections_topic
        self._calibration = calibration
        self._calibration_file = config.calibration_file
        self._target_fps = config.target_fps
        self._rate_gate = FrameRateGate(config.target_fps)
        self._processing_times.clear()
        self.detector.confidence = config.confidence

        if input_changed and old_subscription is not None:
            self.destroy_subscription(old_subscription)
        if output_changed and old_publisher is not None:
            self.destroy_publisher(old_publisher)

        self.set_parameters(
            [
                Parameter("image_topic", value=config.image_topic),
                Parameter("image_type", value=config.image_type),
                Parameter("detections_topic", value=config.detections_topic),
                Parameter("target_fps", value=config.target_fps),
                Parameter("confidence", value=config.confidence),
                Parameter("calibration_file", value=config.calibration_file),
            ]
        )
        with self._lock:
            if input_changed:
                self._last_image_received_s = None
                self._last_preview_jpeg = None
                self._state["last_event"] = None
                self._state["detection_count"] = 0
                self._state["inference_ms"] = None
                self._state["actual_fps"] = 0.0
            self._state.update(
                {
                    "status": "waiting_for_image",
                    "image_topic": config.image_topic,
                    "image_type": config.image_type,
                    "detections_topic": config.detections_topic,
                    "target_fps": config.target_fps,
                    "confidence": config.confidence,
                    "calibration_file": config.calibration_file,
                    "last_error": None,
                    "configuration": {
                        "id": request_id,
                        "status": "applied",
                        "message": "Settings applied; waiting for an image",
                    },
                }
            )
        self.get_logger().info(
            f"Listening to {config.image_topic} ({config.image_type}); "
            f"publishing JSON detections on {config.detections_topic}"
        )

    def _refresh_topics(self) -> None:
        try:
            discovered = self.get_topic_names_and_types()
        except Exception as exc:  # noqa: BLE001 - ROS graph errors are transient
            self.get_logger().warning(f"Could not refresh ROS topics: {exc}")
            return
        image_topics: list[dict[str, str]] = []
        string_topics: list[str] = []
        for topic, types in discovered:
            for ros_type, image_type in _ROS_TYPE_TO_IMAGE.items():
                if ros_type in types:
                    image_topics.append(
                        {"name": topic, "image_type": image_type, "ros_type": ros_type}
                    )
            if "std_msgs/msg/String" in types:
                string_topics.append(topic)
        image_topics.sort(key=lambda item: (item["name"], item["image_type"]))
        with self._lock:
            self._available_image_topics = image_topics
            self._available_string_topics = sorted(set(string_topics))

    def topic_options(self) -> dict[str, Any]:
        with self._lock:
            return {
                "image_topics": list(self._available_image_topics),
                "string_topics": list(self._available_string_topics),
            }

    def dashboard_state(self) -> dict[str, Any]:
        with self._lock:
            snapshot = dict(self._state)
            snapshot["preview_available"] = self._last_preview_jpeg is not None
            last_received = self._last_image_received_s
        snapshot["seconds_since_image"] = (
            None
            if last_received is None
            else max(0.0, time.monotonic() - last_received)
        )
        if (
            snapshot["seconds_since_image"] is not None
            and snapshot["seconds_since_image"] > 2.0
            and snapshot["status"] in {"waiting_for_image", "running"}
        ):
            snapshot["status"] = "stale_image"
        return snapshot

    def preview_jpeg(self) -> bytes | None:
        with self._lock:
            return self._last_preview_jpeg

    @staticmethod
    def _header_data(message: Any) -> dict[str, Any]:
        header = getattr(message, "header", None)
        if header is None:
            return {"frame_id": "", "stamp_sec": 0, "stamp_nanosec": 0}
        return {
            "frame_id": header.frame_id,
            "stamp_sec": int(header.stamp.sec),
            "stamp_nanosec": int(header.stamp.nanosec),
        }

    def _decode_image_message(self, message: Any) -> np.ndarray:
        if isinstance(message, CompressedImage):
            return decode_compressed_image(message.data)
        return decode_raw_image(
            height=message.height,
            width=message.width,
            encoding=message.encoding,
            step=message.step,
            data=message.data,
        )

    @staticmethod
    def _to_ros_image(frame: np.ndarray, header: Any) -> Image:
        contiguous = np.ascontiguousarray(frame)
        height, width = contiguous.shape[:2]
        message = Image()
        message.header = header
        message.height = height
        message.width = width
        message.encoding = "bgr8"
        message.is_bigendian = 0
        message.step = width * 3
        message.data = contiguous.tobytes()
        return message

    def _current_processing_fps(self, now_s: float) -> float:
        self._processing_times.append(now_s)
        cutoff = now_s - 2.0
        while self._processing_times and self._processing_times[0] < cutoff:
            self._processing_times.pop(0)
        if len(self._processing_times) < 2:
            return 0.0
        duration = self._processing_times[-1] - self._processing_times[0]
        return (len(self._processing_times) - 1) / duration if duration > 0 else 0.0

    def _on_image(self, message: Any) -> None:
        received_s = time.monotonic()
        with self._lock:
            self._frame_index += 1
            frame_index = self._frame_index
            self._last_image_received_s = received_s
            self._state["input_frames"] += 1
        if not self._rate_gate.should_process(received_s):
            with self._lock:
                self._state["dropped_frames"] += 1
            return

        try:
            frame = self._decode_image_message(message)
            inference = self.detector.predict(frame)
            height, width = inference.gray_bgr.shape[:2]
            actual_fps = self._current_processing_fps(time.monotonic())
            annotated = _draw_overlay(
                cv2=cv2,
                inference=inference,
                calibration=self._calibration,
                target_fps=self._target_fps,
                actual_fps=actual_fps,
                is_video_file=False,
            )
            event = _make_event(
                source_text=self._active_image_topic,
                frame_index=frame_index,
                source_time_ms=None,
                frame_width=width,
                frame_height=height,
                inference=inference,
                calibration=self._calibration,
            )
            event["header"] = self._header_data(message)
            event["target_fps"] = self._target_fps
            event["actual_fps"] = actual_fps
            for detection_index, detection_record in enumerate(event["detections"]):
                detection_record["detection_index"] = detection_index

            output_message = String()
            output_message.data = json.dumps(
                event, ensure_ascii=True, separators=(",", ":")
            )
            if self._detections_publisher is not None:
                self._detections_publisher.publish(output_message)
            if self._annotated_publisher is not None:
                self._annotated_publisher.publish(
                    self._to_ros_image(annotated, getattr(message, "header", None))
                )

            ok, encoded = cv2.imencode(
                ".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 82]
            )
            preview = encoded.tobytes() if ok else None
            with self._lock:
                self._state.update(
                    {
                        "status": "running",
                        "actual_fps": actual_fps,
                        "inference_ms": inference.inference_ms,
                        "processed_frames": self._state["processed_frames"] + 1,
                        "detection_count": len(inference.detections),
                        "last_event": event,
                        "last_error": None,
                    }
                )
                if preview is not None:
                    self._last_preview_jpeg = preview
        except Exception as exc:  # noqa: BLE001 - keep the ROS executor alive after bad frames
            message_text = str(exc)
            with self._lock:
                self._state["status"] = "error"
                self._state["last_error"] = message_text
            self.get_logger().error(f"Image processing failed: {message_text}")


def detector_main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = PotholeDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def dashboard_main(args: list[str] | None = None) -> None:
    """Entry point imported lazily to keep the detector node independent of HTTP."""
    from .dashboard import run_dashboard

    run_dashboard(args=args)
