"""Small standard-library HTTP dashboard backed by the ROS detector node."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import rclpy
from rclpy.executors import SingleThreadedExecutor

from .ros_node import PotholeDetectorNode


def _read_dashboard_html() -> bytes:
    try:
        from ament_index_python.packages import get_package_share_directory

        html_path = (
            Path(get_package_share_directory("pothole_rover")) / "web" / "index.html"
        )
    except (ImportError, LookupError):
        html_path = Path(__file__).resolve().parent / "web" / "index.html"
    if not html_path.is_file():
        raise FileNotFoundError(f"Dashboard page not found: {html_path}")
    return html_path.read_bytes()


def _handler_for(node: PotholeDetectorNode) -> type[BaseHTTPRequestHandler]:
    html = _read_dashboard_html()

    class DashboardHandler(BaseHTTPRequestHandler):
        server_version = "PotholeRoverDashboard/1.0"

        def log_message(self, format: str, *args: Any) -> None:
            # Keep the ROS console focused on detection and configuration messages.
            del format, args

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            if body:
                self.wfile.write(body)

        def _send_json(self, status: int, value: dict[str, Any]) -> None:
            body = json.dumps(value, ensure_ascii=True, separators=(",", ":")).encode(
                "utf-8"
            )
            self._send(status, body, "application/json; charset=utf-8")

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path == "/" or path == "/index.html":
                self._send(200, html, "text/html; charset=utf-8")
            elif path == "/api/state":
                self._send_json(200, node.dashboard_state())
            elif path == "/api/topics":
                self._send_json(200, node.topic_options())
            elif path == "/api/preview.jpg":
                preview = node.preview_jpeg()
                if preview is None:
                    self._send(204, b"", "image/jpeg")
                else:
                    self._send(200, preview, "image/jpeg")
            else:
                self._send_json(404, {"error": "not found"})

        def do_POST(self) -> None:
            if urlparse(self.path).path != "/api/configure":
                self._send_json(404, {"error": "not found"})
                return
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
                if content_length <= 0 or content_length > 16_384:
                    raise ValueError("Request body must be between 1 and 16384 bytes")
                payload = json.loads(self.rfile.read(content_length))
                if not isinstance(payload, dict):
                    raise TypeError("Request body must be a JSON object")
                request_id = node.request_configuration(payload)
            except (
                json.JSONDecodeError,
                UnicodeDecodeError,
                TypeError,
                ValueError,
            ) as exc:
                self._send_json(400, {"error": str(exc)})
                return
            self._send_json(
                202,
                {
                    "request_id": request_id,
                    "message": "Configuration queued on the ROS executor",
                },
            )

    return DashboardHandler


class _DashboardHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def run_dashboard(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node: PotholeDetectorNode | None = None
    executor: SingleThreadedExecutor | None = None
    server: _DashboardHTTPServer | None = None
    spin_thread: threading.Thread | None = None
    try:
        node = PotholeDetectorNode(node_name="pothole_dashboard")
        host = str(node.get_parameter("dashboard_host").value)
        port = int(node.get_parameter("dashboard_port").value)
        if not 1 <= port <= 65535:
            raise ValueError("dashboard_port must be between 1 and 65535")

        executor = SingleThreadedExecutor()
        executor.add_node(node)
        spin_thread = threading.Thread(
            target=executor.spin, name="pothole-ros-executor", daemon=True
        )
        spin_thread.start()

        server = _DashboardHTTPServer((host, port), _handler_for(node))
        node.get_logger().info(f"Dashboard ready at http://{host}:{port}")
        node.get_logger().info(
            "Select an Image/CompressedImage input topic and a String JSON output topic in the page."
        )
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        if server is not None:
            server.server_close()
        if executor is not None:
            executor.shutdown()
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        if spin_thread is not None and spin_thread.is_alive():
            spin_thread.join(timeout=2.0)


def main(args: list[str] | None = None) -> None:
    run_dashboard(args=args)
