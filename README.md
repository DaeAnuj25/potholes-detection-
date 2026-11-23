# Pothole Rover · ROS 2

This repo includes a ROS 2 pothole-perception node and a browser dashboard. The
node subscribes to camera images, converts each processed frame to grayscale,
runs the supplied Ultralytics model at a default target of 5 FPS, estimates
pothole range when a calibration is supplied, and publishes JSON detection
records. It can also publish an annotated image for other ROS tools to view.

## Model in this checkout

The tracked checkpoint is named **`new_model.pt`** (there is no `best.pt` in the
current checkout). It is used by default; if a `best.pt` is copied into the
package share directory, that file is preferred. The checkpoint metadata
identifies an Ultralytics YOLO11n detector with one class, `pothole`.

Frames are converted to grayscale and then the grayscale values are repeated in
three channels. This keeps the input shape expected by YOLO while ensuring the
model sees grayscale rather than color.

## ROS 2 installation

The package is an `ament_python` package. The commands below assume ROS 2 is
already installed and the repo is at `~/ros2_ws/src/pothole_rover`.

```bash
source /opt/ros/$ROS_DISTRO/setup.bash

# Use a venv that can still see ROS's system Python packages (rclpy, colcon, etc.)
python3 -m venv --system-site-packages ~/venvs/pothole-rover
source ~/venvs/pothole-rover/bin/activate
python -m pip install --upgrade pip
python -m pip install -r ~/ros2_ws/src/pothole_rover/requirements.txt

cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y
python -m colcon build --symlink-install
source install/setup.bash
```

Keep the ROS environment and venv active when building and running. Installing `requirements.txt` brings in Ultralytics, PyTorch, OpenCV and NumPy;
`colcon build` installs the ROS package and copies the model checkpoint into its
share directory. Ultralytics uses a compatible GPU automatically when the
installed PyTorch build supports it. The input and output ROS interfaces use
standard message packages (`sensor_msgs` and `std_msgs`); no custom message
package is needed.

## Run the ROS 2 detector

```bash
ros2 launch pothole_rover pothole_detector.launch.py \
  image_topic:=/camera/image_raw \
  image_type:=image \
  detections_topic:=/pothole/detections \
  target_fps:=5.0
```

Use `weights:=/absolute/path/to/model.pt` or `device:=cpu` / `device:=0` to
override the default model or inference device. The ROS launch defaults use
`new_model.pt` from the installed package share directory.

The input may be either:

- `sensor_msgs/msg/Image` with `bgr8`, `rgb8`, `bgra8`, `rgba8`, or `mono8`
  encoding; or
- `sensor_msgs/msg/CompressedImage` (for example, a camera's
  `/camera/image_raw/compressed` topic).

For a compressed topic, set `image_type:=compressed`. The node uses depth-1,
best-effort input QoS so it favors fresh camera frames instead of building a
large queue. For a live stream it processes at most the configured target rate
(default **5 inferences per second**) and drops excess frames. Achieved rate
still depends on the rover's CPU/GPU; if one inference takes longer than 200 ms,
5 FPS cannot be sustained. The node reports inference time and frame counts in
its logs/dashboard.

Inspect the detection output with:

```bash
ros2 topic echo /pothole/detections
ros2 topic info /pothole/annotated
```

### Detection message contract

The detections topic is `std_msgs/msg/String`. Each processed frame publishes one
JSON object, including frames with no detections. A record contains the source
topic, frame index, source image size, inference time and an array of potholes.
Each pothole includes its class, confidence, pixel box, nearest-edge pixel row,
range fields and calibration status. Before calibration, range values are
`null`; after calibration, an in-range pothole resembles:

```json
{
  "class_name": "pothole",
  "confidence": 0.91,
  "box_xyxy_px": [210, 180, 392, 335],
  "closest_edge_pixel_y": 335,
  "distance_from_rover_front_m": 1.27,
  "distance_from_camera_ground_m": 1.45,
  "distance_from_camera_center_m": 1.51,
  "distance_status": "within_calibration_range"
}
```

The node also publishes the annotated grayscale frame as
`sensor_msgs/msg/Image` on `/pothole/annotated` by default. Set
`annotated_topic:=` (empty) to disable that publisher.

## Linux dashboard

Run the combined detector and dashboard node (this launch already runs
inference, so do not also launch `pothole_detector` unless you want a second
model instance):

```bash
ros2 launch pothole_rover pothole_dashboard.launch.py
```

Open **http://127.0.0.1:8080** on the Linux machine. The page discovers image
and String topics from the ROS graph. From the dashboard you can:

- select a discovered image topic or type an absolute topic name;
- choose raw `Image` or `CompressedImage` input;
- choose an existing String topic or enter a new detections output topic;
- change target FPS and confidence without restarting the node;
- enter a path to a calibration JSON when it is ready;
- view the grayscale annotated preview, live pothole list, confidence, range,
  inference time, processed FPS, received/dropped frames and node status.

The web server binds to localhost by default. If you intentionally want to use it
from another device on a trusted network, launch with
`dashboard_host:=0.0.0.0 dashboard_port:=8080` and configure your firewall
accordingly. The dashboard has no authentication, so do not expose it to an
untrusted network.

“Any topic” here means any ROS topic advertising one of the supported image
message types. The output publisher uses `std_msgs/msg/String` with the JSON
contract above; the chosen topic must not already be advertised with a different
message type. A custom ROS message can be added later if the rest of the rover
project needs a typed interface.

## Calibrate range for the fixed camera

A pothole's pixel width or area is not a dependable distance cue because holes
have different real-world sizes. This implementation instead maps the vertical
image row of the detection box's **bottom-center** (the nearest visible edge) to
known ground distances. With a rigid camera looking at a roughly flat surface,
perspective makes that row a useful range cue. The calibration is a measured,
piecewise-linear mapping, so the camera's fixed angle is accounted for by the
samples rather than needing to be entered manually.

1. Mount the camera in its final position. Keep height, pitch, lens/zoom, crop,
   resolution and rover posture fixed afterwards.
2. Take at least 3 (ideally 4–8) same-resolution images of ground points at known
   distances spanning the area you care about. Measure each distance **forward
   from the rover's front edge**. A marker on flat ground works well; if using
   pothole images, click the nearest visible edge consistently.
3. Run the calibration tool once per image and click the known point in each
   window. The resulting JSON is specific to that camera setup.

```bash
python -m pothole_rover.calibrate \
  --sample calibration/near.jpg=0.50 \
  --sample calibration/mid.jpg=1.00 \
  --sample calibration/far.jpg=2.00 \
  --sample calibration/farthest.jpg=3.50 \
  --camera-to-rover-front-m 0.18 \
  --camera-height-m 0.42 \
  --output calibration.json
```

Replace the example distances and camera dimensions with real measurements. The
`camera-to-rover-front` value is the horizontal distance from the camera's
vertical ground projection to the rover front; it is positive when the camera is
behind the front edge. With this value, the node reports camera-ground distance.
With camera height supplied too, it also reports the approximate straight-line
range from the camera center to the nearest ground point. Unknown camera-relative
ranges stay `null` rather than being guessed.

Use the resulting file in the dashboard's calibration path field, or launch the
node with `calibration_file:=/absolute/path/to/calibration.json`. Rows outside
the calibrated span are reported as bounds (`<=` near the rover or `>=` beyond
the farthest sample) instead of extrapolating.

This is a **ground-plane estimate to the nearest visible pothole edge**, not a
measurement of the depth inside the hole. Strong slopes, uneven terrain, a
shifted camera mount, or a box edge that is not on the surface reduce accuracy.
The model detects potholes only; it is not a general obstacle detector and does
not control rover motors.

## Standalone testing (without ROS)

For a quick local model test, install the Python runtime requirements and use the
original CLI:

```bash
python -m pip install -r requirements.txt
python -m pothole_rover --source 0 --target-fps 5
python -m pothole_rover --source pothole.jpg --save-image runs/annotated.jpg
python -m pothole_rover --source road-test.mp4 --headless \
  --save-video runs/annotated.mp4 --jsonl runs/detections.jsonl
```

Use `--weights path/to/best.pt`, `--device cpu`, `--confidence 0.35`, or
`--image-size 640` as needed. Run `python -m pothole_rover --help` for all
standalone options.

## Tests

```bash
python -m pip install -e ".[dev]"
python -m pytest
```
