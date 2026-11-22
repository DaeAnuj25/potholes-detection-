# Pothole detection for a rover

This repo now contains a small camera/video inference runner around the supplied
Ultralytics model. It converts every incoming frame to grayscale before inference,
samples at a default target of **5 FPS**, draws pothole detections, and can export
per-frame JSONL telemetry. A separate calibration tool maps the nearest visible
edge of a detected pothole to an approximate ground distance.

## Model in this checkout

The tracked checkpoint is named **`new_model.pt`** (there is no `best.pt` in the
current checkout). The runner uses it by default; if you later add `best.pt`, it
will prefer that file, or you can select any checkpoint with `--weights`. The
checkpoint metadata identifies it as an Ultralytics YOLO11n detector with one
class, `pothole`.

## Install

Use Python 3.10, 3.11, or 3.12. PyTorch/Ultralytics installation can take a while.

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell instead:
# .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

For development/tests, install `python -m pip install -e ".[dev]"`.
Ultralytics chooses an available inference device automatically. You can force a
device with `--device cpu` or, for example, `--device 0` when a compatible CUDA
PyTorch installation is present.

## Try the model

Open the default camera (usually camera index `0`):

```bash
python -m pothole_rover --source 0 --target-fps 5
```

The preview is grayscale with colored boxes and confidence scores. Press **Q**
or **Esc** to close it. Test another webcam with `--source 1`, a saved image with
`--source pothole.jpg`, a video with `--source road.mp4`, or a stream URL such as
`--source rtsp://camera-address/stream`.

Save an annotated still and machine-readable detections:

```bash
python -m pothole_rover \
  --source pothole.jpg \
  --save-image runs/pothole-annotated.jpg \
  --jsonl runs/pothole-detections.jsonl
```

Process a video without a display, saving an annotated video and JSONL output:

```bash
python -m pothole_rover \
  --source road-test.mp4 \
  --headless \
  --save-video runs/road-test-annotated.mp4 \
  --jsonl runs/road-test-detections.jsonl
```

Other useful options: `--weights path/to/best.pt`, `--confidence 0.35`,
`--image-size 640`, `--device cpu`, `--width 1280 --height 720`, and
`--jsonl -` to write JSONL records to stdout. Run `python -m pothole_rover --help`
for the full list.

### FPS behavior

For a live camera/stream, the runner drops frames and starts no more than five
inferences per wall-clock second by default. It asks the capture backend for a
one-frame buffer when supported, to reduce stale-frame buildup. For video files,
it samples approximately five frames per **source-video** second. Change this
with `--target-fps`.

Five FPS is a target/maximum, not a hardware guarantee: if one inference takes
longer than 200 ms, the rover computer cannot sustain five inferences per second.
The console and preview report inference time and achieved processing rate. Try a
smaller `--image-size` or a faster device if the achieved rate is too low.

## Calibrate range for your fixed camera

A box's pixel width/area is not a dependable distance cue when potholes have
different real sizes. This implementation instead uses the **vertical image row
of the box's bottom-center** (the nearest visible edge) and your measured ground
distances. With a rigid camera looking at a roughly flat surface, perspective
makes that row a useful range cue. The calibration is an empirical,
piecewise-linear mapping; it does not need a separately measured camera angle.

1. Mount the camera in its final position. Keep camera height, pitch, lens/zoom,
   crop, resolution and rover posture fixed afterwards.
2. On a flat test surface, take at least 3 (ideally 4–8) images with clear ground
   points at known distances spanning the area you care about. Measure each
   distance **forward from the rover's front edge**. Keep every image at the same
   resolution. A visible marker works well; for pothole images, choose the
   nearest visible edge consistently.
3. Run the tool once per image. In each window click the known ground point (or
   the pothole's nearest edge). The tool records the clicked image row and the
   known rover-front distance, then writes `calibration.json`.

Example (replace the image names and distances with your own measurements):

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

`--camera-to-rover-front-m` is the horizontal distance from the camera's vertical
ground projection to the rover front: use a **positive** value when the camera is
behind the front edge (for example, 0.18 m). With it set, each record also gets a
camera-ground distance. If `--camera-height-m` is supplied too, the runner also
reports the approximate straight-line distance from the camera center to the
pothole's nearest ground point. Leave either option out if you do not know it;
unknown camera-relative ranges are reported as `null`, not guessed.

Then run the live camera with the saved map:

```bash
python -m pothole_rover \
  --source 0 \
  --target-fps 5 \
  --calibration calibration.json \
  --jsonl runs/rover-detections.jsonl
```

Out-of-range image rows are reported as a bound (for example, `<= 0.50 m` or
`>= 3.50 m`) instead of extrapolating beyond the measured area. The calibration
file is ordinary JSON and can be inspected or edited if you have already
measured the image rows.

### What the distance means (and does not mean)

The main range is a **ground-plane estimate from the rover's front edge to the
nearest visible pothole edge**. The optional camera distances are also ground
plane/slant estimates, as described above. It is not a measurement of the depth
inside the hole. This assumption becomes less accurate on strongly uneven or
sloped ground, when the mount shifts, or when the bottom of the detection box is
not on the ground surface. Out-of-calibration detections remain detections, but
the range is only a bound.

## JSONL output

Each processed frame produces one line. Detection records include class,
confidence, pixel box, nearest-edge row, distance/status, and optional
camera-relative ranges. For example, before range calibration the distance
fields are `null`; after calibration, an in-range record resembles:

```json
{"class_name":"pothole","confidence":0.91,"box_xyxy_px":[210,180,392,335],"closest_edge_pixel_y":335,"distance_from_rover_front_m":1.27,"distance_from_camera_ground_m":1.45,"distance_from_camera_center_m":1.51,"distance_status":"within_calibration_range"}
```

The stream is suitable for a separate rover-control process to consume. This
runner deliberately does **not** command motors or decide braking behavior.
Test at low speed, add a fail-safe for missing/stale detections, and do not treat
one camera/model as a guaranteed safety system. The current checkpoint detects
potholes only; it is not an obstacle or general terrain detector.

## Tests

```bash
python -m pip install -e ".[dev]"
python -m pytest
```
