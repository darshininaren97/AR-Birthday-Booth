# Birthday Cam

A webcam party application that overlays a party hat on the user's head, decorates the frame with confetti and bunting, and enables hands-free photo capture. The user points their index finger at an on-screen camera icon and holds the gesture to trigger the shutter.

<!-- Add a demo GIF or screenshot here, e.g. -->
<!-- ![demo](demo.gif) -->

## Overview

This project was developed as a beginner computer vision exercise using Python, OpenCV, and MediaPipe. The primary goal was to understand the underlying concepts — face and hand tracking, coordinate geometry, and basic animation — rather than to produce a production-ready application.

## How It Works

- **Face detection** (MediaPipe): Locates the user's face and estimates head tilt from eye positions, which determines how the party hat is rotated and placed.
- **Hand tracking** (MediaPipe): Tracks the position of the index fingertip.
- **Gesture-based shutter**: When the user points at the camera icon and holds the position for approximately one-third of a second, the shutter arms.
- **Countdown**: A five-second countdown follows, giving the user time to pose.
- **Capture**: A shutter/flash animation plays, and a clean photo (with no UI or icon overlay) is saved to disk.
- **Recording**: While a face is detected, the session is simultaneously recorded to `reaction.mp4`.

## Requirements

- Python 3.9+
- A webcam

Install dependencies:

```bash
pip install -r requirements.txt
```

## Setup

### 1. Download the MediaPipe Models

Place both files in the same directory as `birthday_cam.py`:

| File | Source |
|---|---|
| `detector.tflite` | [BlazeFace short-range face detector](https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite) — save as `detector.tflite` |
| `hand_landmarker.task` | [Hand Landmarker (float16)](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task) |

```bash
curl -L -o detector.tflite \
  https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite

curl -L -o hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```

> If either link returns a 404 error (model versions are occasionally updated), refer to the current URLs in the [MediaPipe model index](https://ai.google.dev/edge/mediapipe/solutions/vision/face_detector#models) and the [Hand Landmarker documentation](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker#models).

### 2. Add the Image Assets

Place the following three **transparent PNGs** (with an alpha channel) in the same directory as `birthday_cam.py`:

| File | Purpose | Notes |
|---|---|---|
| `hat.png` | Party hat overlay | A standard party hat pointing upward, on a transparent background |
| `coww.png` | Decorative corner sticker | Any transparent PNG is compatible — in the original build, this depicts a cow wearing a party hat, positioned in the corner of the frame |
| `cam.png` | Camera / shutter button icon | Should be legible at approximately 70×70 pixels |

> **Note:** The image assets used in this project (the party hat, cow, and camera icon) were sourced from the internet and are not original work. Credit belongs to their respective creators.

Custom artwork may be substituted, or free transparent PNGs may be sourced from any suitable site.

### 3. Folder Structure

Once setup is complete, the project directory should be organized as follows:

```
birthday_cam/
├── birthday_cam.py
├── detector.tflite
├── hand_landmarker.task
├── hat.png
├── coww.png
└── cam.png
```

## Usage

```bash
python birthday_cam.py
```

- Point the index finger at the camera icon (bottom-left corner) and hold the position until the ring fills.
- Pose during the five-second countdown.
- The photo is saved as `p1.jpg`, `p2.jpg`, etc. in the project directory.
- The full session (while a face is visible) is saved to `reaction.mp4`.
- Press **Esc** to exit the application.

## Configuration

All configurable parameters are located at the top of `birthday_cam.py`:

| Constant | Default | Description |
|---|---|---|
| `FINGER_HOLD_SECONDS` | `0.33` | Duration the finger must remain on the icon to arm the shutter |
| `COUNTDOWN_SECONDS` | `5` | Time allotted to pose before the photo is captured |
| `FLASH_DURATION_SECONDS` | `0.5` | Duration of the shutter/flash effect |
| `SHUTTER_BUTTON_X/Y/W/H` | `100, 300, 70, 70` | Position and size of the camera icon and finger hitbox |
| `COW_DISPLAY_SIZE` / `COW_POSITION` | `(200, 150)` / `(0, 70)` | Size and position of the cow sticker |
| `OUTPUT_VIDEO_PATH` | `reaction.mp4` | File path for the recorded session |

## Technical Notes

The following section outlines the mathematical basis for the three visual effects implemented in this project. Understanding this is not required to run the application, but is included for reference.

### Confetti

The confetti effect approximates falling particles without simulating true physics. Each confetti piece is represented as `[x, y, speed]`, and updated every frame according to:

```
y = y + speed
```

This is constant-velocity motion, with no gravity, acceleration, or drag. Physically accurate confetti falling under gravity would require the velocity term to increase each frame (`speed += g * dt`). Once a piece passes the bottom of the frame, its `y` coordinate is reset to a random value above the top of the frame (between `-200` and `0`), giving the appearance of a continuous confetti shower while only maintaining five objects in memory.

### Banner (Bunting)

The connecting string is drawn using the equation:

```
y = -0.0008 * (x - 320)² + 120
```

This is a downward-sagging parabola, at its highest at the edges (`x = 0` and `x = 640`) and at its lowest at the midpoint (`x = 320`) — consistent with a string of bunting pinned at two corners and sagging under its own weight. Note that a hanging cable under gravity technically forms a catenary curve (`cosh(x)`) rather than a parabola; however, for shallow sag, a parabola is visually indistinguishable and computationally less expensive.

The triangular flags are positioned at even intervals of `x` along this same curve, with each flag's tip offset 30 pixels below its corresponding point on the string. This ensures the flags hang naturally from the string, rather than appearing as a straight, evenly spaced row.

### Party Hat Rotation

The party hat's orientation is determined using a standard 2D rotation matrix, applied as follows:

1. The midpoint between the two detected eyes is calculated.
2. The head tilt angle `θ` is computed from the angle between the eyes: `θ = atan2(Δy, Δx)`.
3. A vector pointing directly upward from that midpoint, `(0, -1)`, is rotated by `θ` using the standard 2D rotation matrix:

```
[cos θ   -sin θ]   [ 0 ]     [ sin θ ]
[sin θ    cos θ] · [-1 ] =  [-cos θ ]
```

This rotated vector, scaled according to face size, determines the placement of the hat's center. As the user's head tilts, the "upward" direction rotates accordingly, causing the hat to lean in the same direction as the head rather than remaining fixed vertically. The hat image itself is rotated by the same angle to maintain consistent orientation.

## Known Limitations

- The webcam index is hardcoded (`cv2.VideoCapture(0)`); users with multiple cameras may need to modify this value.
- Certain visual elements (confetti bounds, bunting arc) assume an approximate 640×480 frame and may appear misaligned at other resolutions.
- The application tracks only one hand and one shutter gesture at a time.

## License

MIT License.
