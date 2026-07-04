# Birthday Cam 🎉

A webcam party app that puts a hat on your head, decorates the frame with
confetti and bunting, and lets you take a photo hands-free — just point your
index finger at the on-screen camera icon and hold it there.

<!-- Add a demo GIF or screenshot here, e.g. -->
<!-- ![demo](demo.gif) -->

## How it works

- **Face detection** (MediaPipe) finds your face, estimates head tilt from
  your eye positions, and rotates/places a party hat accordingly.
- **Hand tracking** (MediaPipe) watches for your index fingertip.
- When you point at the camera icon and hold for ~⅓ of a second, the shutter
  **arms**.
- A **5-second countdown** appears so you can pose.
- A **shutter/flash animation** plays, and a clean photo (no UI, no icon) is
  saved to disk.
- While a face is detected, the session is also recorded to `reaction.mp4`.

## Requirements

- Python 3.9+
- A webcam

Install dependencies:

```bash
pip install -r requirements.txt
```

## Setup

### 1. Download the MediaPipe models

Place both files in the same folder as `birthday_cam.py`:

| File | Download |
|---|---|
| `detector.tflite` | [BlazeFace short-range face detector](https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite) — save it as `detector.tflite` |
| `hand_landmarker.task` | [Hand Landmarker (float16)](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task) |

```bash
curl -L -o detector.tflite \
  https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite

curl -L -o hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```

> If either link ever 404s (Google occasionally bumps model versions), check the current URLs in the [MediaPipe model index](https://ai.google.dev/edge/mediapipe/solutions/vision/face_detector#models) and [Hand Landmarker docs](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker#models).

### 2. Add the image assets

Place these three **transparent PNGs** (with an alpha channel) in the same
folder as `birthday_cam.py`:

| File | Purpose | Notes |
|---|---|---|
| `hat.png` | Party hat overlay | A typical party hat pointing up, transparent background |
| `coww.png` | Decorative sticker in the corner | Any transparent PNG works — in the original build this is a cow wearing a party hat, peeking into frame from the corner 🐮🎉 |
| `cam.png` | Camera / shutter button icon | Should read clearly at ~70×70px |
> **Note:** The image assets used in this project (e.g., the party hat, cow, and camera icon) were sourced from the internet and are not my original work. All credit belongs to their respective creators.

You can use your own art here, or grab free transparent PNGs from any site.

### 3. Folder structure

Once set up, your folder should look like:

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

- Point your index finger at the camera icon (bottom-left) and hold it
  there until the ring fills up.
- Pose during the 5-second countdown.
- Your photo is saved as `p1.jpg`, `p2.jpg`, etc. in the project folder.
- The full session (while a face is visible) is saved to `reaction.mp4`.
- Press **Esc** to quit.

## Configuration

All the tunable bits live at the top of `birthday_cam.py`:

| Constant | Default | Description |
|---|---|---|
| `FINGER_HOLD_SECONDS` | `0.33` | How long to hold your finger on the icon to arm the shutter |
| `COUNTDOWN_SECONDS` | `5` | Time to pose before the photo is taken |
| `FLASH_DURATION_SECONDS` | `0.5` | Length of the shutter/flash effect |
| `SHUTTER_BUTTON_X/Y/W/H` | `100, 300, 70, 70` | Position/size of the camera icon & finger hitbox |
| `COW_DISPLAY_SIZE` / `COW_POSITION` | `(200, 150)` / `(0, 70)` | Size/position of the cow sticker |
| `OUTPUT_VIDEO_PATH` | `reaction.mp4` | Where the recorded session is saved |

## The math behind the magic 🎓

None of this needs to be understood to run the app — but if you're curious
what's actually happening under the hood, here's the math for the three
"decorations":

### 🎊 Confetti — not quite particle physics

It *looks* like physics, but it's simpler than that. Each confetti piece is
just `[x, y, speed]`, and every frame:

```
y = y + speed
```

That's **constant-velocity motion** — no gravity, no acceleration, no drag.
Real confetti falling under gravity would need `speed += g * dt` each frame
so it speeds up as it falls. Once a piece drifts past the bottom of the
frame, it's teleported back to a random y above the top (`-200` to `0`), so
it looks like a fresh piece — a cheap and easy way to fake an endless
confetti shower with only 5 objects.

### 🎉 Banner — a real parabola, used the way engineers use it for cables

The connecting string is drawn from:

```
y = -0.0008 * (x - 320)² + 120
```

This is a downward-sagging parabola: it's *highest* at the edges (`x = 0`
and `x = 640`) and *dips lowest* in the middle (`x = 320`) — exactly like a
string of bunting pinned up at two corners and sagging under its own weight.
(Physically, a hanging cable actually forms a **catenary**, `cosh(x)`, not a
parabola — but for a shallow sag like this one, a parabola is visually
indistinguishable and much cheaper to compute.)

The triangular flags are then placed at even intervals of `x` **along that
same curve**, with each triangle's tip drawn 30px *below* its string point —
so the flags hang down from the string exactly the way real bunting does,
rather than being pasted on in a straight row.

### 🎩 Party hat — a 2D rotation matrix, literally

This is the neatest bit of math in the whole project. To make the hat tilt
naturally with your head:

1. Find the midpoint between your two detected eyes.
2. Compute the tilt angle `θ` of your head from the angle between your eyes:
   `θ = atan2(Δy, Δx)`.
3. Take the vector that points *straight up* from that midpoint — `(0, -1)`
   — and **rotate it by θ** using the standard 2D rotation matrix:

```
[cos θ   -sin θ]   [ 0 ]     [ sin θ ]
[sin θ    cos θ] · [-1 ] =  [-cos θ ]
```

That rotated vector, scaled by the face size, is exactly where the hat's
center gets placed — so as your head tilts, the "up" direction rotates with
it, and the hat leans the same way your head does instead of always
pointing straight up. The hat image itself is then rotated by the same
angle so its orientation matches too.

## Known limitations

- Hardcoded webcam index (`cv2.VideoCapture(0)`) — if you have multiple
  cameras, you may need to change this.
- Some decorative elements (the confetti bounds, bunting arc) assume a
  roughly 640×480 frame and may look slightly off on other resolutions.
- Only tracks one hand and one shutter gesture at a time.

  
> **Note:** This is a beginner computer vision project created as a learning exercise with Python, OpenCV, and MediaPipe. The focus was on understanding the underlying concepts rather than building a production-ready application.


## License

MIT License. 5... 4... 3... 2... 1... you may now use this code freely.
