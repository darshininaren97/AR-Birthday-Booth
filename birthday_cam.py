"""
Birthday Cam
------------
Webcam app that overlays a party hat on detected faces, decorates the frame
with confetti/banners, and lets the user take a photo by pointing their
index finger at an on-screen camera icon and holding it there.

Flow:
    1. Point at the camera icon and hold briefly to ARM the shutter.
    2. A 5-second countdown gives you time to pose.
    3. A shutter/flash effect plays and a clean (UI-free) photo is saved.

Requires two MediaPipe model files placed next to this script:
    - detector.tflite        (face detector)
    - hand_landmarker.task   (hand landmarker)
And three image assets (with alpha channels):
    - hat.png   (party hat overlay)
    - coww.png  (decorative cow sticker)
    - cam.png   (camera / shutter icon)
"""

import time
import math

import cv2
import numpy as np
import mediapipe as mp


# ============================================================
# CONFIG / CONSTANTS
# ============================================================

FACE_MODEL_PATH = "detector.tflite"
HAND_MODEL_PATH = "hand_landmarker.task"

HAT_IMAGE_PATH = "hat.png"
COW_IMAGE_PATH = "coww.png"
SHUTTER_ICON_PATH = "cam.png"

OUTPUT_VIDEO_PATH = "reaction.mp4"
OUTPUT_VIDEO_FPS = 20

# Landmark index for the index fingertip in MediaPipe's hand model.
INDEX_FINGERTIP_LANDMARK = 8

# Shutter button (also the finger-trigger hitbox) geometry, in pixels.
SHUTTER_BUTTON_X = 100
SHUTTER_BUTTON_Y = 300
SHUTTER_BUTTON_W = 70
SHUTTER_BUTTON_H = 70

# How long the fingertip must stay on the icon to ARM the shutter.
FINGER_HOLD_SECONDS = 0.33

# Time to pose after arming, before the photo is actually taken.
COUNTDOWN_SECONDS = 5

# Length of the shutter/flash effect after capture.
FLASH_DURATION_SECONDS = 0.5

# Cow sticker placement/size.
COW_DISPLAY_SIZE = (200, 150)
COW_POSITION = (0, 70)

# Pastel palette used for the bunting flags.
FLAG_COLORS = [
    (193, 182, 255),
    (230, 216, 173),
    (230, 230, 250),
    (240, 250, 255),
    (189, 252, 201),
]

# Bright palette used for falling confetti squares.
CONFETTI_COLORS = [
    (180, 0, 255),
    (255, 0, 255),
    (255, 191, 0),
    (255, 144, 30),
    (0, 255, 0),
    (150, 255, 50),
    (0, 242, 255),
    (0, 128, 255),
    (255, 0, 128),
]

# Initial confetti pieces: [x, y, fall_speed]
INITIAL_CONFETTI = [
    [50, 0, 3],
    [120, 50, 6],
    [200, 20, 4],
    [300, 100, 8],
    [450, 30, 5],
]

# Speed of the horizontal "scanning" line shown while no face is detected.
SCAN_LINE_SPEED = 10


# ============================================================
# MODEL SETUP
# ============================================================

def create_face_detector():
    """Builds a MediaPipe FaceDetector configured for single-image mode."""
    base_options = mp.tasks.BaseOptions(model_asset_path=FACE_MODEL_PATH)
    options = mp.tasks.vision.FaceDetectorOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
    )
    return mp.tasks.vision.FaceDetector.create_from_options(options)


def create_hand_landmarker():
    """Builds a MediaPipe HandLandmarker configured for video-stream mode."""
    base_options = mp.tasks.BaseOptions(model_asset_path=HAND_MODEL_PATH)
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp.tasks.vision.HandLandmarker.create_from_options(options)


# ============================================================
# IMAGE OVERLAY HELPERS
# ============================================================

def overlay_transparent_image(background, overlay, x, y, size=None):
    """
    Alpha-blends an RGBA `overlay` onto `background` with its top-left
    corner at (x, y). Optionally resizes the overlay to `size` (w, h) first.
    """
    if size is not None:
        overlay = cv2.resize(overlay, size)

    h, w = overlay.shape[:2]
    alpha = overlay[:, :, 3] / 255.0

    for channel in range(3):
        background[y:y + h, x:x + w, channel] = (
            alpha * overlay[:, :, channel]
            + (1 - alpha) * background[y:y + h, x:x + w, channel]
        )

    return background


def overlay_png_rotated(background, overlay, center_x, center_y, angle, target_width):
    """
    Scales `overlay` to `target_width` (preserving aspect ratio), rotates it
    by `angle` degrees, and centers the result at (center_x, center_y) on
    `background`. Handles transparency and clipping against frame edges.
    """
    # Scale the overlay proportionally based on the target width.
    scale_ratio = target_width / overlay.shape[1]
    target_height = int(overlay.shape[0] * scale_ratio)
    scaled_overlay = cv2.resize(overlay, (target_width, target_height))

    # Pad so the image isn't clipped once rotated.
    scaled_h, scaled_w = scaled_overlay.shape[:2]
    diagonal = int(np.sqrt(scaled_h ** 2 + scaled_w ** 2))
    pad_h = (diagonal - scaled_h) // 2
    pad_w = (diagonal - scaled_w) // 2

    padded_overlay = cv2.copyMakeBorder(
        scaled_overlay, pad_h, pad_h, pad_w, pad_w,
        cv2.BORDER_CONSTANT, value=[0, 0, 0, 0]
    )

    # Rotate around the padded image's center.
    rotation_center = (padded_overlay.shape[1] // 2, padded_overlay.shape[0] // 2)
    rotation_matrix = cv2.getRotationMatrix2D(rotation_center, angle, 1.0)
    rotated_overlay = cv2.warpAffine(
        padded_overlay, rotation_matrix,
        (padded_overlay.shape[1], padded_overlay.shape[0]),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=[0, 0, 0, 0]
    )

    # Composite onto the background, centered at (center_x, center_y).
    rot_h, rot_w = rotated_overlay.shape[:2]
    start_x = center_x - rot_w // 2
    start_y = center_y - rot_h // 2

    bg_h, bg_w = background.shape[:2]

    dst_x1, dst_x2 = max(0, start_x), min(bg_w, start_x + rot_w)
    dst_y1, dst_y2 = max(0, start_y), min(bg_h, start_y + rot_h)

    src_x1, src_x2 = max(0, -start_x), min(rot_w, bg_w - start_x)
    src_y1, src_y2 = max(0, -start_y), min(rot_h, bg_h - start_y)

    if dst_x1 >= dst_x2 or dst_y1 >= dst_y2:
        return background

    overlay_crop = rotated_overlay[src_y1:src_y2, src_x1:src_x2]
    alpha = overlay_crop[:, :, 3] / 255.0

    for channel in range(3):
        background[dst_y1:dst_y2, dst_x1:dst_x2, channel] = (
            alpha * overlay_crop[:, :, channel]
            + (1 - alpha) * background[dst_y1:dst_y2, dst_x1:dst_x2, channel]
        )

    return background


# ============================================================
# FACE / HAT PLACEMENT
# ============================================================

def compute_hat_placement(detection, frame_shape):
    """
    Uses the left/right eye keypoints of a face detection to work out:
      - the head-tilt angle (for rotating the hat), and
      - the pixel position just above the head where the hat should sit.

    Returns (center_x, center_y, tilt_angle_degrees) or None if the
    detection doesn't have eye keypoints to work with.
    """
    if not detection.keypoints or len(detection.keypoints) < 2:
        return None

    left_eye, right_eye = detection.keypoints[0], detection.keypoints[1]

    left_x = left_eye.x * frame_shape[1]
    left_y = left_eye.y * frame_shape[0]
    right_x = right_eye.x * frame_shape[1]
    right_y = right_eye.y * frame_shape[0]

    eye_mid_x = (left_x + right_x) / 2
    eye_mid_y = (left_y + right_y) / 2

    dx = right_x - left_x
    dy = right_y - left_y
    tilt_angle = np.degrees(np.arctan2(dy, dx))
    tilt_rad = np.radians(tilt_angle)

    # Distance to place the hat above the eye midpoint, scaled to face size.
    offset_distance = detection.bounding_box.height * 1.2

    # Rotate the "straight up" offset by the head tilt.
    offset_x = offset_distance * np.sin(tilt_rad)
    offset_y = -offset_distance * np.cos(tilt_rad)

    hat_center_x = int(eye_mid_x + offset_x)
    hat_center_y = int(eye_mid_y + offset_y)

    return hat_center_x, hat_center_y, tilt_angle


def draw_face_decorations(frame, detection, bbox, score, hat_image):
    """Draws the hat, face bounding box, and score label for one detected face."""
    placement = compute_hat_placement(detection, frame.shape)

    if placement is not None:
        hat_center_x, hat_center_y, tilt_angle = placement
        frame = overlay_png_rotated(
            frame, hat_image, hat_center_x, hat_center_y, -tilt_angle, bbox.width
        )

    x, y, w, h = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height

    cv2.rectangle(frame, (x, y), (x + w, y + h), (143, 72, 4), 2)
    cv2.putText(
        frame, f"Face {score:.2f}", (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1
    )

    return frame


# ============================================================
# PARTY DECORATIONS
# ============================================================

def draw_birthday_banner(frame):
    """Draws the 'Happy Birthday!!' text banner."""
    cv2.putText(
        frame, "Happy Birthday!!", (80, 70),
        cv2.FONT_ITALIC, 1, (5, 5, 5), 2
    )


def draw_bunting_arc(frame, flag_color_offset):
    """
    Draws a string of triangular bunting flags following a parabolic arc
    across the top of the frame, plus the connecting string dots.

    Returns the updated flag_color_offset (so colors keep cycling next frame).
    """
    # Connecting string, drawn as a dotted parabola.
    for x in range(640):
        y = int(-0.0008 * (x - 320) ** 2 + 120)
        cv2.circle(frame, (x, y), 1, (143, 72, 4), 3)

    # Triangular flags hanging along the same parabola.
    for x in range(10, 640, 40):
        y = int(-0.0008 * (x - 320) ** 2 + 120)

        left = (x - 15, y)
        right = (x + 15, y)
        tip = (x, y + 30)
        triangle = np.array([left, right, tip], dtype=np.int32)

        cv2.line(frame, right, tip, (0, 0, 0), 2)
        cv2.line(frame, tip, left, (0, 0, 0), 2)
        cv2.fillPoly(frame, [triangle], FLAG_COLORS[flag_color_offset % len(FLAG_COLORS)])

        flag_color_offset += 1

    return flag_color_offset


def draw_confetti(frame, confetti_pieces, frame_height):
    """
    Draws and animates falling square confetti pieces in place.
    Each piece is [x, y, fall_speed]; pieces respawn above the frame
    once they fall past the bottom.
    """
    for i, (x, y, speed) in enumerate(confetti_pieces):
        square = np.array(
            [[x, y], [x + 15, y], [x + 15, y + 15], [x, y + 15]],
            dtype=np.int32
        )
        cv2.fillPoly(frame, [square], CONFETTI_COLORS[i % len(CONFETTI_COLORS)])

        confetti_pieces[i][1] += speed
        if confetti_pieces[i][1] > frame_height:
            confetti_pieces[i][1] = np.random.randint(-200, 0)


def draw_scanning_line(frame, line_y, direction):
    """
    Draws a horizontal line that bounces up and down the frame — shown
    while no face is detected. Returns the updated (line_y, direction).
    """
    cv2.line(frame, (0, line_y), (680, line_y), (128, 0, 128), 3)

    line_y += direction
    if line_y >= 679:
        direction = -SCAN_LINE_SPEED
    if line_y <= 0:
        direction = SCAN_LINE_SPEED

    return line_y, direction


# ============================================================
# HAND / FINGERTIP DETECTION
# ============================================================

def detect_index_fingertip(hand_landmarker, mp_image, elapsed_ms, frame_w, frame_h):
    """
    Runs the hand landmarker and returns the pixel position of the index
    fingertip, or None if no hand is detected.
    """
    result = hand_landmarker.detect_for_video(mp_image, elapsed_ms)

    if not result.hand_landmarks:
        return None

    tip = result.hand_landmarks[0][INDEX_FINGERTIP_LANDMARK]
    return int(tip.x * frame_w), int(tip.y * frame_h)


def draw_fingertip_marker(frame, fingertip_px):
    """Draws a marker and coordinate label at the tracked fingertip."""
    cv2.circle(frame, fingertip_px, 10, (0, 255, 0), -1)
    cv2.putText(
        frame, f"({fingertip_px[0]}, {fingertip_px[1]})",
        (fingertip_px[0] + 15, fingertip_px[1]),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
    )


def is_pointing_at_shutter_button(fingertip_px):
    """Returns True if the fingertip position falls within the shutter button."""
    if fingertip_px is None:
        return False
    x, y = fingertip_px
    return (
        SHUTTER_BUTTON_X <= x <= SHUTTER_BUTTON_X + SHUTTER_BUTTON_W
        and SHUTTER_BUTTON_Y <= y <= SHUTTER_BUTTON_Y + SHUTTER_BUTTON_H
    )


# ============================================================
# SHUTTER STATE MACHINE
#
#   "idle"      -> waiting for a finger-point to arm the shutter
#   "countdown" -> armed, counting down so the person can pose
#   "flash"     -> photo just taken, playing the shutter/flash effect
# ============================================================

class ShutterState:
    IDLE = "idle"
    COUNTDOWN = "countdown"
    FLASH = "flash"


def handle_idle_state(frame, fingertip_px, finger_hold_start_time, frame_w, frame_h, now):
    """
    Idle state: watches for the fingertip resting on the shutter button.
    Draws a highlight box and a progress ring once a hold begins.

    Returns (next_state, updated_finger_hold_start_time).
    """
    if not is_pointing_at_shutter_button(fingertip_px):
        return ShutterState.IDLE, None

    cv2.rectangle(
        frame,
        (SHUTTER_BUTTON_X, SHUTTER_BUTTON_Y),
        (SHUTTER_BUTTON_X + SHUTTER_BUTTON_W, SHUTTER_BUTTON_Y + SHUTTER_BUTTON_H),
        (0, 255, 0), 3
    )

    if finger_hold_start_time is None:
        finger_hold_start_time = now

    hold_elapsed = now - finger_hold_start_time
    progress = min(hold_elapsed / FINGER_HOLD_SECONDS, 1.0)

    # Progress ring, centered on screen, shows how close we are to arming.
    ring_center = (frame_w // 2, frame_h // 2)
    cv2.ellipse(
        frame, ring_center, (60, 60), -90, 0, int(360 * progress),
        (0, 255, 0), 3
    )

    if hold_elapsed >= FINGER_HOLD_SECONDS:
        return ShutterState.COUNTDOWN, None

    return ShutterState.IDLE, finger_hold_start_time


def handle_countdown_state(frame, clean_frame, countdown_start_time, frame_w, frame_h, now, photo_counter):
    """
    Countdown state: shows a big number counting down, then saves a clean
    (UI-free) photo once time runs out.

    Returns (next_state, updated_photo_counter).
    """
    remaining = COUNTDOWN_SECONDS - (now - countdown_start_time)

    if remaining <= 0:
        photo_counter += 1
        cv2.imwrite(f"p{photo_counter}.jpg", clean_frame)
        return ShutterState.FLASH, photo_counter

    display_number = math.ceil(remaining)
    text = str(display_number)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale, thickness = 2, 5
    (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_x = frame_w // 2 - text_w // 2
    text_y = frame_h // 2 + text_h // 2

    # Backing circle so the number reads clearly on any background.
    cv2.circle(frame, (frame_w // 2, frame_h // 2), 70, (0, 0, 0), 1)
    cv2.circle(frame, (frame_w // 2, frame_h // 2), 60, (255, 255, 255), 2)
    cv2.putText(frame, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    cv2.putText(
        frame, "Get ready!",
        (frame_w // 2 - 80, frame_h // 2 - max(text_w, text_h) - 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA
    )

    return ShutterState.COUNTDOWN, photo_counter


def handle_flash_state(frame, flash_start_time, frame_w, frame_h, now):
    """
    Flash state: animates shutter blades closing/opening plus a bright
    flash, then returns to idle once the effect finishes.

    Returns next_state.
    """
    elapsed = now - flash_start_time
    t = min(elapsed / FLASH_DURATION_SECONDS, 1.0)

    # Shutter blades: closing over the first half, opening over the second half.
    closing = (t / 0.5) if t < 0.5 else (1 - (t - 0.5) / 0.5)
    bar_height = int(closing * (frame_h / 2))
    if bar_height > 0:
        cv2.rectangle(frame, (0, 0), (frame_w, bar_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (0, frame_h - bar_height), (frame_w, frame_h), (0, 0, 0), -1)

    # Bright white flash, strongest right as the blades meet.
    flash_alpha = max(0.0, 1 - abs(t - 0.5) / 0.25)
    if flash_alpha > 0:
        white = np.full_like(frame, 255)
        cv2.addWeighted(frame, 1 - flash_alpha, white, flash_alpha, 0, frame)

    if elapsed >= FLASH_DURATION_SECONDS:
        return ShutterState.IDLE

    return ShutterState.FLASH


# ============================================================
# MAIN LOOP
# ============================================================

def main():
    face_detector = create_face_detector()
    hand_landmarker = create_hand_landmarker()
    hand_tracking_start_time = time.time()

    hat_image = cv2.imread(HAT_IMAGE_PATH, cv2.IMREAD_UNCHANGED)
    cow_image = cv2.imread(COW_IMAGE_PATH, cv2.IMREAD_UNCHANGED)
    shutter_icon_image = cv2.imread(SHUTTER_ICON_PATH, cv2.IMREAD_UNCHANGED)

    cap = cv2.VideoCapture(0)
    ret, first_frame = cap.read()
    if not ret:
        raise RuntimeError("Could not read from webcam.")
    frame_h, frame_w = first_frame.shape[:2]

    video_writer = cv2.VideoWriter(
        OUTPUT_VIDEO_PATH, cv2.VideoWriter_fourcc(*"mp4v"),
        OUTPUT_VIDEO_FPS, (frame_w, frame_h)
    )

    # Animation / decoration state.
    scan_line_y = 0
    scan_line_direction = SCAN_LINE_SPEED
    flag_color_offset = 0
    confetti_pieces = [list(piece) for piece in INITIAL_CONFETTI]
    photo_counter = 0

    # Shutter state machine state.
    shutter_state = ShutterState.IDLE
    finger_hold_start_time = None
    countdown_start_time = None
    flash_start_time = None

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame_h, frame_w = frame.shape[:2]
        now = time.time()

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # ---- Face detection ----
        detection_result = face_detector.detect(mp_image)
        detections = detection_result.detections
        face_detected = len(detections) > 0

        # ---- Fingertip tracking (only needed while idle, waiting to arm) ----
        fingertip_px = None
        if shutter_state == ShutterState.IDLE:
            elapsed_ms = int((now - hand_tracking_start_time) * 1000)
            fingertip_px = detect_index_fingertip(
                hand_landmarker, mp_image, elapsed_ms, frame_w, frame_h
            )
            if fingertip_px is not None:
                draw_fingertip_marker(frame, fingertip_px)

        # ---- Face decorations, or scanning line if nobody's there ----
        if face_detected:
            for detection in detections:
                bbox = detection.bounding_box
                score = detection.categories[0].score
                frame = draw_face_decorations(frame, detection, bbox, score, hat_image)

            draw_birthday_banner(frame)
            flag_color_offset = draw_bunting_arc(frame, flag_color_offset)
            draw_confetti(frame, confetti_pieces, frame_h)
        else:
            scan_line_y, scan_line_direction = draw_scanning_line(
                frame, scan_line_y, scan_line_direction
            )

        # ---- Cow sticker ----
        overlay_transparent_image(frame, cow_image, *COW_POSITION, size=COW_DISPLAY_SIZE)

        # Snapshot before shutter UI is drawn, so the icon/countdown never
        # end up baked into the saved photo.
        clean_frame = frame.copy()

        # ---- Shutter icon (also the finger-trigger button) ----
        overlay_transparent_image(
            frame, shutter_icon_image, SHUTTER_BUTTON_X, SHUTTER_BUTTON_Y,
            size=(SHUTTER_BUTTON_W, SHUTTER_BUTTON_H)
        )

        # ---- Shutter state machine ----
        if shutter_state == ShutterState.IDLE:
            shutter_state, finger_hold_start_time = handle_idle_state(
                frame, fingertip_px, finger_hold_start_time, frame_w, frame_h, now
            )
            if shutter_state == ShutterState.COUNTDOWN:
                countdown_start_time = now

        elif shutter_state == ShutterState.COUNTDOWN:
            shutter_state, photo_counter = handle_countdown_state(
                frame, clean_frame, countdown_start_time, frame_w, frame_h, now, photo_counter
            )
            if shutter_state == ShutterState.FLASH:
                flash_start_time = now

        elif shutter_state == ShutterState.FLASH:
            shutter_state = handle_flash_state(frame, flash_start_time, frame_w, frame_h, now)
            if shutter_state == ShutterState.IDLE:
                finger_hold_start_time = None

        cv2.imshow("FACE", frame)

        if face_detected:
            video_writer.write(frame)

        if cv2.waitKey(1) == 27:  # Esc key
            break

    video_writer.release()
    cap.release()
    cv2.destroyAllWindows()
    face_detector.close()
    hand_landmarker.close()


if __name__ == "__main__":
    main()
