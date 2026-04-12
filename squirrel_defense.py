"""
🐿️ Squirrel Defense — Main Loop
=================================
by Troy (age 9) and Dad

Watches the bird feeder with the Pi Camera. When a squirrel
is detected, activates a deterrent (relay on GPIO 18) to shoo
it away. Birds and raccoons are logged but left alone.

Usage:
    python squirrel_defense.py              # run armed (deterrent active)
    python squirrel_defense.py --disarmed   # run in watch-only mode (no shooing)

Press Ctrl+C to stop.
"""

import json
import logging
import os
import signal
import sys
import time
from datetime import datetime

import numpy as np
from PIL import Image

try:
    from picamera2 import Picamera2
except ImportError:
    Picamera2 = None  # Not on Pi — camera functions unavailable

from detect_animals import (
    find_bsr_model,
    load_bsr_labelmap,
    load_model,
    detect_in_image,
    CONFIDENCE_THRESHOLD,
)

# ============================================================
#  ⚙️  SETTINGS — Troy, tweak these!
# ============================================================

# How confident does the AI need to be to shoo? (0.0 to 1.0)
SHOO_THRESHOLD = 0.5

# Seconds to wait between shoos (so we don't overdo it)
COOLDOWN_SECONDS = 8

# How often to check the camera (seconds between frames)
FRAME_INTERVAL = 0.5

# GPIO pin for the relay (physical pin 12 = GPIO 18)
RELAY_GPIO_PIN = 18

# How long to keep the relay closed (trigger pull duration)
RELAY_PULSE_SECONDS = 0.3

# Where to save detection snapshots
DETECTIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "detections")

# Where to save logs
LOGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")

# Camera crop config (shared with capture.py)
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "camera_config.json")


# ============================================================
#  📝  Logging — writes to both terminal and logs/ directory
# ============================================================

def setup_logging():
    os.makedirs(LOGS_DIR, exist_ok=True)
    log_file = os.path.join(LOGS_DIR, f"defense_{datetime.now().strftime('%Y%m%d')}.log")

    log = logging.getLogger("squirrel_defense")
    log.setLevel(logging.INFO)

    # File handler — one log file per day
    fh = logging.FileHandler(log_file)
    fh.setFormatter(logging.Formatter("%(asctime)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    log.addHandler(fh)

    # Console handler — also print to terminal
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(ch)

    return log


# ============================================================
#  📷  Camera
# ============================================================

def load_crop_config():
    """Load crop region from camera_config.json if it exists."""
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH) as f:
            config = json.load(f)
        if "crop" in config:
            box = config["crop"]
            return (box["left"], box["top"], box["right"], box["bottom"])
    return None


def start_camera():
    """Start the Pi Camera and return the Picamera2 instance."""
    if Picamera2 is None:
        print("❌ picamera2 not installed! Run: pip install picamera2")
        sys.exit(1)
    camera = Picamera2()
    config = camera.create_still_configuration()
    camera.configure(config)
    camera.start()
    # Let auto-exposure settle on startup
    time.sleep(2)
    print("📷 Camera ready!")
    return camera


def capture_frame(camera, crop_box=None):
    """Capture a single frame as a PIL Image, optionally cropped."""
    img = camera.capture_image("main")
    if crop_box:
        img = img.crop(crop_box)
    return img


# ============================================================
#  💨  Relay (deterrent trigger)
# ============================================================

class Relay:
    """Controls the 5V relay module on a GPIO pin."""

    def __init__(self, pin, pulse_seconds=0.3, enabled=True):
        self.pin = pin
        self.pulse_seconds = pulse_seconds
        self.enabled = enabled
        self.device = None

        if self.enabled:
            try:
                from gpiozero import OutputDevice
                self.device = OutputDevice(pin)
                print(f"💨 Relay ready on GPIO {pin}")
            except Exception as e:
                print(f"⚠️  Could not initialize relay on GPIO {pin}: {e}")
                print("   Running in watch-only mode.")
                self.enabled = False
        else:
            print("💨 Relay DISARMED (watch-only mode)")

    def shoo(self):
        """Pulse the relay to activate the deterrent."""
        if not self.enabled or not self.device:
            print("   💨 [DRY RUN] Would shoo!")
            return

        print("   💨 SHOO!")
        self.device.on()
        time.sleep(self.pulse_seconds)
        self.device.off()

    def cleanup(self):
        """Release GPIO resources."""
        if self.device:
            self.device.off()
            self.device.close()


# ============================================================
#  🧠  Detection (reuses detect_animals.py engine)
# ============================================================

def detect_from_image(img, interpreter, input_details, output_details,
                      model_size, labels):
    """Run BSR detection on a PIL Image. Returns list of detections."""
    width, height = model_size
    original_size = img.size
    img_resized = img.resize((width, height))

    input_data = np.expand_dims(np.array(img_resized, dtype=np.uint8), axis=0)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    num_outputs = len(output_details)

    if num_outputs >= 4:
        boxes = interpreter.get_tensor(output_details[0]['index'])[0]
        classes = interpreter.get_tensor(output_details[1]['index'])[0]
        scores = interpreter.get_tensor(output_details[2]['index'])[0]
        count = int(interpreter.get_tensor(output_details[3]['index'])[0])
    else:
        raw = {}
        for d in output_details:
            tensor = interpreter.get_tensor(d['index'])[0]
            raw[tensor.shape[-1]] = tensor
        raw_scores = raw[90]
        raw_boxes = raw[4]
        best_class = np.argmax(raw_scores, axis=1)
        best_score = np.max(raw_scores, axis=1)
        mask = best_score >= SHOO_THRESHOLD
        if np.any(mask):
            from detect_animals import _nms
            filt_boxes = raw_boxes[mask]
            filt_scores = best_score[mask]
            filt_classes = best_class[mask]
            keep = _nms(filt_boxes, filt_scores)
            boxes = filt_boxes[keep]
            scores = filt_scores[keep]
            classes = filt_classes[keep]
            count = len(keep)
        else:
            return []

    detections = []
    for i in range(min(count, len(scores))):
        score = float(scores[i])
        if score < SHOO_THRESHOLD:
            continue
        class_id = int(classes[i])
        label = labels.get(class_id, f"class_{class_id}")
        detections.append({"label": label, "score": score})

    return detections


# ============================================================
#  📊  Event tracker — deduplicates detections
# ============================================================

class EventTracker:
    """Tracks detection state to avoid logging the same scene repeatedly.

    Only reports a new "event" when:
    - The set of detected labels changes (e.g. nothing→bird, bird→squirrel)
    - The scene was clear for at least `quiet_seconds` and the same animal returns
    - A squirrel is detected and the shoo cooldown has elapsed (always an event)
    """

    def __init__(self, cooldown_seconds=8, quiet_seconds=30):
        self.cooldown_seconds = cooldown_seconds
        self.quiet_seconds = quiet_seconds
        self.last_labels = set()
        self.last_event_time = 0
        self.last_detection_time = 0
        self.last_shoo_time = 0

    def _detection_labels(self, detections):
        return frozenset(d["label"] for d in detections)

    def update(self, detections, now):
        """Process a frame's detections. Returns an action dict.

        Returns:
            {
                "is_new_event": bool,   — should we log/snapshot this?
                "should_shoo": bool,    — should we activate the deterrent?
                "cooldown_remaining": float or None,
            }
        """
        current_labels = self._detection_labels(detections)
        squirrels = any(d["label"] == "squirrel" for d in detections)

        # Determine if this is a new event worth logging
        scene_changed = current_labels != self.last_labels
        was_quiet = (now - self.last_detection_time) >= self.quiet_seconds
        is_new_event = scene_changed or was_quiet

        # Determine if we should shoo
        should_shoo = False
        cooldown_remaining = None
        if squirrels:
            time_since_shoo = now - self.last_shoo_time
            if time_since_shoo >= self.cooldown_seconds:
                should_shoo = True
                is_new_event = True  # always log a shoo event
            else:
                cooldown_remaining = self.cooldown_seconds - time_since_shoo

        # Update state
        self.last_labels = current_labels
        self.last_detection_time = now
        if is_new_event:
            self.last_event_time = now
        if should_shoo:
            self.last_shoo_time = now

        return {
            "is_new_event": is_new_event,
            "should_shoo": should_shoo,
            "cooldown_remaining": cooldown_remaining,
        }

    def clear(self):
        """Called when a frame has no detections."""
        self.last_labels = set()


# ============================================================
#  💾  Save detection snapshots
# ============================================================

def save_snapshot(img, detections, timestamp, log):
    """Save a detection snapshot with labels in the filename."""
    os.makedirs(DETECTIONS_DIR, exist_ok=True)
    det_labels = "_".join(sorted(set(d["label"] for d in detections)))
    filename = f"{timestamp}_{det_labels}.jpg"
    path = os.path.join(DETECTIONS_DIR, filename)
    img.save(path)
    log.info("Snapshot saved: detections/%s", filename)


# ============================================================
#  🚀  Main loop
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="🐿️ Squirrel Defense")
    parser.add_argument("--disarmed", action="store_true",
                        help="Watch-only mode (no shooing)")
    parser.add_argument("--threshold", type=float, default=SHOO_THRESHOLD,
                        help=f"Shoo confidence threshold (default: {SHOO_THRESHOLD})")
    parser.add_argument("--cooldown", type=float, default=COOLDOWN_SECONDS,
                        help=f"Seconds between shoos (default: {COOLDOWN_SECONDS})")
    args = parser.parse_args()

    shoo_threshold = args.threshold
    cooldown = args.cooldown

    log = setup_logging()

    print()
    print("╔══════════════════════════════════════════════╗")
    print("║   🐿️  SQUIRREL DEFENSE by Troy & Dad         ║")
    print("║   Protecting the bird feeder since 2026!     ║")
    print("╚══════════════════════════════════════════════╝")
    print()

    # Load model
    model_path = find_bsr_model()
    if model_path is None:
        print("❌ BSR model not found! See README for download instructions.")
        sys.exit(1)
    labels = load_bsr_labelmap(os.path.dirname(model_path))
    interpreter, input_details, output_details, model_size = load_model(model_path)

    # Set up camera
    crop_box = load_crop_config()
    if crop_box:
        print(f"✂️  Crop region: {crop_box}")
    camera = start_camera()

    # Set up relay
    relay = Relay(RELAY_GPIO_PIN, RELAY_PULSE_SECONDS, enabled=not args.disarmed)

    # Track detection events
    tracker = EventTracker(cooldown_seconds=cooldown)
    frame_count = 0

    # Clean shutdown on Ctrl+C
    def shutdown(sig, frame):
        log.info("Shutting down...")
        relay.cleanup()
        camera.stop()
        camera.close()
        log.info("Goodbye!")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    mode = "DISARMED 👀" if args.disarmed else "ARMED 💨"
    log.info("Started [%s] — threshold: %.0f%%, cooldown: %ss",
             mode, shoo_threshold * 100, cooldown)
    print(f"   Press Ctrl+C to stop.\n")

    # === THE LOOP ===
    while True:
        frame_count += 1
        img = capture_frame(camera, crop_box)
        detections = detect_from_image(
            img, interpreter, input_details, output_details, model_size, labels
        )

        if not detections:
            tracker.clear()
            # Print a dot every 10 frames so we know it's alive
            if frame_count % 10 == 0:
                print(".", end="", flush=True)
            time.sleep(FRAME_INTERVAL)
            continue

        # Something detected — check if it's a new event
        now = time.time()
        result = tracker.update(detections, now)

        if not result["is_new_event"]:
            time.sleep(FRAME_INTERVAL)
            continue

        # New event! Log and act on it.
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary = ", ".join(f"{d['label']} {d['score']:.0%}" for d in detections)
        log.info("Detected: %s", summary)
        save_snapshot(img, detections, timestamp, log)

        if result["should_shoo"]:
            relay.shoo()
            log.info("Shooed a squirrel!")
        elif result["cooldown_remaining"] is not None:
            log.info("Cooldown: %.1fs remaining", result["cooldown_remaining"])
        elif any(d["label"] == "bird" for d in detections):
            log.info("Bird visiting — welcome!")
        elif any(d["label"] == "raccoon" for d in detections):
            log.info("Raccoon spotted — logging only")

        time.sleep(FRAME_INTERVAL)


if __name__ == "__main__":
    main()
