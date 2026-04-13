"""
🔍 Focus Calibration — capture images at different focus distances.

Saves a series of photos so you can view them on your Mac and pick
the sharpest one. Then use the winning lens position in your config.

Usage:
    python calibrate_focus.py                # sweep from 0.5m to infinity
    python calibrate_focus.py --min 1 --max 5  # sweep 1m to 5m range
    python calibrate_focus.py --distance 3     # single shot at 3 meters
"""

import argparse
import os
import time

from picamera2 import Picamera2
from libcamera import controls

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "captures", "focus")


def capture_at_position(camera, lens_position, output_dir):
    """Capture one image at a specific lens position."""
    camera.set_controls({"AfMode": controls.AfModeEnum.Manual, "LensPosition": lens_position})
    # Flush frames so the lens physically moves and new settings take effect
    for _ in range(10):
        camera.capture_array()
    time.sleep(1)

    if lens_position > 0:
        distance_m = 1.0 / lens_position
        label = f"{distance_m:.1f}m"
    else:
        label = "inf"

    filename = f"focus_{label}_lp{lens_position:.2f}.jpg"
    path = os.path.join(output_dir, filename)
    camera.capture_file(path)
    print(f"  📷 {filename}  (lens position: {lens_position:.2f}, ~{label})")
    return path


def sweep(min_distance, max_distance, steps=10):
    """Capture images across a range of focus distances."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    camera = Picamera2()
    config = camera.create_still_configuration()
    camera.configure(config)
    camera.start()
    time.sleep(2)

    # Convert distances to lens positions (dioptres = 1/distance)
    # Larger lens_position = closer focus
    lp_max = 1.0 / min_distance   # closest
    lp_min = 1.0 / max_distance   # farthest
    step_size = (lp_max - lp_min) / (steps - 1)

    print(f"\n🔍 Focus sweep: {min_distance}m to {max_distance}m ({steps} steps)")
    print(f"   Saving to: captures/focus/\n")

    positions = [lp_min + i * step_size for i in range(steps)]
    # Also include infinity (0.0) and autofocus
    positions = [0.0] + positions

    for lp in positions:
        capture_at_position(camera, lp, OUTPUT_DIR)

    # One more with autofocus for comparison
    print(f"\n  📷 focus_auto.jpg  (autofocus)")
    camera.set_controls({"AfMode": controls.AfModeEnum.Auto})
    camera.autofocus_cycle()
    time.sleep(2)
    camera.capture_file(os.path.join(OUTPUT_DIR, "focus_auto.jpg"))

    camera.stop()
    camera.close()

    print(f"\n✅ Done! {steps + 2} images saved to captures/focus/")
    print(f"   Copy to your Mac:")
    print(f'   scp "ttboss@squirrel-defense.local:~/animal-detector/captures/focus/*.jpg" ~/Desktop/')
    print(f"\n   Pick the sharpest one, then set it:")
    print(f"   python calibrate_focus.py --set-focus LENS_POSITION")


def single_shot(distance):
    """Capture one image at a specific distance."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    camera = Picamera2()
    config = camera.create_still_configuration()
    camera.configure(config)
    camera.start()
    time.sleep(2)

    lp = 1.0 / distance
    print(f"\n🔍 Single capture at ~{distance}m (lens position: {lp:.2f})")
    capture_at_position(camera, lp, OUTPUT_DIR)

    camera.stop()
    camera.close()


def set_focus(lens_position):
    """Save the chosen lens position to camera_config.json."""
    import json
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "camera_config.json")

    config = {}
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)

    config["lens_position"] = lens_position
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    if lens_position > 0:
        distance = 1.0 / lens_position
        print(f"🔍 Focus set to lens position {lens_position:.2f} (~{distance:.1f}m)")
    elif lens_position == 0:
        print(f"🔍 Focus set to infinity")
    else:
        print(f"🔍 Focus set to autofocus")

    print(f"💾 Saved to {config_path}")
    print(f"   Restart the service: sudo systemctl restart squirrel-defense")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="🔍 Focus Calibration")
    parser.add_argument("--min", type=float, default=0.5,
                        help="Closest distance in meters (default: 0.5)")
    parser.add_argument("--max", type=float, default=10,
                        help="Farthest distance in meters (default: 10)")
    parser.add_argument("--steps", type=int, default=10,
                        help="Number of focus steps (default: 10)")
    parser.add_argument("--distance", type=float,
                        help="Capture a single image at this distance (meters)")
    parser.add_argument("--set-focus", type=float, metavar="LENS_POSITION",
                        help="Save a lens position to config (-1 for autofocus, 0 for infinity)")
    args = parser.parse_args()

    if args.set_focus is not None:
        set_focus(args.set_focus)
    elif args.distance:
        single_shot(args.distance)
    else:
        sweep(args.min, args.max, args.steps)
