"""
📷 Capture an image from the Pi Camera Module v3.

Usage:
    python capture.py                      # capture and crop (if configured)
    python capture.py -o my_photo.jpg      # save to specific path
    python capture.py --calibrate          # capture a grid image to help set crop region
    python capture.py --set-crop 400 200 1600 1200  # set crop to (left, top, right, bottom)
    python capture.py --clear-crop         # remove crop config
    python capture.py --no-crop            # capture without cropping even if configured
"""

import argparse
import json
import os
import time
from datetime import datetime

from picamera2 import Picamera2
from PIL import Image, ImageDraw, ImageFont

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "camera_config.json")


def load_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH) as f:
            return json.load(f)
    return {}


def save_config(config):
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)
    print(f"💾 Config saved: {CONFIG_PATH}")


def capture_raw():
    """Capture a full-resolution image and return it as a PIL Image."""
    from libcamera import controls as libcam_controls
    camera = Picamera2()
    config = camera.create_still_configuration()
    camera.configure(config)
    camera.start()

    # Apply saved focus setting
    cam_config = load_config()
    lens_position = cam_config.get("lens_position", None)
    if lens_position is not None and lens_position >= 0:
        camera.set_controls({
            "AfMode": libcam_controls.AfModeEnum.Manual,
            "LensPosition": lens_position,
        })
    elif lens_position is not None and lens_position < 0:
        camera.set_controls({"AfMode": libcam_controls.AfModeEnum.Auto})

    time.sleep(2)
    img = camera.capture_image("main")
    camera.stop()
    camera.close()
    return img


def crop_image(img, crop_box):
    """Crop a PIL Image to (left, top, right, bottom)."""
    return img.crop(crop_box)


def capture(output_path=None, apply_crop=True):
    if output_path is None:
        os.makedirs("captures", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join("captures", f"{timestamp}.jpg")

    img = capture_raw()

    config = load_config()
    if apply_crop and "crop" in config:
        box = config["crop"]
        crop_box = (box["left"], box["top"], box["right"], box["bottom"])
        img = crop_image(img, crop_box)
        print(f"✂️  Cropped to: {crop_box}")

    img.save(output_path)
    print(f"📷 Saved: {output_path}")
    return output_path


def calibrate():
    """Capture a full image with a grid overlay to help pick crop coordinates."""
    os.makedirs("captures", exist_ok=True)
    img = capture_raw()
    w, h = img.size

    draw = ImageDraw.Draw(img)
    step = 200

    for x in range(0, w, step):
        draw.line([(x, 0), (x, h)], fill=(255, 0, 0, 128), width=1)
        draw.text((x + 4, 4), str(x), fill=(255, 0, 0))

    for y in range(0, h, step):
        draw.line([(0, y), (w, y)], fill=(255, 0, 0, 128), width=1)
        draw.text((4, y + 4), str(y), fill=(255, 0, 0))

    # Show current crop region if configured
    config = load_config()
    if "crop" in config:
        box = config["crop"]
        draw.rectangle(
            [box["left"], box["top"], box["right"], box["bottom"]],
            outline=(0, 255, 0), width=3,
        )
        draw.text((box["left"] + 4, box["top"] + 4), "current crop", fill=(0, 255, 0))

    output_path = os.path.join("captures", "calibration.jpg")
    img.save(output_path)
    print(f"📷 Calibration image saved: {output_path}")
    print(f"   Image size: {w}x{h}")
    print(f"   Grid lines every {step}px with coordinates labeled.")
    print()
    print("   View this image, then set your crop region:")
    print(f"   python capture.py --set-crop LEFT TOP RIGHT BOTTOM")
    print(f"   Example: python capture.py --set-crop 400 200 1600 1200")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="📷 Capture from Pi Camera")
    parser.add_argument("-o", "--output", help="Output file path")
    parser.add_argument("--calibrate", action="store_true", help="Capture a grid image for crop calibration")
    parser.add_argument("--set-crop", nargs=4, type=int, metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"),
                        help="Set the crop region in pixels")
    parser.add_argument("--clear-crop", action="store_true", help="Remove saved crop config")
    parser.add_argument("--no-crop", action="store_true", help="Capture without cropping")
    args = parser.parse_args()

    if args.calibrate:
        calibrate()
    elif args.set_crop:
        left, top, right, bottom = args.set_crop
        config = load_config()
        config["crop"] = {"left": left, "top": top, "right": right, "bottom": bottom}
        save_config(config)
        print(f"✂️  Crop region set to: ({left}, {top}, {right}, {bottom})")
    elif args.clear_crop:
        config = load_config()
        config.pop("crop", None)
        save_config(config)
        print("✂️  Crop region cleared.")
    else:
        capture(args.output, apply_crop=not args.no_crop)
