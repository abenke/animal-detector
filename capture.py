"""
📷 Capture an image from the Pi Camera Module v3.

Usage:
    python capture.py                  # saves to captures/YYYYMMDD_HHMMSS.jpg
    python capture.py -o my_photo.jpg  # saves to specific path
"""

import argparse
import os
import time
from datetime import datetime

from picamera2 import Picamera2


def capture(output_path=None):
    if output_path is None:
        os.makedirs("captures", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join("captures", f"{timestamp}.jpg")

    camera = Picamera2()
    config = camera.create_still_configuration()
    camera.configure(config)
    camera.start()

    # Let the camera adjust exposure/white balance
    time.sleep(2)

    camera.capture_file(output_path)
    camera.stop()
    camera.close()

    print(f"📷 Saved: {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="📷 Capture from Pi Camera")
    parser.add_argument("-o", "--output", help="Output file path (default: captures/YYYYMMDD_HHMMSS.jpg)")
    args = parser.parse_args()
    capture(args.output)
