"""
💨 Manually pulse the relay to test the deterrent.

Usage:
    python shoo.py              # default 0.3s pulse on GPIO 18
    python shoo.py --duration 1 # 1-second pulse
    python shoo.py --pin 17     # different GPIO pin
"""

import argparse
import sys
import time

from gpiozero import OutputDevice


def shoo(pin=18, duration=0.3):
    print(f"💨 Pulsing GPIO {pin} for {duration}s...")
    relay = OutputDevice(pin)
    try:
        relay.on()
        time.sleep(duration)
        relay.off()
        print("✅ Done!")
    finally:
        relay.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="💨 Manual relay pulse")
    parser.add_argument("--pin", type=int, default=18, help="GPIO pin (default: 18)")
    parser.add_argument("--duration", type=float, default=0.3,
                        help="Pulse duration in seconds (default: 0.3)")
    args = parser.parse_args()

    try:
        shoo(args.pin, args.duration)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
