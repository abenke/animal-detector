# Animal Detector v2

By Troy (age 9) and Dad. Detects birds, squirrels, and raccoons in photos using TFLite models.

## Setup

### 1. Install Python dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install numpy Pillow
```

Then install a TFLite backend (pick one):

```bash
pip install ai-edge-litert       # recommended, works on Pi + Python 3.13
pip install tflite-runtime        # alternative (no Python 3.13 wheels yet)
pip install tensorflow            # full TF, heavy (no Python 3.13 support yet)
```

### 2. Download the BirdSquirrelRaccoon (BSR) model

The BSR model is a custom TFLite model by [EdjeElectronics](https://github.com/EdjeElectronics) trained to detect birds, squirrels, and raccoons.

```bash
curl -L 'https://www.dropbox.com/s/cpaon1j1r1yzflx/BirdSquirrelRaccoon_TFLite_model.zip?dl=1' -o bsr_model.zip
unzip bsr_model.zip -d model_bsr
```

After unzipping, you should have:

```
model_bsr/
  BirdSquirrelRaccoon_TFLite_model/
    detect.tflite
    labelmap.txt
    ...
```

### 3. (Raspberry Pi only) Pi Camera

To use `capture.py` for taking photos with a Pi Camera Module:

```bash
pip install picamera2
```

### 4. (Optional) COCO model

The general-purpose COCO model downloads automatically on first use. It detects 80 object types but does not include squirrels.

## Usage

```bash
# Scan a single image (uses BSR model by default)
python3 detect_animals.py photo.jpg

# Scan a folder of images
python3 detect_animals.py my_photos/

# Use the general COCO model instead
python3 detect_animals.py --model coco photo.jpg

# Adjust confidence threshold (default: 0.4)
python3 detect_animals.py --threshold 0.3 photo.jpg
```

Annotated output images are saved to the `output/` folder.

## Running on the Raspberry Pi

### Squirrel Defense (main loop)

The main loop watches the camera, detects animals, and activates the deterrent when a squirrel is spotted:

```bash
python squirrel_defense.py              # armed mode
python squirrel_defense.py --disarmed   # watch-only (no shooing)
python squirrel_defense.py --threshold 0.6 --cooldown 10  # custom settings
```

### Run as a service (survives SSH disconnect, starts on boot)

```bash
sudo cp squirrel-defense.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable squirrel-defense   # start on boot
sudo systemctl start squirrel-defense    # start now
```

Manage the service:

```bash
sudo systemctl stop squirrel-defense     # stop
sudo systemctl start squirrel-defense    # start
sudo systemctl restart squirrel-defense  # restart (e.g. after code changes)
sudo systemctl status squirrel-defense   # check if running
journalctl -u squirrel-defense -f        # tail live logs
```

Logs are also saved to `logs/defense_YYYYMMDD.log` (one file per day).

### Web viewer (as a service)

```bash
sudo cp web-viewer.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable web-viewer
sudo systemctl start web-viewer
```

Then open `http://squirrel-defense.local:8080` from any device on the local network.

### Camera calibration

Capture a grid image to find the feeder coordinates, then set a crop region so the detector only looks at the feeder area:

```bash
python capture.py --calibrate            # save a grid image
python capture.py --set-crop LEFT TOP RIGHT BOTTOM
python capture.py                        # future captures auto-crop
```

## Testing

Test images live in `tests/images/`. To run the test suite:

```bash
pip install pytest
pytest tests/ -v
```

Tests verify that the BSR model correctly identifies birds, squirrels, and non-animal images with expected confidence levels. CI runs automatically on push and PR via GitHub Actions.
