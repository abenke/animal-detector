# Animal Detector v2

By Troy (age 9) and Dad. Detects birds, squirrels, and raccoons in photos using TFLite models.

## Setup

### 1. Install Python dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install numpy Pillow tensorflow
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

### 3. (Optional) COCO model

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

## Test images

The `animal*.png` files in this repo can be used to verify the detector is working:

```bash
python3 detect_animals.py animal1.png animal2.png animal3.png
```
