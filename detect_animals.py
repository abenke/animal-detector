"""
🐿️ Animal Detector v2 - by Troy (age 9) and Dad
==================================================
Now with SQUIRREL detection! Uses the BirdSquirrelRaccoon
TFLite model by EdjeElectronics, or the general COCO model.

How to use:
    python3 detect_animals.py photo.jpg
    python3 detect_animals.py my_photos/            (scans a whole folder!)
    python3 detect_animals.py --model coco photo.jpg (use general model)
"""

import sys
import os
import argparse

# --- Check that everything is installed ---
try:
    import numpy as np
    from PIL import Image, ImageDraw
except ImportError:
    print("\n🔧 First time setup needed! Run this command:\n")
    print("   pip3 install numpy Pillow tensorflow\n")
    sys.exit(1)

try:
    from ai_edge_litert.interpreter import Interpreter as tflite_Interpreter
    TFLITE_BACKEND = "litert"
except ImportError:
    try:
        import tflite_runtime.interpreter as tflite
        tflite_Interpreter = tflite.Interpreter
        TFLITE_BACKEND = "tflite_runtime"
    except ImportError:
        try:
            import tensorflow as tf
            tflite_Interpreter = tf.lite.Interpreter
            TFLITE_BACKEND = "tensorflow"
        except ImportError:
            print("\n🔧 Need a TFLite backend! Run one of:\n")
            print("   pip install ai-edge-litert        (recommended)")
            print("   pip install tflite-runtime")
            print("   pip install tensorflow\n")
            sys.exit(1)


# ============================================================
#  🎯 CONFIGURATION - Troy, edit this part!
# ============================================================

ANIMAL_ACTIONS = {
    "squirrel": "🐿️ SERVO ACTIVATE - shoo the squirrel!",
    "bird":     "🐦 Log photo - welcome visitor!",
    "raccoon":  "🦝 SERVO ACTIVATE + send alert!",
    "cat":      "😺 No action - friendly neighbor cat",
    "dog":      "🐕 No action - good boy!",
    "bear":     "🐻 ALERT ALERT ALERT!",
}

DEFAULT_ACTION = "📸 Unknown animal - log photo"

# How confident does the AI need to be? (0.0 to 1.0)
CONFIDENCE_THRESHOLD = 0.4


# ============================================================
#  Model setup
# ============================================================

def find_bsr_model():
    """Find the BirdSquirrelRaccoon model in common locations."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    possible_paths = [
        os.path.join(script_dir, "model_bsr", "detect.tflite"),
        os.path.join(script_dir, "model_bsr", "BirdSquirrelRaccoon_TFLite_model", "detect.tflite"),
        os.path.join(script_dir, "BirdSquirrelRaccoon_TFLite_model", "detect.tflite"),
        os.path.join(script_dir, "model", "detect.tflite"),
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None


def download_coco_model():
    """Download the SSD MobileNet COCO model if needed."""
    import urllib.request
    import zipfile
    import tempfile
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_coco")
    model_path = os.path.join(model_dir, "detect.tflite")
    if os.path.exists(model_path):
        return model_path
    os.makedirs(model_dir, exist_ok=True)
    url = (
        "https://storage.googleapis.com/download.tensorflow.org/models/"
        "tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip"
    )
    print("📥 Downloading COCO model (first time only, ~3 MB)...")
    try:
        zip_path = os.path.join(tempfile.gettempdir(), "coco_model.zip")
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(model_dir)
        os.remove(zip_path)
        print("   ✅ Download complete!\n")
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        sys.exit(1)
    return model_path


def load_bsr_labelmap(model_dir):
    """Load labels from the BSR model's labelmap.txt."""
    labelmap_path = os.path.join(model_dir, "labelmap.txt")
    if not os.path.exists(labelmap_path):
        return {0: "bird", 1: "squirrel", 2: "raccoon"}
    labels = {}
    idx = 0
    with open(labelmap_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line == '???':
                continue
            labels[idx] = line.lower()
            idx += 1
    return labels


COCO_LABELS = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
    5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
    10: "fire hydrant", 12: "stop sign", 13: "parking meter", 14: "bench",
    15: "bird", 16: "cat", 17: "dog", 18: "horse", 19: "sheep",
    20: "cow", 21: "elephant", 22: "bear", 23: "zebra", 24: "giraffe",
    26: "backpack", 27: "umbrella", 30: "handbag", 31: "tie",
    32: "suitcase", 33: "frisbee", 34: "skis", 35: "snowboard",
    36: "sports ball", 37: "kite", 38: "baseball bat", 39: "baseball glove",
    40: "skateboard", 41: "surfboard", 42: "tennis racket", 43: "bottle",
    45: "wine glass", 46: "cup", 47: "fork", 48: "knife", 49: "spoon",
    50: "bowl", 51: "banana", 52: "apple", 53: "sandwich", 54: "orange",
    55: "broccoli", 56: "carrot", 57: "hot dog", 58: "pizza", 59: "donut",
    60: "cake", 61: "chair", 62: "couch", 63: "potted plant", 64: "bed",
    66: "dining table", 69: "oven", 71: "sink", 72: "refrigerator",
    73: "book", 74: "clock", 75: "vase", 76: "scissors",
    77: "teddy bear", 78: "hair drier", 79: "toothbrush",
}
COCO_ANIMAL_IDS = {15, 16, 17, 18, 19, 20, 21, 22, 23, 24}


def load_model(model_path):
    """Load a TFLite model."""
    print(f"🧠 Loading AI model...")
    interpreter = tflite_Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    h, w = input_details[0]['shape'][1], input_details[0]['shape'][2]
    print(f"   Input size: {w}x{h} pixels")
    print(f"   ✅ Ready!\n")
    return interpreter, input_details, output_details, (w, h)


# ============================================================
#  Detection engine
# ============================================================

def _nms(boxes, scores, iou_threshold=0.5):
    """Simple non-maximum suppression. boxes are [ymin, xmin, ymax, xmax]."""
    if len(boxes) == 0:
        return []
    order = scores.argsort()[::-1]
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        rest = order[1:]
        yy1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        xx1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        yy2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        xx2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.maximum(0, yy2 - yy1) * np.maximum(0, xx2 - xx1)
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_rest = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        iou = inter / (area_i + area_rest - inter + 1e-6)
        order = rest[iou < iou_threshold]
    return keep


def detect_in_image(image_path, interpreter, input_details, output_details,
                    model_size, labels, is_bsr=False):
    """Run detection on a single image."""
    width, height = model_size
    img = Image.open(image_path).convert("RGB")
    original_size = img.size
    img_resized = img.resize((width, height))

    input_data = np.expand_dims(np.array(img_resized, dtype=np.uint8), axis=0)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    num_outputs = len(output_details)

    if num_outputs >= 4:
        # Classic TFLite detection model with built-in NMS
        boxes = interpreter.get_tensor(output_details[0]['index'])[0]
        classes = interpreter.get_tensor(output_details[1]['index'])[0]
        scores = interpreter.get_tensor(output_details[2]['index'])[0]
        count = int(interpreter.get_tensor(output_details[3]['index'])[0])
    else:
        # MediaPipe-style raw output: class_scores + boxes, needs NMS
        raw = {}
        for d in output_details:
            tensor = interpreter.get_tensor(d['index'])[0]
            raw[tensor.shape[-1]] = tensor
        raw_scores = raw[90]   # [num_anchors, 90] class scores
        raw_boxes = raw[4]     # [num_anchors, 4]  box coords

        # Find best class per anchor and filter by threshold
        best_class = np.argmax(raw_scores, axis=1)
        best_score = np.max(raw_scores, axis=1)
        mask = best_score >= CONFIDENCE_THRESHOLD
        if np.any(mask):
            filt_boxes = raw_boxes[mask]
            filt_scores = best_score[mask]
            filt_classes = best_class[mask]
            keep = _nms(filt_boxes, filt_scores)
            boxes = filt_boxes[keep]
            scores = filt_scores[keep]
            classes = filt_classes[keep]
            count = len(keep)
        else:
            boxes = np.empty((0, 4))
            scores = np.empty((0,))
            classes = np.empty((0,))
            count = 0

    detections = []
    for i in range(min(count, len(scores))):
        score = float(scores[i])
        if score < CONFIDENCE_THRESHOLD:
            continue

        class_id = int(classes[i])
        label = labels.get(class_id, f"class_{class_id}")

        ymin, xmin, ymax, xmax = boxes[i]
        box = {
            'xmin': float(xmin) * original_size[0],
            'ymin': float(ymin) * original_size[1],
            'xmax': float(xmax) * original_size[0],
            'ymax': float(ymax) * original_size[1],
        }

        is_animal = True if is_bsr else (class_id in COCO_ANIMAL_IDS)

        detections.append({
            'label': label, 'class_id': class_id,
            'score': score, 'box': box, 'is_animal': is_animal,
        })

    return detections, img


def get_action(label):
    """Decide what the servo should do."""
    return ANIMAL_ACTIONS.get(label.lower(), DEFAULT_ACTION)


def draw_detections(img, detections, output_path):
    """Draw boxes around detected objects and save."""
    draw = ImageDraw.Draw(img)
    colors = {
        'squirrel': (255, 80, 0),
        'bird':     (0, 180, 0),
        'raccoon':  (180, 0, 180),
        'cat':      (0, 150, 255),
        'dog':      (255, 200, 0),
    }

    for det in detections:
        box = det['box']
        label = det['label']
        color = colors.get(label.lower(), (0, 200, 0) if det['is_animal'] else (150, 150, 150))

        draw.rectangle(
            [box['xmin'], box['ymin'], box['xmax'], box['ymax']],
            outline=color, width=3
        )

        text = f"{label} {det['score']:.0%}"
        text_y = max(box['ymin'] - 22, 0)
        text_bbox = draw.textbbox((box['xmin'] + 4, text_y), text)
        draw.rectangle(
            [text_bbox[0] - 2, text_bbox[1] - 2, text_bbox[2] + 2, text_bbox[3] + 2],
            fill=color
        )
        draw.text((box['xmin'] + 4, text_y), text, fill=(255, 255, 255))

    img.save(output_path)


def process_image(image_path, interpreter, input_details, output_details,
                  model_size, labels, is_bsr):
    """Process one image: detect, report, save."""
    filename = os.path.basename(image_path)
    print(f"{'='*60}")
    print(f"📷 Scanning: {filename}")
    print(f"{'='*60}")

    detections, img = detect_in_image(
        image_path, interpreter, input_details, output_details,
        model_size, labels, is_bsr
    )

    if not detections:
        print("   Nothing detected. Try lowering --threshold.\n")
        return 0

    animals = [d for d in detections if d['is_animal']]
    others = [d for d in detections if not d['is_animal']]

    if animals:
        print(f"\n   🎯 ANIMALS FOUND: {len(animals)}")
        print(f"   {'─'*40}")
        for det in animals:
            action = get_action(det['label'])
            print(f"   🦁 {det['label']:12s}  confidence: {det['score']:.0%}")
            print(f"      Action → {action}")
        print()

    if others:
        print(f"   📦 Other objects: {', '.join(d['label'] for d in others)}\n")

    # Save annotated image
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)
    name, ext = os.path.splitext(filename)
    output_path = os.path.join(output_dir, f"{name}_detected{ext}")
    draw_detections(img, detections, output_path)
    print(f"   💾 Saved: output/{name}_detected{ext}\n")

    return len(animals)


# ============================================================
#  🚀 MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="🐿️ Animal Detector v2")
    parser.add_argument("images", nargs="*", help="Image files or folders to scan")
    parser.add_argument("--model", choices=["bsr", "coco"], default="bsr",
                        help="'bsr' = Bird/Squirrel/Raccoon (default), 'coco' = general")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Confidence threshold 0.0-1.0 (default: 0.4)")
    args = parser.parse_args()

    if args.threshold is not None:
        global CONFIDENCE_THRESHOLD
        CONFIDENCE_THRESHOLD = args.threshold

    print()
    print("╔══════════════════════════════════════════════╗")
    print("║   🐿️  ANIMAL DETECTOR v2 by Troy & Dad      ║")
    print("║   Now with SQUIRREL detection!              ║")
    print("╚══════════════════════════════════════════════╝")
    print()

    if not args.images:
        print("Usage:")
        print("   python3 detect_animals.py photo.jpg")
        print("   python3 detect_animals.py photos/")
        print("   python3 detect_animals.py --model coco photo.jpg")
        print("   python3 detect_animals.py --threshold 0.3 photo.jpg")
        print()
        print("Models:")
        print("   bsr  = Bird, Squirrel, Raccoon detector (default)")
        print("   coco = General detector (80 objects, no squirrels)")
        print()
        sys.exit(0)

    # Load model
    if args.model == "bsr":
        model_path = find_bsr_model()
        if model_path is None:
            print("❌ BirdSquirrelRaccoon model not found!\n")
            print("   Download it:")
            print("   curl -L 'https://www.dropbox.com/s/cpaon1j1r1yzflx/BirdSquirrelRaccoon_TFLite_model.zip?dl=1' -o bsr_model.zip")
            print("   unzip bsr_model.zip -d model_bsr\n")
            print("   Or use COCO instead:  python3 detect_animals.py --model coco photo.jpg\n")
            sys.exit(1)
        labels = load_bsr_labelmap(os.path.dirname(model_path))
        is_bsr = True
        print(f"🔬 Model: BirdSquirrelRaccoon ({', '.join(labels.values())})")
    else:
        model_path = download_coco_model()
        labels = COCO_LABELS
        is_bsr = False
        print(f"🔬 Model: COCO EfficientDet (80 object types)")

    print(f"📊 Confidence threshold: {CONFIDENCE_THRESHOLD:.0%}\n")

    interpreter, input_details, output_details, model_size = load_model(model_path)

    # Gather images
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    image_files = []
    for arg in args.images:
        if os.path.isdir(arg):
            for f in sorted(os.listdir(arg)):
                if os.path.splitext(f)[1].lower() in image_extensions:
                    image_files.append(os.path.join(arg, f))
        elif os.path.isfile(arg):
            image_files.append(arg)
        else:
            print(f"⚠️  Not found: {arg}")

    if not image_files:
        print("❌ No image files found!")
        sys.exit(1)

    print(f"Found {len(image_files)} image(s) to scan.\n")

    total_animals = 0
    for image_path in image_files:
        try:
            total_animals += process_image(
                image_path, interpreter, input_details, output_details,
                model_size, labels, is_bsr
            )
        except Exception as e:
            print(f"❌ Error processing {image_path}: {e}\n")

    print(f"{'='*60}")
    print(f"✅ Done! Scanned {len(image_files)} image(s), found {total_animals} animal(s).")
    if total_animals > 0:
        print(f"   Check the 'output' folder for annotated images!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()