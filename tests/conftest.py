import os
import sys
import pytest

# Add project root to path so we can import detect_animals
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detect_animals import (
    find_bsr_model,
    load_bsr_labelmap,
    load_model,
    detect_in_image,
)

IMAGES_DIR = os.path.join(os.path.dirname(__file__), "images")


@pytest.fixture(scope="session")
def bsr_model():
    """Load the BSR model once for all tests."""
    model_path = find_bsr_model()
    if model_path is None:
        pytest.skip("BSR model not installed — run: "
                     "curl -L 'https://www.dropbox.com/s/cpaon1j1r1yzflx/"
                     "BirdSquirrelRaccoon_TFLite_model.zip?dl=1' -o bsr_model.zip "
                     "&& unzip bsr_model.zip -d model_bsr")
    labels = load_bsr_labelmap(os.path.dirname(model_path))
    interpreter, input_details, output_details, model_size = load_model(model_path)
    return interpreter, input_details, output_details, model_size, labels


@pytest.fixture(scope="session")
def detect(bsr_model):
    """Return a callable that runs detection on a test image by name."""
    interpreter, input_details, output_details, model_size, labels = bsr_model

    def _detect(image_name):
        image_path = os.path.join(IMAGES_DIR, image_name)
        detections, _ = detect_in_image(
            image_path, interpreter, input_details, output_details,
            model_size, labels, is_bsr=True,
        )
        return detections

    return _detect
