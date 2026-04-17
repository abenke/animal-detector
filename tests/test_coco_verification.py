"""
Tests for COCO verification — the second-pass model that filters
BSR false positives by confirming an actual animal is present.

Uses real detection snapshots from production:
- *-correct.jpg: real birds (COCO should confirm animal present)
- *-incorrect.jpg: false positives on foliage/people (COCO should reject)
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detect_animals import download_coco_model, load_model, COCO_LABELS, COCO_ANIMAL_IDS
from squirrel_defense import verify_with_coco
from PIL import Image

IMAGES_DIR = os.path.join(os.path.dirname(__file__), "images")


@pytest.fixture(scope="session")
def coco_model():
    """Load the COCO model once for all tests."""
    coco_path = download_coco_model()
    interpreter, input_details, output_details, model_size = load_model(coco_path)
    return interpreter, input_details, output_details, model_size


def run_coco(coco_model, image_name):
    interpreter, input_details, output_details, model_size = coco_model
    img = Image.open(os.path.join(IMAGES_DIR, image_name)).convert("RGB")
    return verify_with_coco(img, interpreter, input_details, output_details, model_size)


class TestCocoConfirmsTruePositives:
    def test_bird_at_feeder_1(self, coco_model):
        """Real bird in foliage — COCO should confirm animal."""
        assert run_coco(coco_model, "live_20260417_065554_bird-correct.jpg") is True

    def test_bird_at_feeder_2(self, coco_model):
        """Real bird, 92% BSR confidence — COCO should confirm."""
        assert run_coco(coco_model, "live_20260417_074252_bird-correct.jpg") is True

    def test_bird_at_feeder_3(self, coco_model):
        """Brown bird below foliage — COCO should confirm."""
        assert run_coco(coco_model, "live_20260417_082906_bird-correct.jpg") is True

    def test_bird_at_feeder_4(self, coco_model):
        """Clear bird below canopy — COCO should confirm."""
        assert run_coco(coco_model, "live_20260417_083133_bird-correct.jpg") is True


class TestCocoRejectsFalsePositives:
    def test_foliage_false_positive_1(self, coco_model):
        """Dense variegated foliage, no animal — COCO should reject."""
        assert run_coco(coco_model, "live_20260416_094435_bird-incorrect.jpg") is False

    def test_foliage_false_positive_2(self, coco_model):
        """Variegated shrub, no animal — COCO should reject."""
        assert run_coco(coco_model, "live_20260417_074131_bird-incorrect.jpg") is False

    def test_foliage_false_positive_3(self, coco_model):
        """Same bush, different time — COCO should reject."""
        assert run_coco(coco_model, "live_20260417_074715_bird-incorrect.jpg") is False

    def test_foliage_false_positive_4(self, coco_model):
        """Foliage patterns again — COCO should reject."""
        assert run_coco(coco_model, "live_20260417_074915_bird-incorrect.jpg") is False

    def test_person_false_positive(self, coco_model):
        """Person detected as bird — COCO should reject (not an animal)."""
        assert run_coco(coco_model, "live_20260414_204856_bird-incorrect.jpg") is False
