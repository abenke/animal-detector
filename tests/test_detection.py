"""
Test suite for the animal detector using the BSR model.

Each test validates that the model correctly identifies animals
in the test images with expected labels and confidence ranges.
"""

import pytest
from conftest import run_detection


# --- Squirrel detections ---

class TestSquirrels:
    def test_squirrel_at_feeder(self, bsr_model):
        """animal2.png: squirrel reaching for bird feeder."""
        detections = run_detection(bsr_model, "animal2.png")
        squirrels = [d for d in detections if d["label"] == "squirrel"]
        assert len(squirrels) >= 1
        assert squirrels[0]["score"] >= 0.75

    def test_squirrel_hanging(self, bsr_model):
        """animal3.png: squirrel hanging off bird feeder."""
        detections = run_detection(bsr_model, "animal3.png")
        squirrels = [d for d in detections if d["label"] == "squirrel"]
        assert len(squirrels) >= 1
        assert squirrels[0]["score"] >= 0.80


# --- Bird detections ---

class TestBirds:
    def test_two_finches(self, bsr_model):
        """animal4.png: two house finches on a feeder."""
        detections = run_detection(bsr_model, "animal4.png")
        birds = [d for d in detections if d["label"] == "bird"]
        assert len(birds) >= 2
        assert birds[0]["score"] >= 0.40

    def test_cardinal(self, bsr_model):
        """animal5.png: Northern Cardinal on feeder."""
        detections = run_detection(bsr_model, "animal5.png")
        birds = [d for d in detections if d["label"] == "bird"]
        assert len(birds) >= 1
        assert birds[0]["score"] >= 0.80

    def test_blue_jay(self, bsr_model):
        """animal6.png: Blue Jay at suet feeder."""
        detections = run_detection(bsr_model, "animal6.png")
        birds = [d for d in detections if d["label"] == "bird"]
        assert len(birds) >= 1
        assert birds[0]["score"] >= 0.75

    def test_great_tit(self, bsr_model):
        """animal7.png: Great Tit on seed feeder."""
        detections = run_detection(bsr_model, "animal7.png")
        birds = [d for d in detections if d["label"] == "bird"]
        assert len(birds) >= 1
        assert birds[0]["score"] >= 0.70

    def test_multiple_sparrows(self, bsr_model):
        """animal8.png: group of sparrows on platform feeder."""
        detections = run_detection(bsr_model, "animal8.png")
        birds = [d for d in detections if d["label"] == "bird"]
        assert len(birds) >= 2
        assert birds[0]["score"] >= 0.55

    def test_marsh_tit(self, bsr_model):
        """animal9.png: small bird next to peanut feeder."""
        detections = run_detection(bsr_model, "animal9.png")
        birds = [d for d in detections if d["label"] == "bird"]
        assert len(birds) >= 1
        assert birds[0]["score"] >= 0.80

    def test_chicken(self, bsr_model):
        """animal10.png: chicken on railing (BSR sees as bird)."""
        detections = run_detection(bsr_model, "animal10.png")
        birds = [d for d in detections if d["label"] == "bird"]
        assert len(birds) >= 1
        assert birds[0]["score"] >= 0.50

    def test_robins(self, bsr_model):
        """animal11.png: two American Robins on railing."""
        detections = run_detection(bsr_model, "animal11.png")
        birds = [d for d in detections if d["label"] == "bird"]
        assert len(birds) >= 1
        assert birds[0]["score"] >= 0.85

    def test_woodpecker(self, bsr_model):
        """animal12.png: Downy Woodpecker on feeder."""
        detections = run_detection(bsr_model, "animal12.png")
        birds = [d for d in detections if d["label"] == "bird"]
        assert len(birds) >= 1
        assert birds[0]["score"] >= 0.60


# --- Negative test ---

class TestNegative:
    def test_no_animals_in_screenshot(self, bsr_model):
        """animal1.png: screenshot of a web page, no animals."""
        detections = run_detection(bsr_model, "animal1.png")
        animals = [d for d in detections if d["is_animal"]]
        assert len(animals) == 0


# --- Action mapping ---

class TestActions:
    def test_squirrel_triggers_servo(self, bsr_model):
        """Squirrel detection should map to servo activation."""
        from detect_animals import get_action
        detections = run_detection(bsr_model, "animal2.png")
        squirrels = [d for d in detections if d["label"] == "squirrel"]
        assert len(squirrels) >= 1
        action = get_action(squirrels[0]["label"])
        assert "SERVO" in action

    def test_bird_logged(self, bsr_model):
        """Bird detection should map to photo logging."""
        from detect_animals import get_action
        detections = run_detection(bsr_model, "animal5.png")
        birds = [d for d in detections if d["label"] == "bird"]
        assert len(birds) >= 1
        action = get_action(birds[0]["label"])
        assert "Log" in action or "welcome" in action.lower()
