"""
Tests for EventTracker — the deduplication logic that decides
when to log, snapshot, and shoo based on detection state changes.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from squirrel_defense import EventTracker


def bird(score=0.80):
    return {"label": "bird", "score": score}

def squirrel(score=0.75):
    return {"label": "squirrel", "score": score}

def raccoon(score=0.70):
    return {"label": "raccoon", "score": score}


class TestNewEvents:
    def test_first_detection_is_new(self):
        tracker = EventTracker()
        result = tracker.update([bird()], now=100)
        assert result["is_new_event"] is True

    def test_same_bird_next_frame_is_not_new(self):
        tracker = EventTracker()
        tracker.update([bird()], now=100)
        result = tracker.update([bird()], now=100.5)
        assert result["is_new_event"] is False

    def test_bird_then_squirrel_is_new(self):
        tracker = EventTracker()
        tracker.update([bird()], now=100)
        result = tracker.update([squirrel()], now=100.5)
        assert result["is_new_event"] is True

    def test_bird_then_nothing_then_bird_is_new(self):
        """Same animal returning after a quiet period is a new event."""
        tracker = EventTracker(quiet_seconds=30)
        tracker.update([bird()], now=100)
        tracker.clear()  # no detections for a while
        # Simulate 31 seconds of nothing (last_detection_time was 100)
        result = tracker.update([bird()], now=131)
        assert result["is_new_event"] is True

    def test_bird_then_brief_gap_then_bird_is_not_new(self):
        """Same animal returning after a short gap is not a new event."""
        tracker = EventTracker(quiet_seconds=30)
        tracker.update([bird()], now=100)
        tracker.clear()
        result = tracker.update([bird()], now=105)
        assert result["is_new_event"] is False

    def test_nothing_detected_resets_labels(self):
        tracker = EventTracker()
        tracker.update([bird()], now=100)
        tracker.clear()
        # After clear, any detection is new (if enough time passed)
        result = tracker.update([bird()], now=200)
        assert result["is_new_event"] is True

    def test_two_birds_then_one_bird_is_new(self):
        """Change in set of labels counts as new (bird+raccoon → bird)."""
        tracker = EventTracker()
        tracker.update([bird(), raccoon()], now=100)
        result = tracker.update([bird()], now=100.5)
        assert result["is_new_event"] is True


class TestShooing:
    def test_squirrel_shoos_on_first_detection(self):
        tracker = EventTracker(cooldown_seconds=8)
        result = tracker.update([squirrel()], now=100)
        assert result["should_shoo"] is True
        assert result["cooldown_remaining"] is None

    def test_squirrel_cooldown_prevents_second_shoo(self):
        tracker = EventTracker(cooldown_seconds=8)
        tracker.update([squirrel()], now=100)
        result = tracker.update([squirrel()], now=103)
        assert result["should_shoo"] is False
        assert result["cooldown_remaining"] is not None
        assert 4.9 <= result["cooldown_remaining"] <= 5.1

    def test_squirrel_shoos_again_after_cooldown(self):
        tracker = EventTracker(cooldown_seconds=8)
        tracker.update([squirrel()], now=100)
        result = tracker.update([squirrel()], now=109)
        assert result["should_shoo"] is True

    def test_bird_does_not_shoo(self):
        tracker = EventTracker()
        result = tracker.update([bird()], now=100)
        assert result["should_shoo"] is False

    def test_raccoon_does_not_shoo(self):
        tracker = EventTracker()
        result = tracker.update([raccoon()], now=100)
        assert result["should_shoo"] is False

    def test_squirrel_shoo_is_always_logged_as_event(self):
        """Even if labels haven't changed, a shoo is a new event."""
        tracker = EventTracker(cooldown_seconds=8)
        tracker.update([squirrel()], now=100)
        # Same squirrel, within quiet_seconds, but cooldown elapsed
        result = tracker.update([squirrel()], now=109)
        assert result["is_new_event"] is True
        assert result["should_shoo"] is True


class TestCooldownEdgeCases:
    def test_cooldown_exact_boundary(self):
        tracker = EventTracker(cooldown_seconds=8)
        tracker.update([squirrel()], now=100)
        result = tracker.update([squirrel()], now=108)
        assert result["should_shoo"] is True

    def test_cooldown_just_before_boundary(self):
        tracker = EventTracker(cooldown_seconds=8)
        tracker.update([squirrel()], now=100)
        result = tracker.update([squirrel()], now=107.9)
        assert result["should_shoo"] is False

    def test_multiple_shoos_track_last_shoo_time(self):
        tracker = EventTracker(cooldown_seconds=8)
        tracker.update([squirrel()], now=100)  # shoos
        tracker.update([squirrel()], now=109)  # shoos (9s later)
        result = tracker.update([squirrel()], now=112)  # 3s after second shoo
        assert result["should_shoo"] is False
        assert 4.9 <= result["cooldown_remaining"] <= 5.1
