"""Tests for :mod:`scheduler.data`."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest

from scheduler.data import ConceptState, ReviewEvent, UserState


# ---------------------------------------------------------------------------
# ReviewEvent
# ---------------------------------------------------------------------------

class TestReviewEvent:
    def test_valid(self):
        ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
        ev = ReviewEvent(timestamp=ts, score=0.8, elapsed_days=3.0)
        assert ev.score == 0.8
        assert ev.elapsed_days == 3.0

    def test_score_boundary_zero(self):
        ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
        ev = ReviewEvent(timestamp=ts, score=0.0, elapsed_days=0.0)
        assert ev.score == 0.0

    def test_score_boundary_one(self):
        ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
        ev = ReviewEvent(timestamp=ts, score=1.0, elapsed_days=0.0)
        assert ev.score == 1.0

    def test_invalid_score_high(self):
        ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
        with pytest.raises(ValueError, match="score"):
            ReviewEvent(timestamp=ts, score=1.5, elapsed_days=0.0)

    def test_invalid_score_negative(self):
        ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
        with pytest.raises(ValueError, match="score"):
            ReviewEvent(timestamp=ts, score=-0.1, elapsed_days=0.0)

    def test_invalid_elapsed_days_negative(self):
        ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
        with pytest.raises(ValueError, match="elapsed_days"):
            ReviewEvent(timestamp=ts, score=0.5, elapsed_days=-1.0)


# ---------------------------------------------------------------------------
# ConceptState
# ---------------------------------------------------------------------------

class TestConceptState:
    def _make_concept(self) -> ConceptState:
        cs = ConceptState(concept_id="c1")
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        cs.add_review(score=0.9, timestamp=base)
        cs.add_review(score=0.7, timestamp=base + timedelta(days=5))
        cs.add_review(score=0.8, timestamp=base + timedelta(days=12))
        return cs

    def test_add_review_count(self):
        cs = self._make_concept()
        assert cs.review_count == 3

    def test_add_review_elapsed_days(self):
        cs = self._make_concept()
        assert cs.reviews[0].elapsed_days == 0.0
        assert cs.reviews[1].elapsed_days == pytest.approx(5.0, abs=1e-4)
        assert cs.reviews[2].elapsed_days == pytest.approx(7.0, abs=1e-4)

    def test_last_review(self):
        cs = self._make_concept()
        assert cs.last_review is not None
        assert cs.last_review.score == 0.8

    def test_last_review_empty(self):
        cs = ConceptState(concept_id="empty")
        assert cs.last_review is None

    def test_feature_sequence(self):
        cs = self._make_concept()
        seq = cs.as_feature_sequence()
        assert len(seq) == 3
        # Each feature vector now has 5 elements
        assert len(seq[0]) == 5
        assert seq[0][0] == 0.0     # elapsed_days
        assert seq[0][1] == 0.9     # score
        assert seq[1][0] == pytest.approx(5.0, abs=1e-4)

    def test_stability_estimate_no_reviews(self):
        cs = ConceptState(concept_id="c0")
        assert cs.stability_estimate() == 1.0

    def test_stability_estimate_with_reviews(self):
        cs = self._make_concept()
        stab = cs.stability_estimate()
        assert stab > 0  # should be a positive number

    # ---- New FSRS-inspired state tests ----

    def test_difficulty_initialised(self):
        cs = ConceptState(concept_id="c0")
        assert cs.difficulty == pytest.approx(0.3)

    def test_stability_initialised(self):
        cs = ConceptState(concept_id="c0")
        assert cs.stability == pytest.approx(1.0)

    def test_difficulty_decreases_on_high_score(self):
        cs = ConceptState(concept_id="c0")
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        cs.add_review(score=1.0, timestamp=base)
        assert cs.difficulty < 0.3

    def test_difficulty_increases_on_low_score(self):
        cs = ConceptState(concept_id="c0")
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        cs.add_review(score=0.0, timestamp=base)
        assert cs.difficulty > 0.3

    def test_stability_grows_on_success(self):
        cs = ConceptState(concept_id="c0")
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        cs.add_review(score=0.9, timestamp=base)
        cs.add_review(score=0.9, timestamp=base + timedelta(days=5))
        assert cs.stability > 1.0

    def test_stability_shrinks_on_failure(self):
        cs = ConceptState(concept_id="c0")
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        cs.add_review(score=0.9, timestamp=base)
        s_after_success = cs.stability
        cs.add_review(score=0.2, timestamp=base + timedelta(days=5))
        assert cs.stability < s_after_success

    def test_retrievability_power_law(self):
        cs = ConceptState(concept_id="c0", stability=10.0)
        # R(t) = (1 + t/S)^(-1)
        assert cs.retrievability(0.0) == pytest.approx(1.0)
        assert cs.retrievability(10.0) == pytest.approx(0.5)
        assert cs.retrievability(30.0) == pytest.approx(0.25)

    def test_category_field(self):
        cs = ConceptState(concept_id="c0", category="math")
        assert cs.category == "math"

    def test_feature_sequence_contains_difficulty_and_stability(self):
        cs = ConceptState(concept_id="c0")
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        cs.add_review(score=0.8, timestamp=base)
        seq = cs.as_feature_sequence()
        # [elapsed_days, score, difficulty, stability, norm_review_count]
        assert len(seq[0]) == 5
        assert seq[0][2] == pytest.approx(0.3)   # initial difficulty
        assert seq[0][3] == pytest.approx(1.0)    # initial stability
        assert seq[0][4] == pytest.approx(1.0)    # norm review count (1/1)


# ---------------------------------------------------------------------------
# UserState
# ---------------------------------------------------------------------------

class TestUserState:
    def test_get_or_create_concept(self):
        user = UserState(user_id="u1")
        cs = user.get_or_create_concept("c1")
        assert isinstance(cs, ConceptState)
        # Second call returns the same object
        assert user.get_or_create_concept("c1") is cs

    def test_concept_ids(self):
        user = UserState(user_id="u1")
        user.get_or_create_concept("c1")
        user.get_or_create_concept("c2")
        assert set(user.concept_ids) == {"c1", "c2"}

    def test_total_reviews(self):
        user = UserState(user_id="u1")
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        cs1 = user.get_or_create_concept("c1")
        cs1.add_review(0.9, base)
        cs2 = user.get_or_create_concept("c2")
        cs2.add_review(0.8, base)
        cs2.add_review(0.7, base + timedelta(days=3))
        assert user.total_reviews() == 3

    def test_get_or_create_concept_with_category(self):
        user = UserState(user_id="u1")
        cs = user.get_or_create_concept("c1", category="math")
        assert cs.category == "math"
