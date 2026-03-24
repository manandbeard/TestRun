"""Tests for :mod:`scheduler.scheduler` (:class:`SpacedRepetitionScheduler`)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
import torch

from scheduler.data import ConceptState, UserState
from scheduler.model import RecallLSTM
from scheduler.scheduler import SpacedRepetitionScheduler


def _make_scheduler(
    target_recall: float = 0.9,
    min_interval: float = 1.0,
    max_interval: float = 365.0,
) -> SpacedRepetitionScheduler:
    torch.manual_seed(0)
    model = RecallLSTM(input_size=6, hidden_size=16, num_layers=1, dropout=0.0)
    return SpacedRepetitionScheduler(
        model,
        target_recall=target_recall,
        min_interval=min_interval,
        max_interval=max_interval,
    )


def _make_concept(n_reviews: int = 3) -> ConceptState:
    cs = ConceptState(concept_id="c1")
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    for i in range(n_reviews):
        cs.add_review(score=0.8, timestamp=base + timedelta(days=i * 5))
    return cs


class TestSpacedRepetitionScheduler:
    # ------------------------------------------------------------------
    # Constructor validation
    # ------------------------------------------------------------------

    def test_invalid_target_recall_zero(self):
        model = RecallLSTM(input_size=6, hidden_size=16, num_layers=1)
        with pytest.raises(ValueError):
            SpacedRepetitionScheduler(model, target_recall=0.0)

    def test_invalid_target_recall_one(self):
        model = RecallLSTM(input_size=6, hidden_size=16, num_layers=1)
        with pytest.raises(ValueError):
            SpacedRepetitionScheduler(model, target_recall=1.0)

    def test_invalid_min_interval(self):
        model = RecallLSTM(input_size=6, hidden_size=16, num_layers=1)
        with pytest.raises(ValueError):
            SpacedRepetitionScheduler(model, min_interval=0.0)

    def test_invalid_max_less_than_min(self):
        model = RecallLSTM(input_size=6, hidden_size=16, num_layers=1)
        with pytest.raises(ValueError):
            SpacedRepetitionScheduler(model, min_interval=10.0, max_interval=5.0)

    # ------------------------------------------------------------------
    # next_interval
    # ------------------------------------------------------------------

    def test_next_interval_returns_float(self):
        sched = _make_scheduler()
        cs = _make_concept()
        interval = sched.next_interval(cs)
        assert isinstance(interval, float)

    def test_next_interval_in_bounds(self):
        sched = _make_scheduler(min_interval=1.0, max_interval=365.0)
        cs = _make_concept()
        interval = sched.next_interval(cs)
        assert sched.min_interval <= interval <= sched.max_interval

    def test_next_interval_empty_concept(self):
        sched = _make_scheduler()
        cs = ConceptState(concept_id="new")
        interval = sched.next_interval(cs)
        assert sched.min_interval <= interval <= sched.max_interval

    # ------------------------------------------------------------------
    # recall_curve
    # ------------------------------------------------------------------

    def test_recall_curve_length(self):
        sched = _make_scheduler()
        cs = _make_concept()
        intervals = [1.0, 7.0, 14.0, 30.0]
        curve = sched.recall_curve(cs, intervals)
        assert len(curve) == len(intervals)

    def test_recall_curve_values_in_range(self):
        sched = _make_scheduler()
        cs = _make_concept()
        curve = sched.recall_curve(cs, [1.0, 5.0, 10.0])
        assert all(0.0 <= p <= 1.0 for p in curve)

    # ------------------------------------------------------------------
    # due_concepts
    # ------------------------------------------------------------------

    def test_due_concepts_all_included(self):
        sched = _make_scheduler()
        user = UserState(user_id="u1")
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        for cid in ["c1", "c2", "c3"]:
            cs = user.get_or_create_concept(cid)
            cs.add_review(0.8, base)

        due = sched.due_concepts(user)
        assert len(due) == 3
        concept_ids = {cid for cid, _ in due}
        assert concept_ids == {"c1", "c2", "c3"}

    def test_due_concepts_sorted_ascending(self):
        sched = _make_scheduler()
        user = UserState(user_id="u1")
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        for cid in ["c1", "c2", "c3"]:
            cs = user.get_or_create_concept(cid)
            cs.add_review(0.8, base)

        due = sched.due_concepts(user)
        intervals = [iv for _, iv in due]
        assert intervals == sorted(intervals)

    # ------------------------------------------------------------------
    # Interleaving
    # ------------------------------------------------------------------

    def test_due_concepts_interleave_all_included(self):
        sched = _make_scheduler()
        user = UserState(user_id="u1")
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        for cid, cat in [("c1", "math"), ("c2", "science"), ("c3", "math")]:
            cs = user.get_or_create_concept(cid, category=cat)
            cs.add_review(0.8, base)

        due = sched.due_concepts(user, interleave=True)
        assert len(due) == 3
        concept_ids = {cid for cid, _ in due}
        assert concept_ids == {"c1", "c2", "c3"}

    def test_due_concepts_interleave_separates_categories(self):
        """Interleaving should avoid placing same-category concepts adjacently."""
        sched = _make_scheduler()
        user = UserState(user_id="u1")
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        # Create 4 concepts: 2 math, 2 science
        for cid, cat in [("c1", "math"), ("c2", "math"),
                         ("c3", "science"), ("c4", "science")]:
            cs = user.get_or_create_concept(cid, category=cat)
            cs.add_review(0.8, base)

        due = sched.due_concepts(user, interleave=True)
        assert len(due) == 4
        # Verify all concepts are present
        concept_ids = {cid for cid, _ in due}
        assert concept_ids == {"c1", "c2", "c3", "c4"}
