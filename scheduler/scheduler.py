"""Spaced-repetition scheduler backed by a personalised :class:`RecallLSTM`.

The scheduler answers one question: *"Given a student's review history for a
concept, what is the optimal next review interval?"*

Strategy
--------
We search for the largest interval ``I`` (days) such that the model predicts
recall probability ≥ ``target_recall`` (default 0.9 — the "retrievability
target").  This mirrors the intuition behind SM-2 and FSRS, but the forgetting
curve is *learned* from data rather than hard-coded.

A binary-search over ``[min_interval, max_interval]`` is used for efficiency.

Interleaving
------------
Research by Rohrer et al. (2015) and Kornell & Bjork (2008) shows that
interleaving different *categories* of material during a study session
leads to better discriminative learning than blocking by topic.  The
``due_concepts`` method supports an ``interleave`` flag that reorders
the review schedule to maximise category diversity.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

import torch

from .data import ConceptState
from .model import RecallLSTM


class SpacedRepetitionScheduler:
    """Schedule next reviews using a trained :class:`RecallLSTM`.

    Parameters
    ----------
    model:
        A trained (or meta-trained) :class:`RecallLSTM`.
    target_recall:
        Desired recall probability at the next review (default 0.9).
    min_interval:
        Minimum allowed interval in days (default 1.0).
    max_interval:
        Maximum allowed interval in days (default 365.0).
    binary_search_steps:
        Number of bisection iterations used to locate the target interval
        (default 20, giving precision ≈ (max – min) / 2^20 ≈ 0.0003 days).
    device:
        Torch device for inference (defaults to the model's current device).
    """

    def __init__(
        self,
        model: RecallLSTM,
        target_recall: float = 0.9,
        min_interval: float = 1.0,
        max_interval: float = 365.0,
        binary_search_steps: int = 20,
        device: Optional[torch.device] = None,
    ) -> None:
        if not 0.0 < target_recall < 1.0:
            raise ValueError("target_recall must be in (0, 1)")
        if min_interval <= 0:
            raise ValueError("min_interval must be positive")
        if max_interval <= min_interval:
            raise ValueError("max_interval must be greater than min_interval")

        self.model = model
        self.target_recall = target_recall
        self.min_interval = min_interval
        self.max_interval = max_interval
        self.binary_search_steps = binary_search_steps
        self.device = device or next(model.parameters()).device

    # ------------------------------------------------------------------
    # Core scheduling
    # ------------------------------------------------------------------

    def next_interval(self, concept_state: ConceptState) -> float:
        """Compute the recommended next review interval (days).

        The interval is the largest value in ``[min_interval, max_interval]``
        for which predicted recall ≥ ``target_recall``.

        Parameters
        ----------
        concept_state:
            The concept's full review history for this user.

        Returns
        -------
        float
            Recommended interval in days.
        """
        features = concept_state.as_feature_sequence()

        # Edge case: recall at min_interval is already below target
        if self.model.predict(features, self.min_interval, self.device) < self.target_recall:
            return self.min_interval

        # Edge case: recall is still above target even at max_interval
        if self.model.predict(features, self.max_interval, self.device) >= self.target_recall:
            return self.max_interval

        # Binary search for the largest interval with recall >= target
        lo, hi = self.min_interval, self.max_interval
        for _ in range(self.binary_search_steps):
            mid = (lo + hi) / 2.0
            prob = self.model.predict(features, mid, self.device)
            if prob >= self.target_recall:
                lo = mid  # can go longer
            else:
                hi = mid  # too long, shorten

        return lo

    # ------------------------------------------------------------------
    # Recall probability curve
    # ------------------------------------------------------------------

    def recall_curve(
        self,
        concept_state: ConceptState,
        intervals: list[float],
    ) -> list[float]:
        """Return the predicted recall probability at each queried interval.

        Parameters
        ----------
        concept_state:
            Review history.
        intervals:
            List of future intervals (days) to evaluate.

        Returns
        -------
        list[float]
            Recall probabilities corresponding to each interval.
        """
        features = concept_state.as_feature_sequence()
        return [
            self.model.predict(features, interval, self.device)
            for interval in intervals
        ]

    # ------------------------------------------------------------------
    # Convenience: schedule all concepts for a user
    # ------------------------------------------------------------------

    def due_concepts(
        self,
        user_state,  # UserState – avoid circular import by duck-typing
        reference_now_days: float = 0.0,
        interleave: bool = False,
    ) -> list[tuple[str, float]]:
        """Return ``(concept_id, recommended_interval)`` for every concept.

        Parameters
        ----------
        user_state:
            A :class:`~scheduler.data.UserState` instance.
        reference_now_days:
            *Unused* – present for future extensions that track absolute time.
        interleave:
            When ``True``, reorder the schedule to maximise category
            diversity (interleaving effect – Rohrer et al. 2015).  Concepts
            with no ``category`` set are treated as their own unique group.
            When ``False`` (default), concepts are sorted by interval
            ascending (most urgent first), preserving the original behaviour.

        Returns
        -------
        list[tuple[str, float]]
            Sorted by recommended interval (ascending) or interleaved by
            category.
        """
        results = []
        for concept_id, cs in user_state.concept_states.items():
            interval = self.next_interval(cs)
            results.append((concept_id, interval, cs.category))

        # Default: sort by urgency
        results.sort(key=lambda t: t[1])

        if not interleave:
            return [(cid, iv) for cid, iv, _ in results]

        return self._interleave_by_category(results)

    # ------------------------------------------------------------------
    # Interleaving helper
    # ------------------------------------------------------------------

    @staticmethod
    def _interleave_by_category(
        items: list[tuple[str, float, str]],
    ) -> list[tuple[str, float]]:
        """Reorder *items* to maximise category diversity.

        Items within each category retain their urgency order but categories
        are round-robin interleaved so that consecutive items are unlikely
        to share a category.
        """
        if not items:
            return []

        # Group by category (empty category → use concept_id as unique group)
        buckets: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for cid, iv, cat in items:
            key = cat if cat else cid
            buckets[key].append((cid, iv))

        # Sort category keys by the urgency of their first item so that the
        # most urgent categories lead.
        sorted_keys = sorted(buckets, key=lambda k: buckets[k][0][1])

        # Round-robin across category queues
        queues: list[list[tuple[str, float]]] = [
            buckets[k] for k in sorted_keys
        ]
        interleaved: list[tuple[str, float]] = []
        indices = [0] * len(queues)

        while True:
            progress = False
            for qi, q in enumerate(queues):
                if indices[qi] < len(q):
                    interleaved.append(q[indices[qi]])
                    indices[qi] += 1
                    progress = True
            if not progress:
                break

        return interleaved
