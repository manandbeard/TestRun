"""Data models for the spaced-repetition scheduler.

Three core abstractions:

* :class:`ReviewEvent`  – a single study event for one (user, concept) pair.
* :class:`ConceptState` – accumulated review history for one concept owned by
  a specific user.
* :class:`UserState`    – collection of all concept states for one user.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional


@dataclass
class ReviewEvent:
    """A single review of a concept by a user.

    Parameters
    ----------
    timestamp:
        UTC datetime of the review.
    score:
        Normalised performance score in *[0, 1]* (e.g. 0 = complete blank,
        1 = perfect recall).
    elapsed_days:
        Fractional days since the *previous* review of this concept (0.0 for
        the very first review).
    """

    timestamp: datetime
    score: float  # [0, 1]
    elapsed_days: float  # >= 0

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"score must be in [0, 1], got {self.score}")
        if self.elapsed_days < 0:
            raise ValueError(
                f"elapsed_days must be >= 0, got {self.elapsed_days}"
            )


@dataclass
class ConceptState:
    """All review history for a single (user, concept) pair.

    Parameters
    ----------
    concept_id:
        Opaque identifier for the concept (e.g. UUID string or integer).
    reviews:
        Chronologically-ordered list of :class:`ReviewEvent` objects.
    """

    concept_id: str
    reviews: List[ReviewEvent] = field(default_factory=list)

    # ---------------------------------------------------------------------------
    # Convenience helpers
    # ---------------------------------------------------------------------------

    def add_review(self, score: float, timestamp: Optional[datetime] = None) -> ReviewEvent:
        """Append a new review and return the created :class:`ReviewEvent`.

        The ``elapsed_days`` is computed automatically from the previous review.
        """
        if timestamp is None:
            timestamp = datetime.now(tz=timezone.utc)

        if self.reviews:
            delta = timestamp - self.reviews[-1].timestamp
            elapsed_days = max(delta.total_seconds() / 86_400.0, 0.0)
        else:
            elapsed_days = 0.0

        event = ReviewEvent(
            timestamp=timestamp,
            score=score,
            elapsed_days=elapsed_days,
        )
        self.reviews.append(event)
        return event

    @property
    def last_review(self) -> Optional[ReviewEvent]:
        """Most recent review or ``None`` if no reviews exist yet."""
        return self.reviews[-1] if self.reviews else None

    @property
    def review_count(self) -> int:
        return len(self.reviews)

    def as_feature_sequence(self) -> List[List[float]]:
        """Return reviews as a list of ``[elapsed_days, score]`` feature vectors."""
        return [[r.elapsed_days, r.score] for r in self.reviews]

    # Ebbinghaus-based stability estimate (used as a baseline / warm-start)
    def stability_estimate(self) -> float:
        """Rough stability estimate using an exponential-forgetting heuristic.

        Returns the average recall-weighted interval so far, which acts as a
        prior on how long the student can wait between reviews.
        """
        if not self.reviews:
            return 1.0  # default 1-day interval for new concepts

        weighted_sum = 0.0
        weight_total = 0.0
        for review in self.reviews:
            w = review.score
            weighted_sum += w * (review.elapsed_days + 1.0)
            weight_total += w

        if weight_total == 0:
            return 1.0

        return weighted_sum / weight_total


@dataclass
class UserState:
    """All concept states for one user.

    Parameters
    ----------
    user_id:
        Opaque user identifier.
    concept_states:
        Mapping from ``concept_id`` to :class:`ConceptState`.
    """

    user_id: str
    concept_states: dict[str, ConceptState] = field(default_factory=dict)

    def get_or_create_concept(self, concept_id: str) -> ConceptState:
        """Return the existing :class:`ConceptState` or create a new one."""
        if concept_id not in self.concept_states:
            self.concept_states[concept_id] = ConceptState(concept_id=concept_id)
        return self.concept_states[concept_id]

    @property
    def concept_ids(self) -> List[str]:
        return list(self.concept_states.keys())

    def total_reviews(self) -> int:
        return sum(cs.review_count for cs in self.concept_states.values())
