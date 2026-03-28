"""Data models for the spaced-repetition scheduler.

Three core abstractions:

* :class:`ReviewEvent`  – a single study event for one (user, concept) pair.
* :class:`ConceptState` – accumulated review history for one concept owned by
  a specific user, with FSRS-inspired difficulty/stability tracking.
* :class:`UserState`    – collection of all concept states for one user.

Research references
-------------------
* Power-law forgetting: Wixted & Ebbesen (1991), Wixted (2004).
* FSRS state model:  Ye (2023) – Free Spaced Repetition Scheduler.
* Desirable difficulties / retrieval effort: Bjork & Bjork (2011).
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

    def to_dict(self) -> dict:
        """Serialise to a JSON-compatible dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "score": self.score,
            "elapsed_days": self.elapsed_days,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ReviewEvent":
        """Deserialise from a dictionary."""
        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            score=data["score"],
            elapsed_days=data["elapsed_days"],
        )


# ---------------------------------------------------------------------------
# FSRS-inspired parameter defaults
# ---------------------------------------------------------------------------
_INITIAL_DIFFICULTY = 0.3
_INITIAL_STABILITY = 1.0
_DIFFICULTY_WEIGHT = 0.1
_DIFFICULTY_MEAN_REVERSION = 0.05
_STABILITY_GROWTH_BASE = 1.0
_STABILITY_DIFFICULTY_DECAY = 0.8
_STABILITY_SELF_DECAY = 0.2
_STABILITY_FAIL_PENALTY = 0.5


@dataclass
class ConceptState:
    """All review history for a single (user, concept) pair.

    In addition to the raw review list the state now tracks two FSRS-inspired
    latent variables that are updated after every review:

    * **difficulty** (D) – how hard the concept is for this student (0 = easy,
      1 = very hard).  Updated with mean-reversion towards the population
      average.
    * **stability** (S) – current memory stability in *days*.  Governs how
      quickly the student forgets (larger → slower forgetting).

    Parameters
    ----------
    concept_id:
        Opaque identifier for the concept (e.g. UUID string or integer).
    reviews:
        Chronologically-ordered list of :class:`ReviewEvent` objects.
    difficulty:
        Current difficulty estimate (0–1).
    stability:
        Current memory stability in days.
    category:
        Optional category / topic tag used for interleaved scheduling.
    """

    concept_id: str
    reviews: List[ReviewEvent] = field(default_factory=list)
    difficulty: float = _INITIAL_DIFFICULTY
    stability: float = _INITIAL_STABILITY
    category: str = ""

    # ---------------------------------------------------------------------------
    # Convenience helpers
    # ---------------------------------------------------------------------------

    def add_review(self, score: float, timestamp: Optional[datetime] = None) -> ReviewEvent:
        """Append a new review, update D/S, and return the :class:`ReviewEvent`.

        The ``elapsed_days`` is computed automatically from the previous review.
        Difficulty and stability are updated using FSRS-inspired update rules
        after the event is recorded.
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

        # ---- FSRS-inspired D/S update (Ye 2023) ----
        self._update_difficulty(score)
        self._update_stability(score, elapsed_days)

        return event

    # ---- Internal D/S updates -----------------------------------------------

    def _update_difficulty(self, score: float) -> None:
        """Shift D towards population mean; decrease when score is high."""
        delta_d = -_DIFFICULTY_WEIGHT * (score - 0.6)
        mean_revert = _DIFFICULTY_MEAN_REVERSION * (_INITIAL_DIFFICULTY - self.difficulty)
        self.difficulty = max(0.0, min(1.0, self.difficulty + delta_d + mean_revert))

    def _update_stability(self, score: float, elapsed_days: float) -> None:
        """Grow S on successful recall; shrink on failure."""
        if score >= 0.6:
            # Successful recall – stability grows
            retrievability = self.retrievability(elapsed_days)
            # Desirable-difficulty bonus (Bjork & Bjork 2011): when retrieval
            # was harder (lower retrievability), the bonus is larger, which
            # amplifies stability growth — rewarding effortful recall.
            retrieval_bonus = 1.0 - retrievability
            growth = (
                _STABILITY_GROWTH_BASE
                * math.exp(-_STABILITY_DIFFICULTY_DECAY * self.difficulty)
                * max(self.stability, 0.01) ** (-_STABILITY_SELF_DECAY)
                * (1.0 + retrieval_bonus)
            )
            self.stability = max(self.stability * (1.0 + growth * score), 0.01)
        else:
            # Failed recall – stability shrinks
            self.stability = max(
                self.stability * _STABILITY_FAIL_PENALTY * (1.0 + score),
                0.01,
            )

    # ---- Forgetting curve ---------------------------------------------------

    def retrievability(self, elapsed_days: float) -> float:
        """Power-law recall probability after *elapsed_days* since last review.

        Uses a power-law forgetting curve ``R(t) = (1 + t/S)^(-1)`` which
        better fits empirical data than the classic exponential Ebbinghaus
        curve (Wixted & Ebbesen 1991, Wixted 2004).
        """
        return (1.0 + elapsed_days / max(self.stability, 1e-6)) ** (-1.0)

    # ---- Feature extraction --------------------------------------------------

    @property
    def last_review(self) -> Optional[ReviewEvent]:
        """Most recent review or ``None`` if no reviews exist yet."""
        return self.reviews[-1] if self.reviews else None

    @property
    def review_count(self) -> int:
        return len(self.reviews)

    def as_feature_sequence(self) -> List[List[float]]:
        """Return reviews as a list of enriched feature vectors.

        Each vector is ``[elapsed_days, score, difficulty, stability,
        normalised_review_count]``.  The extra features capture the
        FSRS-inspired state at each time step, giving the model strictly
        more information than the original ``[elapsed_days, score]``
        representation.
        """
        n = len(self.reviews)
        running_d = _INITIAL_DIFFICULTY
        running_s = _INITIAL_STABILITY
        features: List[List[float]] = []
        for idx, r in enumerate(self.reviews):
            norm_count = (idx + 1) / max(n, 1)
            features.append([
                r.elapsed_days,
                r.score,
                running_d,
                running_s,
                norm_count,
            ])
            # After appending the feature vector with pre-review state, update
            # D/S for the next iteration so subsequent steps see post-review state.
            delta_d = -_DIFFICULTY_WEIGHT * (r.score - 0.6)
            mean_revert = _DIFFICULTY_MEAN_REVERSION * (_INITIAL_DIFFICULTY - running_d)
            running_d = max(0.0, min(1.0, running_d + delta_d + mean_revert))

            if r.score >= 0.6:
                retrievability = (1.0 + r.elapsed_days / max(running_s, 1e-6)) ** (-1.0)
                retrieval_bonus = 1.0 - retrievability
                growth = (
                    _STABILITY_GROWTH_BASE
                    * math.exp(-_STABILITY_DIFFICULTY_DECAY * running_d)
                    * max(running_s, 0.01) ** (-_STABILITY_SELF_DECAY)
                    * (1.0 + retrieval_bonus)
                )
                running_s = max(running_s * (1.0 + growth * r.score), 0.01)
            else:
                running_s = max(
                    running_s * _STABILITY_FAIL_PENALTY * (1.0 + r.score),
                    0.01,
                )
        return features

    # ---- Legacy helpers kept for backwards compatibility ---------------------

    def stability_estimate(self) -> float:
        """Power-law stability estimate.

        Uses the tracked ``stability`` field directly, which is updated on
        every review via FSRS-inspired rules.  Falls back to a 1-day default
        for new concepts.
        """
        if not self.reviews:
            return _INITIAL_STABILITY
        return self.stability

    # ---- Serialisation -------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialise to a JSON-compatible dictionary."""
        return {
            "concept_id": self.concept_id,
            "reviews": [r.to_dict() for r in self.reviews],
            "difficulty": self.difficulty,
            "stability": self.stability,
            "category": self.category,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ConceptState":
        """Deserialise from a dictionary."""
        cs = cls(
            concept_id=data["concept_id"],
            difficulty=data.get("difficulty", _INITIAL_DIFFICULTY),
            stability=data.get("stability", _INITIAL_STABILITY),
            category=data.get("category", ""),
        )
        cs.reviews = [ReviewEvent.from_dict(r) for r in data.get("reviews", [])]
        return cs


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

    def get_or_create_concept(
        self,
        concept_id: str,
        category: str = "",
    ) -> ConceptState:
        """Return the existing :class:`ConceptState` or create a new one."""
        if concept_id not in self.concept_states:
            self.concept_states[concept_id] = ConceptState(
                concept_id=concept_id, category=category,
            )
        return self.concept_states[concept_id]

    @property
    def concept_ids(self) -> List[str]:
        return list(self.concept_states.keys())

    def total_reviews(self) -> int:
        return sum(cs.review_count for cs in self.concept_states.values())

    # ---- Serialisation -------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialise to a JSON-compatible dictionary."""
        return {
            "user_id": self.user_id,
            "concept_states": {
                cid: cs.to_dict()
                for cid, cs in self.concept_states.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "UserState":
        """Deserialise from a dictionary."""
        user = cls(user_id=data["user_id"])
        for cid, cs_data in data.get("concept_states", {}).items():
            user.concept_states[cid] = ConceptState.from_dict(cs_data)
        return user
