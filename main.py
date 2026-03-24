#!/usr/bin/env python3
"""Demo script for the LSTM + Reptile spaced-repetition scheduler.

Generates synthetic review data for a handful of simulated users, runs the
Reptile meta-training loop, personalises the model to a new user, and prints
the recommended next-review intervals.

Run:
    python main.py
"""

from __future__ import annotations

import math
import random

import torch

from scheduler import (
    RecallLSTM,
    ReptileTrainer,
    SpacedRepetitionScheduler,
    UserState,
)
from scheduler.reptile import build_task_sample, TaskSample

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

CATEGORIES = ["math", "science", "history", "language"]


def _forgetting_curve(elapsed_days: float, stability: float) -> float:
    """Power-law probability of recall after *elapsed_days* days.

    Uses the research-backed power-law curve R(t) = (1 + t/S)^(-1)
    (Wixted & Ebbesen 1991) instead of Ebbinghaus exponential.
    """
    return (1.0 + elapsed_days / max(stability, 1e-6)) ** (-1.0)


def _simulate_user(
    user_id: str,
    n_concepts: int = 5,
    reviews_per_concept: int = 8,
    stability_range: tuple[float, float] = (1.0, 14.0),
) -> tuple[UserState, TaskSample]:
    """Return a simulated :class:`UserState` and a corresponding task sample."""
    user = UserState(user_id=user_id)
    task: TaskSample = []

    for c in range(n_concepts):
        concept_id = f"{user_id}_c{c}"
        category = CATEGORIES[c % len(CATEGORIES)]
        stability = random.uniform(*stability_range)
        cs = user.get_or_create_concept(concept_id, category=category)

        elapsed = 0.0
        for rev in range(reviews_per_concept):
            score = _forgetting_curve(elapsed, stability) * random.uniform(0.7, 1.0)
            score = min(max(score, 0.0), 1.0)
            features_so_far = cs.as_feature_sequence()

            # Build a training example: predict recall after the next interval
            next_elapsed = random.uniform(1.0, stability * 2)
            label = _forgetting_curve(next_elapsed, stability)
            label = float(label > 0.85)  # binarise

            sample = build_task_sample(features_so_far, next_elapsed, label)
            task.append(sample)

            # Advance the concept state
            from datetime import datetime, timedelta, timezone
            ts = datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(
                days=sum(r.elapsed_days for r in cs.reviews) + elapsed
            )
            cs.add_review(score=score, timestamp=ts)
            elapsed = next_elapsed

    return user, task


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=== Attention-LSTM + Reptile Spaced-Repetition Scheduler Demo ===\n")
    print("Research-backed improvements:")
    print("  • Power-law forgetting curve (Wixted & Ebbesen 1991)")
    print("  • FSRS-inspired difficulty/stability tracking (Ye 2023)")
    print("  • Attention over review history (desirable difficulties)")
    print("  • Cosine-annealed meta-LR + gradient clipping")
    print("  • Interleaved scheduling (Rohrer et al. 2015)\n")

    # 1. Build meta-training dataset (10 simulated users)
    print("Generating synthetic user data …")
    all_tasks: list[TaskSample] = []
    for i in range(10):
        _, task = _simulate_user(f"meta_user_{i}")
        all_tasks.append(task)
    print(f"  {len(all_tasks)} meta-training tasks generated.\n")

    # 2. Initialise model + trainer (input_size=6: 5 features + query_interval)
    model = RecallLSTM(input_size=6, hidden_size=64, num_layers=2, dropout=0.1)
    trainer = ReptileTrainer(
        model,
        inner_lr=0.01,
        inner_steps=5,
        meta_lr=0.1,
        meta_lr_min=0.001,
        max_grad_norm=5.0,
    )

    # 3. Meta-train
    print("Running Reptile meta-training (50 epochs, cosine LR) …")
    losses = trainer.meta_train(
        all_tasks,
        epochs=50,
        tasks_per_epoch=4,
        on_epoch_end=lambda e, l: print(f"  epoch {e+1:3d}  loss={l:.4f}")
        if (e + 1) % 10 == 0
        else None,
    )
    print(f"\n  Final meta-train loss: {losses[-1]:.4f}\n")

    # 4. Simulate a *new* user (unseen during meta-training)
    print("Simulating a new student …")
    new_user, new_task = _simulate_user(
        "new_student", n_concepts=3, reviews_per_concept=4
    )
    print(f"  {new_user.total_reviews()} reviews recorded across "
          f"{len(new_user.concept_ids)} concepts.\n")

    # 5. Personalise the meta-model to the new student
    print("Personalising model to new student (10 fine-tuning steps) …")
    personal_model = trainer.personalise(new_task, steps=10)

    # 6. Schedule next reviews (with interleaving)
    print("\nRecommended next-review intervals (interleaved by category):")
    scheduler = SpacedRepetitionScheduler(personal_model, target_recall=0.9)
    due = scheduler.due_concepts(new_user, interleave=True)
    for concept_id, interval in due:
        cs = new_user.concept_states[concept_id]
        cat_label = f"[{cs.category}]" if cs.category else ""
        print(f"  {concept_id:30s} {cat_label:12s}  → {interval:.1f} days"
              f"  (D={cs.difficulty:.2f}, S={cs.stability:.1f})")

    print("\nDone.")


if __name__ == "__main__":
    main()
