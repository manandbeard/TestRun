#!/usr/bin/env python3
"""Pre-train a Reptile meta-model and save the checkpoint.

This script generates synthetic review data, runs the Reptile meta-training
loop, and persists the resulting model weights to ``meta_model.pt``.  The
web application loads this checkpoint at startup so that personalisation
is available immediately without a long training step.

Run:
    python train_meta_model.py
"""

from __future__ import annotations

import math
import os
import random

import torch

from scheduler import RecallLSTM, ReptileTrainer, UserState
from scheduler.reptile import build_task_sample, TaskSample

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

CATEGORIES = ["math", "science", "history", "language"]
MODEL_PATH = os.path.join(os.path.dirname(__file__), "meta_model.pt")


def _forgetting_curve(elapsed_days: float, stability: float) -> float:
    return (1.0 + elapsed_days / max(stability, 1e-6)) ** (-1.0)


def _simulate_user(
    user_id: str,
    n_concepts: int = 5,
    reviews_per_concept: int = 8,
    stability_range: tuple[float, float] = (1.0, 14.0),
) -> TaskSample:
    from datetime import datetime, timedelta, timezone

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

            next_elapsed = random.uniform(1.0, stability * 2)
            label = _forgetting_curve(next_elapsed, stability)
            label = float(label > 0.85)

            sample = build_task_sample(features_so_far, next_elapsed, label)
            task.append(sample)

            ts = datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(
                days=sum(r.elapsed_days for r in cs.reviews) + elapsed
            )
            cs.add_review(score=score, timestamp=ts)
            elapsed = next_elapsed

    return task


def main() -> None:
    print("Generating synthetic meta-training data (20 users) …")
    all_tasks: list[TaskSample] = []
    for i in range(20):
        task = _simulate_user(f"meta_user_{i}")
        all_tasks.append(task)

    model = RecallLSTM(input_size=6, hidden_size=64, num_layers=2, dropout=0.1)
    trainer = ReptileTrainer(
        model,
        inner_lr=0.01,
        inner_steps=5,
        meta_lr=0.1,
        meta_lr_min=0.001,
        max_grad_norm=5.0,
    )

    print("Running Reptile meta-training (100 epochs) …")
    losses = trainer.meta_train(
        all_tasks,
        epochs=100,
        tasks_per_epoch=4,
        on_epoch_end=lambda e, l: (
            print(f"  epoch {e + 1:3d}  loss={l:.4f}")
            if (e + 1) % 20 == 0
            else None
        ),
    )
    print(f"  Final loss: {losses[-1]:.4f}")

    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
