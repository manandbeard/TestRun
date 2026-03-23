"""Tests for :mod:`scheduler.reptile` (:class:`ReptileTrainer`)."""

from __future__ import annotations

import copy
import random

import torch
import pytest

from scheduler.model import RecallLSTM
from scheduler.reptile import ReptileTrainer, build_task_sample, TaskSample

SEED = 0


def _simple_task(n_samples: int = 6) -> TaskSample:
    """Build a tiny synthetic task sample."""
    rng = random.Random(SEED)
    task: TaskSample = []
    features: list[list[float]] = []
    for i in range(n_samples):
        elapsed = float(rng.uniform(1, 10))
        score = float(rng.uniform(0.4, 1.0))
        interval = float(rng.uniform(1, 14))
        label = 1.0 if rng.random() > 0.4 else 0.0
        sample = build_task_sample(features, interval, label)
        task.append(sample)
        features.append([elapsed, score])
    return task


class TestBuildTaskSample:
    def test_shape(self):
        features = [[0.0, 0.9], [5.0, 0.8]]
        x, y = build_task_sample(features, 7.0, 1.0)
        assert x.shape == (2, 3)  # (seq_len=2, input_size=3)
        assert y.item() == 1.0

    def test_empty_features(self):
        x, y = build_task_sample([], 3.0, 0.0)
        assert x.shape == (1, 3)  # synthetic step for empty history
        assert y.item() == 0.0

    def test_query_interval_appended_last(self):
        features = [[2.0, 0.7], [5.0, 0.8]]
        x, _ = build_task_sample(features, 10.0, 1.0)
        # The last time step should contain the query interval in column 2
        assert x[-1, 2].item() == pytest.approx(10.0)
        # Earlier steps should have 0 in column 2
        assert x[0, 2].item() == pytest.approx(0.0)


class TestReptileTrainer:
    def _trainer(self) -> ReptileTrainer:
        torch.manual_seed(SEED)
        model = RecallLSTM(input_size=3, hidden_size=16, num_layers=1, dropout=0.0)
        return ReptileTrainer(model, inner_lr=0.05, inner_steps=3, meta_lr=0.1)

    def test_meta_train_runs(self):
        trainer = self._trainer()
        tasks = [_simple_task() for _ in range(4)]
        losses = trainer.meta_train(tasks, epochs=5, tasks_per_epoch=2)
        assert len(losses) == 5
        assert all(isinstance(l, float) for l in losses)

    def test_meta_train_updates_params(self):
        trainer = self._trainer()
        params_before = [p.data.clone() for p in trainer.model.parameters()]
        tasks = [_simple_task() for _ in range(4)]
        trainer.meta_train(tasks, epochs=3, tasks_per_epoch=2)
        params_after = [p.data for p in trainer.model.parameters()]
        # At least one parameter should have changed
        changed = any(
            not torch.allclose(b, a)
            for b, a in zip(params_before, params_after)
        )
        assert changed

    def test_meta_train_empty_tasks_raises(self):
        trainer = self._trainer()
        with pytest.raises(ValueError, match="non-empty"):
            trainer.meta_train([], epochs=1)

    def test_personalise_returns_different_model(self):
        trainer = self._trainer()
        tasks = [_simple_task() for _ in range(4)]
        trainer.meta_train(tasks, epochs=3, tasks_per_epoch=2)

        user_task = _simple_task(4)
        personal_model = trainer.personalise(user_task, steps=3)

        # Personalised model should differ from meta-model
        meta_params = list(trainer.model.parameters())
        pers_params = list(personal_model.parameters())
        different = any(
            not torch.allclose(m.data, p.data)
            for m, p in zip(meta_params, pers_params)
        )
        assert different

    def test_personalise_does_not_modify_meta_model(self):
        trainer = self._trainer()
        tasks = [_simple_task() for _ in range(4)]
        trainer.meta_train(tasks, epochs=2, tasks_per_epoch=2)

        params_before = [p.data.clone() for p in trainer.model.parameters()]
        trainer.personalise(_simple_task(4), steps=3)
        params_after = [p.data for p in trainer.model.parameters()]

        for b, a in zip(params_before, params_after):
            assert torch.allclose(b, a)
