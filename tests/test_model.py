"""Tests for :mod:`scheduler.model` (:class:`RecallLSTM`)."""

from __future__ import annotations

import torch
import pytest

from scheduler.model import RecallLSTM


class TestRecallLSTM:
    def _model(self) -> RecallLSTM:
        return RecallLSTM(input_size=3, hidden_size=16, num_layers=1, dropout=0.0)

    # ------------------------------------------------------------------
    # Architecture
    # ------------------------------------------------------------------

    def test_output_shape_batch(self):
        model = self._model()
        x = torch.rand(4, 6, 3)  # (batch=4, seq_len=6, input_size=3)
        out = model(x)
        assert out.shape == (4,)

    def test_output_range(self):
        model = self._model()
        x = torch.rand(8, 5, 3)
        out = model(x)
        assert (out >= 0.0).all() and (out <= 1.0).all()

    def test_single_step_sequence(self):
        model = self._model()
        x = torch.rand(1, 1, 3)
        out = model(x)
        assert out.shape == (1,)
        assert 0.0 <= out.item() <= 1.0

    # ------------------------------------------------------------------
    # predict() convenience method
    # ------------------------------------------------------------------

    def test_predict_with_history(self):
        model = self._model()
        features = [[0.0, 0.9], [5.0, 0.7], [7.0, 0.8]]
        prob = model.predict(features, query_interval_days=10.0)
        assert isinstance(prob, float)
        assert 0.0 <= prob <= 1.0

    def test_predict_empty_history(self):
        model = self._model()
        prob = model.predict([], query_interval_days=5.0)
        assert 0.0 <= prob <= 1.0

    def test_predict_no_mutation(self):
        """predict() must not alter model parameters."""
        model = self._model()
        params_before = [p.data.clone() for p in model.parameters()]
        model.predict([[0.0, 1.0]], 3.0)
        params_after = [p.data for p in model.parameters()]
        for b, a in zip(params_before, params_after):
            assert torch.allclose(b, a)

    # ------------------------------------------------------------------
    # num_layers > 1
    # ------------------------------------------------------------------

    def test_multilayer_lstm(self):
        model = RecallLSTM(input_size=3, hidden_size=16, num_layers=2, dropout=0.1)
        x = torch.rand(2, 4, 3)
        out = model(x)
        assert out.shape == (2,)
        assert (out >= 0.0).all() and (out <= 1.0).all()
