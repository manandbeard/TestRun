"""Tests for :mod:`scheduler.model` (:class:`RecallLSTM`)."""

from __future__ import annotations

import torch
import pytest

from scheduler.model import RecallLSTM


class TestRecallLSTM:
    def _model(self) -> RecallLSTM:
        return RecallLSTM(input_size=6, hidden_size=16, num_layers=1, dropout=0.0)

    # ------------------------------------------------------------------
    # Architecture
    # ------------------------------------------------------------------

    def test_output_shape_batch(self):
        model = self._model()
        x = torch.rand(4, 6, 6)  # (batch=4, seq_len=6, input_size=6)
        out = model(x)
        assert out.shape == (4,)

    def test_output_range(self):
        model = self._model()
        x = torch.rand(8, 5, 6)
        out = model(x)
        assert (out >= 0.0).all() and (out <= 1.0).all()

    def test_single_step_sequence(self):
        model = self._model()
        x = torch.rand(1, 1, 6)
        out = model(x)
        assert out.shape == (1,)
        assert 0.0 <= out.item() <= 1.0

    # ------------------------------------------------------------------
    # predict() convenience method
    # ------------------------------------------------------------------

    def test_predict_with_history(self):
        model = self._model()
        features = [[0.0, 0.9, 0.3, 1.0, 0.33], [5.0, 0.7, 0.28, 2.5, 0.67], [7.0, 0.8, 0.25, 4.0, 1.0]]
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
        model.predict([[0.0, 1.0, 0.3, 1.0, 1.0]], 3.0)
        params_after = [p.data for p in model.parameters()]
        for b, a in zip(params_before, params_after):
            assert torch.allclose(b, a)

    # ------------------------------------------------------------------
    # num_layers > 1
    # ------------------------------------------------------------------

    def test_multilayer_lstm(self):
        model = RecallLSTM(input_size=6, hidden_size=16, num_layers=2, dropout=0.1)
        x = torch.rand(2, 4, 6)
        out = model(x)
        assert out.shape == (2,)
        assert (out >= 0.0).all() and (out <= 1.0).all()

    # ------------------------------------------------------------------
    # Attention mechanism
    # ------------------------------------------------------------------

    def test_has_attention_layer(self):
        model = self._model()
        assert hasattr(model, "attention")

    def test_attention_differentiable(self):
        """Ensure attention weights participate in gradient computation."""
        model = self._model()
        model.train()
        x = torch.rand(2, 4, 6, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()
        assert model.attention.weight.grad is not None
