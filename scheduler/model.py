"""Attention-augmented LSTM model for predicting recall probability.

The :class:`RecallLSTM` takes a variable-length sequence of past reviews
and predicts the probability that the student will recall the concept after
a given future interval ``query_interval_days``.

Architecture
------------
Input  : [elapsed_days, score, difficulty, stability, norm_review_count,
          query_interval_days]  – one time step per review.  The query
         interval is appended at the *last* time step only; earlier steps
         use 0.
LSTM   : ``hidden_size`` units, ``num_layers`` stacked layers.
Attention : Learned scalar attention weights over LSTM hidden states so that
            the model can focus on the most informative past reviews
            (e.g. recent reviews and difficult retrievals carry more
            signal – consistent with desirable-difficulties theory,
            Bjork & Bjork 2011).
Output : sigmoid-activated scalar – P(recall | history, interval).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class RecallLSTM(nn.Module):
    """Attention-augmented LSTM that predicts recall probability.

    Parameters
    ----------
    input_size:
        Number of input features per time step (default 6:
        ``elapsed_days``, ``score``, ``difficulty``, ``stability``,
        ``norm_review_count``, ``query_interval``).
    hidden_size:
        Number of hidden units in each LSTM layer.
    num_layers:
        Number of stacked LSTM layers.
    dropout:
        Dropout probability applied between LSTM layers (only active when
        ``num_layers > 1``).
    """

    def __init__(
        self,
        input_size: int = 6,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Attention layer: learn which past reviews are most informative
        self.attention = nn.Linear(hidden_size, 1)

        self.output_layer = nn.Linear(hidden_size, 1)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, x: Tensor) -> Tensor:
        """Run the LSTM with attention and return recall probability.

        Parameters
        ----------
        x:
            Float tensor of shape ``(batch, seq_len, input_size)``.

        Returns
        -------
        Tensor
            Recall probability, shape ``(batch,)``.
        """
        outputs, _ = self.lstm(x)  # (batch, seq_len, hidden_size)

        # Attention: learn importance weights across time steps
        attn_scores = self.attention(outputs).squeeze(-1)  # (batch, seq_len)
        attn_weights = torch.softmax(attn_scores, dim=-1)  # (batch, seq_len)

        # Weighted sum of hidden states → context vector
        context = torch.bmm(
            attn_weights.unsqueeze(1), outputs
        ).squeeze(1)  # (batch, hidden_size)

        logit = self.output_layer(context).squeeze(-1)  # (batch,)
        return torch.sigmoid(logit)

    # ------------------------------------------------------------------
    # Convenience: predict a single concept state
    # ------------------------------------------------------------------

    def predict(
        self,
        feature_sequence: list[list[float]],
        query_interval_days: float,
        device: torch.device | None = None,
    ) -> float:
        """Return the scalar recall probability for one concept.

        Parameters
        ----------
        feature_sequence:
            List of feature vectors from
            :meth:`ConceptState.as_feature_sequence`.  Each inner list has
            five elements: ``[elapsed_days, score, difficulty, stability,
            norm_review_count]``.
        query_interval_days:
            The prospective interval (days) we want to evaluate recall for.
        device:
            Torch device for inference.

        Returns
        -------
        float
            Recall probability in ``[0, 1]``.
        """
        if device is None:
            device = next(self.parameters()).device

        if not feature_sequence:
            # No history – use a single synthetic step
            feature_sequence = [[0.0, 0.0, 0.3, 1.0, 0.0]]

        # Build tensor: append query_interval as the last feature at the
        # final time step (0 elsewhere).
        seq = []
        for i, feats in enumerate(feature_sequence):
            interval = query_interval_days if i == len(feature_sequence) - 1 else 0.0
            seq.append(list(feats) + [interval])

        x = torch.tensor(seq, dtype=torch.float32, device=device).unsqueeze(0)
        self.eval()
        with torch.no_grad():
            prob = self(x).item()
        return float(prob)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save model weights and architecture hyperparameters to *path*."""
        torch.save(
            {
                "input_size": self.input_size,
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "state_dict": self.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path: str, device: torch.device | None = None) -> "RecallLSTM":
        """Load a model from *path* and return it."""
        checkpoint = torch.load(path, map_location=device or "cpu", weights_only=True)
        model = cls(
            input_size=checkpoint["input_size"],
            hidden_size=checkpoint["hidden_size"],
            num_layers=checkpoint["num_layers"],
        )
        model.load_state_dict(checkpoint["state_dict"])
        if device is not None:
            model.to(device)
        return model
