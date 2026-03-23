"""LSTM model for predicting recall probability.

The :class:`RecallLSTM` takes a variable-length sequence of past reviews
``(elapsed_days, score)`` and predicts the probability that the student will
recall the concept after a given future interval ``query_interval_days``.

Architecture
------------
Input  : [elapsed_days, score, query_interval_days]  – one time step per review
         (the query interval is appended at the *last* time step only; earlier
         steps use 0).
LSTM   : ``hidden_size`` units, ``num_layers`` stacked layers.
Output : sigmoid-activated scalar – P(recall | history, interval).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class RecallLSTM(nn.Module):
    """LSTM that predicts recall probability for a queried future interval.

    Parameters
    ----------
    input_size:
        Number of input features per time step (default 3:
        ``elapsed_days``, ``score``, ``query_interval``).
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
        input_size: int = 3,
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
        self.output_layer = nn.Linear(hidden_size, 1)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, x: Tensor) -> Tensor:
        """Run the LSTM and return recall probability.

        Parameters
        ----------
        x:
            Float tensor of shape ``(batch, seq_len, input_size)``.

        Returns
        -------
        Tensor
            Recall probability, shape ``(batch,)``.
        """
        _, (h_n, _) = self.lstm(x)
        # Use the final hidden state of the top layer
        last_hidden = h_n[-1]  # (batch, hidden_size)
        logit = self.output_layer(last_hidden).squeeze(-1)  # (batch,)
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
            List of ``[elapsed_days, score]`` pairs (output of
            :meth:`ConceptState.as_feature_sequence`).
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
            feature_sequence = [[0.0, 0.0]]

        # Build tensor: [elapsed_days, score, query_interval (only at last step)]
        seq = []
        for i, (elapsed, score) in enumerate(feature_sequence):
            interval = query_interval_days if i == len(feature_sequence) - 1 else 0.0
            seq.append([elapsed, score, interval])

        x = torch.tensor(seq, dtype=torch.float32, device=device).unsqueeze(0)
        self.eval()
        with torch.no_grad():
            prob = self(x).item()
        return float(prob)
