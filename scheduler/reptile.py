"""Reptile meta-learning trainer for :class:`RecallLSTM`.

Reptile (Nichol et al., 2018) is a first-order meta-learning algorithm that
learns a weight initialisation θ from which fast adaptation to individual
users (tasks) is possible with only a few gradient steps.

Meta-update rule
----------------
For each meta-iteration:
  1. Sample a mini-batch of tasks (users).
  2. For each task, clone θ and run ``inner_steps`` SGD updates on the
     task's labelled data to obtain θ'_i.
  3. Update θ  ←  θ + ε * (mean(θ'_i) − θ).

Usage
-----
>>> trainer = ReptileTrainer(model)
>>> trainer.meta_train(task_dataset, epochs=100)
"""

from __future__ import annotations

import copy
import random
from typing import Callable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor

from .model import RecallLSTM


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

# A "task batch" is a list of (X, y) tensors where
#   X : (seq_len, input_size)  – feature sequence for *one* concept history
#   y : scalar float            – observed recall label (0 or 1)
TaskSample = List[Tuple[Tensor, Tensor]]


def build_task_sample(
    feature_sequence: list[list[float]],
    query_interval_days: float,
    recall_label: float,
) -> Tuple[Tensor, Tensor]:
    """Create a single (X, y) training example.

    Parameters
    ----------
    feature_sequence:
        ``[elapsed_days, score]`` pairs from :meth:`ConceptState.as_feature_sequence`.
    query_interval_days:
        The prospective interval being evaluated.
    recall_label:
        Ground-truth recall outcome (1.0 = recalled, 0.0 = forgot).
    """
    if not feature_sequence:
        feature_sequence = [[0.0, 0.0]]

    seq = []
    for i, (elapsed, score) in enumerate(feature_sequence):
        interval = query_interval_days if i == len(feature_sequence) - 1 else 0.0
        seq.append([elapsed, score, interval])

    x = torch.tensor(seq, dtype=torch.float32)
    y = torch.tensor(recall_label, dtype=torch.float32)
    return x, y


class ReptileTrainer:
    """Trains :class:`RecallLSTM` using the Reptile meta-learning algorithm.

    Parameters
    ----------
    model:
        The :class:`RecallLSTM` instance whose parameters will be meta-trained.
    inner_lr:
        Learning rate for the inner (task-level) SGD optimiser.
    inner_steps:
        Number of gradient steps taken per task in the inner loop.
    meta_lr:
        Step size ε for the Reptile outer update.
    device:
        Torch device to use (defaults to CPU).
    """

    def __init__(
        self,
        model: RecallLSTM,
        inner_lr: float = 0.01,
        inner_steps: int = 5,
        meta_lr: float = 0.1,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.meta_lr = meta_lr
        self.device = device or torch.device("cpu")
        self.model.to(self.device)
        self._criterion = nn.BCELoss()

    # ------------------------------------------------------------------
    # Inner loop: adapt a cloned model to one task
    # ------------------------------------------------------------------

    def _inner_update(
        self, task_samples: TaskSample
    ) -> RecallLSTM:
        """Run ``inner_steps`` SGD updates on a cloned model and return it."""
        fast_model = copy.deepcopy(self.model)
        fast_model.train()
        optimiser = optim.SGD(fast_model.parameters(), lr=self.inner_lr)

        for _ in range(self.inner_steps):
            random.shuffle(task_samples)
            total_loss = torch.tensor(0.0, device=self.device)

            for x, y in task_samples:
                x = x.unsqueeze(0).to(self.device)  # (1, seq_len, input_size)
                y_hat = fast_model(x)                # (1,)
                loss = self._criterion(y_hat, y.unsqueeze(0).to(self.device))
                total_loss += loss

            optimiser.zero_grad()
            avg_loss = total_loss / len(task_samples)
            avg_loss.backward()
            optimiser.step()

        return fast_model

    # ------------------------------------------------------------------
    # Outer (Reptile) update
    # ------------------------------------------------------------------

    def _reptile_update(self, adapted_models: List[RecallLSTM]) -> None:
        """Apply the Reptile update: θ ← θ + ε * (mean(θ'_i) − θ)."""
        meta_params = dict(self.model.named_parameters())

        # Accumulate the mean of adapted parameters
        mean_adapted: dict[str, Tensor] = {}
        for name, _ in meta_params.items():
            mean_adapted[name] = torch.zeros_like(meta_params[name])

        for adapted in adapted_models:
            for name, param in adapted.named_parameters():
                mean_adapted[name] = mean_adapted[name] + param.data

        n = len(adapted_models)
        with torch.no_grad():
            for name, meta_param in meta_params.items():
                theta_prime = mean_adapted[name] / n
                meta_param.data += self.meta_lr * (theta_prime - meta_param.data)

    # ------------------------------------------------------------------
    # Public training API
    # ------------------------------------------------------------------

    def meta_train(
        self,
        all_task_samples: List[TaskSample],
        epochs: int = 100,
        tasks_per_epoch: int = 4,
        on_epoch_end: Optional[Callable[[int, float], None]] = None,
    ) -> List[float]:
        """Run the Reptile meta-training loop.

        Parameters
        ----------
        all_task_samples:
            List of tasks; each task is a :data:`TaskSample` (list of
            ``(X, y)`` pairs representing one user's review history).
        epochs:
            Number of meta-update iterations.
        tasks_per_epoch:
            Number of tasks sampled each epoch.
        on_epoch_end:
            Optional callback ``fn(epoch, avg_inner_loss)`` called after each
            meta-update.

        Returns
        -------
        List[float]
            Per-epoch average inner-loop loss for monitoring.
        """
        if not all_task_samples:
            raise ValueError("all_task_samples must be non-empty")

        epoch_losses: List[float] = []

        for epoch in range(epochs):
            sampled = random.sample(
                all_task_samples,
                k=min(tasks_per_epoch, len(all_task_samples)),
            )

            adapted_models: List[RecallLSTM] = []
            inner_losses: List[float] = []

            for task in sampled:
                adapted = self._inner_update(task)
                adapted_models.append(adapted)

                # Record one-pass loss for monitoring
                adapted.eval()
                with torch.no_grad():
                    batch_loss = 0.0
                    for x, y in task:
                        x = x.unsqueeze(0).to(self.device)
                        y_hat = adapted(x)
                        loss = self._criterion(
                            y_hat, y.unsqueeze(0).to(self.device)
                        )
                        batch_loss += loss.item()
                    inner_losses.append(batch_loss / len(task))

            self._reptile_update(adapted_models)

            avg_loss = sum(inner_losses) / len(inner_losses)
            epoch_losses.append(avg_loss)

            if on_epoch_end is not None:
                on_epoch_end(epoch, avg_loss)

        return epoch_losses

    # ------------------------------------------------------------------
    # Fine-tune on a single user's data (inference-time personalisation)
    # ------------------------------------------------------------------

    def personalise(
        self,
        task_samples: TaskSample,
        steps: Optional[int] = None,
    ) -> RecallLSTM:
        """Return a fine-tuned copy of the model adapted to one user.

        Parameters
        ----------
        task_samples:
            The user's ``(X, y)`` training pairs.
        steps:
            Override ``inner_steps`` for personalisation (defaults to
            ``self.inner_steps``).

        Returns
        -------
        RecallLSTM
            A personalised model (the meta-model is *not* modified).
        """
        original_steps = self.inner_steps
        if steps is not None:
            self.inner_steps = steps
        personalised = self._inner_update(task_samples)
        self.inner_steps = original_steps
        return personalised
