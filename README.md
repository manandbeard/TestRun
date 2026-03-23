# Personalized Spaced-Repetition Scheduler

A personalized spaced-repetition scheduler built on an **LSTM neural network** trained with a **Reptile-style meta-learning** loop.

## Overview

The system learns a weight initialisation that can be quickly adapted (personalised) to any new student with just a few gradient steps — this is the key insight behind the [Reptile](https://arxiv.org/abs/1803.02999) meta-learning algorithm.

```
┌─────────────────────────────────────────────────────────────┐
│                   Reptile Meta-Training                      │
│                                                             │
│   For each epoch:                                           │
│     sample K users  →  inner SGD on each user's reviews     │
│     outer update:  θ ← θ + ε · (mean(θ'_i) − θ)           │
└────────────────────────┬────────────────────────────────────┘
                         │  meta-weights θ
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Personalisation (inference time)                │
│                                                             │
│   new_user_model = fine-tune θ on user's K review pairs     │
└────────────────────────┬────────────────────────────────────┘
                         │  personalised model
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              SpacedRepetitionScheduler                       │
│                                                             │
│   binary-search for largest interval I such that            │
│   P(recall | history, I) ≥ target_recall (default 0.90)     │
└─────────────────────────────────────────────────────────────┘
```

## Components

| File | Description |
|------|-------------|
| `scheduler/data.py` | Data models: `ReviewEvent`, `ConceptState`, `UserState` |
| `scheduler/model.py` | `RecallLSTM` – predicts P(recall) for a queried future interval |
| `scheduler/reptile.py` | `ReptileTrainer` – Reptile meta-learning loop + personalisation |
| `scheduler/scheduler.py` | `SpacedRepetitionScheduler` – binary-search scheduling |
| `main.py` | End-to-end demo script |
| `tests/` | Pytest unit tests |

## Quick Start

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the demo

```bash
python main.py
```

The demo generates synthetic review histories for 10 simulated users, runs 50 epochs of Reptile meta-training, personalises the model to a new student, and prints the recommended next-review interval for each concept.

### Run tests

```bash
pytest tests/ -v
```

## Architecture Details

### RecallLSTM

Input features per time step: `[elapsed_days, score, query_interval]`
- `elapsed_days` — fractional days since the previous review of this concept
- `score` — normalised recall performance `[0, 1]`
- `query_interval` — the prospective future interval being evaluated (appended only at the last step)

The LSTM reads the full review history and its final hidden state is projected through a linear layer + sigmoid to produce P(recall).

### Reptile Meta-Learning

Reptile is a first-order meta-learning algorithm (no second-order gradients).  The outer loop maintains a set of meta-parameters θ that act as a warm initialisation.  For each meta-iteration:

1. Sample a mini-batch of user "tasks".
2. Clone θ and run `inner_steps` SGD updates per task → θ'_i.
3. Update: **θ ← θ + ε · (mean(θ'_i) − θ)**.

At inference time, `trainer.personalise(user_task)` clones the meta-model and fine-tunes it on the target user's reviews in just a few steps.

### Spaced-Repetition Scheduling

The scheduler performs a binary search over `[min_interval, max_interval]` (days) to find the largest interval I* such that the personalised model predicts recall probability ≥ `target_recall` (default 0.9).

```python
from scheduler import RecallLSTM, ReptileTrainer, SpacedRepetitionScheduler, UserState

# After meta-training...
personal_model = trainer.personalise(user_task_samples)
sched = SpacedRepetitionScheduler(personal_model, target_recall=0.9)

user = UserState(user_id="alice")
cs = user.get_or_create_concept("photosynthesis")
cs.add_review(score=0.85)

interval_days = sched.next_interval(cs)
print(f"Review again in {interval_days:.1f} days")
```
