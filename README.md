# Personalized Spaced-Repetition Scheduler

A personalized spaced-repetition scheduler built on an **attention-augmented LSTM neural network** trained with a **Reptile-style meta-learning** loop, incorporating the latest findings from the science of learning.

## Research-Backed Improvements

| Improvement | Research Basis |
|-------------|---------------|
| **Power-law forgetting curve** | Wixted & Ebbesen (1991), Wixted (2004) — empirical data shows forgetting follows a power law `R(t) = (1 + t/S)^(-1)` rather than exponential decay |
| **FSRS-inspired D/S tracking** | Ye (2023) — per-concept difficulty and stability parameters dynamically updated on each review |
| **Attention over review history** | Bjork & Bjork (2011) desirable-difficulties theory — not all past reviews are equally predictive; learned attention weights let the model focus on the most informative ones |
| **Cosine-annealed meta-LR** | Loshchilov & Hutter (2016) — smooth decay of the Reptile outer learning rate improves convergence |
| **Gradient clipping** | Pascanu et al. (2013) — prevents exploding gradients in LSTM inner loop |
| **Interleaved scheduling** | Rohrer et al. (2015), Kornell & Bjork (2008) — interleaving different categories during review sessions improves discriminative learning |
| **Desirable-difficulty retrieval bonus** | Bjork & Bjork (2011) — stability grows more when retrieval was harder (lower retrievability at review time) |

## Overview

The system learns a weight initialisation that can be quickly adapted (personalised) to any new student with just a few gradient steps — this is the key insight behind the [Reptile](https://arxiv.org/abs/1803.02999) meta-learning algorithm.

```
┌─────────────────────────────────────────────────────────────┐
│                   Reptile Meta-Training                      │
│                                                             │
│   For each epoch:                                           │
│     sample K users  →  inner SGD on each user's reviews     │
│     outer update:  θ ← θ + ε_t · (mean(θ'_i) − θ)         │
│     (ε_t follows cosine annealing schedule)                 │
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
│   optional: interleave concepts by category                 │
└─────────────────────────────────────────────────────────────┘
```

## Components

| File | Description |
|------|-------------|
| `scheduler/data.py` | Data models: `ReviewEvent`, `ConceptState` (with FSRS-inspired D/S tracking), `UserState` |
| `scheduler/model.py` | `RecallLSTM` – attention-augmented LSTM predicting P(recall), with save/load support |
| `scheduler/reptile.py` | `ReptileTrainer` – Reptile meta-learning with cosine LR + gradient clipping |
| `scheduler/scheduler.py` | `SpacedRepetitionScheduler` – binary-search scheduling + interleaving |
| `app.py` | Flask web application with REST API and single-page frontend |
| `train_meta_model.py` | Pre-train a Reptile meta-model and save to `meta_model.pt` |
| `templates/index.html` | Browser UI for study scheduling |
| `main.py` | CLI demo script |
| `tests/` | Pytest unit tests (83 tests) |

## Quick Start

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run as a web app

```bash
# 1. Pre-train the meta-model (one-time, ~2 min)
python train_meta_model.py

# 2. Start the web server
python app.py
```

Open http://localhost:5000 in your browser. Enter a user ID, add concepts with categories, record review scores, and see your personalised schedule with FSRS-tracked difficulty/stability and optional interleaving.

### Run the CLI demo

```bash
python main.py
```

### Run tests

```bash
pytest tests/ -v
```

### REST API

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/users` | Create/get a user (`{"user_id": "alice"}`) |
| `GET`  | `/api/users/<id>` | Get full user state |
| `POST` | `/api/users/<id>/concepts` | Add a concept (`{"concept_id": "...", "category": "..."}`) |
| `POST` | `/api/users/<id>/concepts/<cid>/review` | Record a review (`{"score": 0.85}`) |
| `GET`  | `/api/users/<id>/schedule[?interleave=true]` | Get personalised review schedule |
| `GET`  | `/api/users/<id>/concepts/<cid>/recall-curve` | Get predicted recall probabilities |

## Architecture Details

### RecallLSTM (Attention-Augmented)

Input features per time step: `[elapsed_days, score, difficulty, stability, norm_review_count, query_interval]`
- `elapsed_days` — fractional days since the previous review of this concept
- `score` — normalised recall performance `[0, 1]`
- `difficulty` — FSRS-inspired concept difficulty at this time step `[0, 1]`
- `stability` — current memory stability in days (larger → slower forgetting)
- `norm_review_count` — normalised review index (progress through history)
- `query_interval` — the prospective future interval being evaluated (appended only at the last step)

The LSTM reads the full review history, and a **learned attention mechanism** computes importance weights over all time steps.  The weighted sum of hidden states is projected through a linear layer + sigmoid to produce P(recall).  This lets the model automatically learn that recent reviews and difficult retrievals carry more predictive signal.

### FSRS-Inspired State Model

Each `ConceptState` tracks two latent variables updated after every review:

- **Difficulty (D)** — how hard the concept is for this student.  Decreases on high scores, increases on low scores, with mean-reversion to population average.
- **Stability (S)** — memory stability in days.  Grows on successful recall (with a desirable-difficulty bonus when retrieval was harder), shrinks on failure.

The **power-law forgetting curve** `R(t) = (1 + t/S)^(-1)` replaces the classic exponential Ebbinghaus curve and better fits empirical forgetting data.

### Reptile Meta-Learning

Reptile is a first-order meta-learning algorithm (no second-order gradients).  The outer loop maintains a set of meta-parameters θ that act as a warm initialisation.  For each meta-iteration:

1. Sample a mini-batch of user "tasks".
2. Clone θ and run `inner_steps` SGD updates per task (with gradient clipping) → θ'_i.
3. Update: **θ ← θ + ε_t · (mean(θ'_i) − θ)**, where ε_t follows a **cosine annealing schedule**.

At inference time, `trainer.personalise(user_task)` clones the meta-model and fine-tunes it on the target user's reviews in just a few steps.

### Spaced-Repetition Scheduling

The scheduler performs a binary search over `[min_interval, max_interval]` (days) to find the largest interval I* such that the personalised model predicts recall probability ≥ `target_recall` (default 0.9).

The `due_concepts()` method supports an **interleave** mode that reorders the review schedule to maximise category diversity, based on research showing that interleaved practice improves discriminative learning.

```python
from scheduler import RecallLSTM, ReptileTrainer, SpacedRepetitionScheduler, UserState

# After meta-training...
personal_model = trainer.personalise(user_task_samples)
sched = SpacedRepetitionScheduler(personal_model, target_recall=0.9)

user = UserState(user_id="alice")
cs = user.get_or_create_concept("photosynthesis", category="biology")
cs.add_review(score=0.85)

interval_days = sched.next_interval(cs)
print(f"Review again in {interval_days:.1f} days")
print(f"Concept difficulty: {cs.difficulty:.2f}, stability: {cs.stability:.1f} days")

# Interleaved schedule across all concepts
due = sched.due_concepts(user, interleave=True)
```
