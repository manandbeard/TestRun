"""Flask web application for the spaced-repetition scheduler.

Exposes a REST API that lets browser clients:

* Create users and concepts
* Record reviews (scores)
* Retrieve personalised scheduling recommendations

State is stored in-memory (keyed by ``user_id``).  A pre-trained
meta-model checkpoint (``meta_model.pt``) is loaded at startup and used
to personalise a per-user model on the fly.

Run:
    python app.py
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Dict

from flask import Flask, jsonify, render_template, request

import torch

from scheduler import (
    RecallLSTM,
    ReptileTrainer,
    SpacedRepetitionScheduler,
    UserState,
)
from scheduler.reptile import build_task_sample, TaskSample

# ---------------------------------------------------------------------------
# Application setup
# ---------------------------------------------------------------------------

app = Flask(__name__)

# In-memory stores --------------------------------------------------------
_users: Dict[str, UserState] = {}
_user_tasks: Dict[str, TaskSample] = {}

# Model paths
_META_MODEL_PATH = os.environ.get(
    "META_MODEL_PATH",
    os.path.join(os.path.dirname(__file__), "meta_model.pt"),
)

# Global model / trainer (initialised in _init_model)
_meta_model: RecallLSTM | None = None
_trainer: ReptileTrainer | None = None


def _init_model() -> None:
    """Load (or create) the meta-model and trainer."""
    global _meta_model, _trainer  # noqa: PLW0603

    if os.path.isfile(_META_MODEL_PATH):
        _meta_model = RecallLSTM.load(_META_MODEL_PATH)
    else:
        # Fall back to a freshly-initialised model (no pre-training)
        _meta_model = RecallLSTM(
            input_size=6, hidden_size=64, num_layers=2, dropout=0.1,
        )

    _trainer = ReptileTrainer(
        _meta_model,
        inner_lr=0.01,
        inner_steps=5,
        meta_lr=0.1,
        meta_lr_min=0.001,
        max_grad_norm=5.0,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_user(user_id: str) -> UserState:
    if user_id not in _users:
        _users[user_id] = UserState(user_id=user_id)
        _user_tasks[user_id] = []
    return _users[user_id]


def _personalise(user_id: str) -> RecallLSTM:
    """Return a personalised model for *user_id*."""
    assert _trainer is not None
    task = _user_tasks.get(user_id, [])
    if not task:
        # No data yet – return the meta-model itself
        assert _meta_model is not None
        return _meta_model
    return _trainer.personalise(task, steps=10)


# ---------------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


# ---------------------------------------------------------------------------
# REST API
# ---------------------------------------------------------------------------

@app.route("/api/users", methods=["POST"])
def create_user():
    """Create or retrieve a user.

    JSON body: ``{"user_id": "alice"}``
    """
    data = request.get_json(silent=True) or {}
    user_id = data.get("user_id", "").strip()
    if not user_id:
        return jsonify({"error": "user_id is required"}), 400

    user = _get_user(user_id)
    return jsonify(user.to_dict()), 201


@app.route("/api/users/<user_id>", methods=["GET"])
def get_user(user_id: str):
    """Return the full state for a user."""
    if user_id not in _users:
        return jsonify({"error": "user not found"}), 404
    return jsonify(_users[user_id].to_dict())


@app.route("/api/users/<user_id>/concepts", methods=["POST"])
def add_concept(user_id: str):
    """Create a concept for a user.

    JSON body: ``{"concept_id": "photosynthesis", "category": "biology"}``
    """
    data = request.get_json(silent=True) or {}
    concept_id = data.get("concept_id", "").strip()
    if not concept_id:
        return jsonify({"error": "concept_id is required"}), 400

    category = data.get("category", "")
    user = _get_user(user_id)
    cs = user.get_or_create_concept(concept_id, category=category)
    return jsonify(cs.to_dict()), 201


@app.route("/api/users/<user_id>/concepts/<concept_id>/review", methods=["POST"])
def add_review(user_id: str, concept_id: str):
    """Record a review for a concept.

    JSON body: ``{"score": 0.85}``
    """
    data = request.get_json(silent=True) or {}
    try:
        score = float(data.get("score", -1))
    except (TypeError, ValueError):
        return jsonify({"error": "score must be a number in [0, 1]"}), 400

    if not 0.0 <= score <= 1.0:
        return jsonify({"error": "score must be in [0, 1]"}), 400

    user = _get_user(user_id)
    if concept_id not in user.concept_states:
        return jsonify({"error": "concept not found — add it first"}), 404

    cs = user.concept_states[concept_id]
    ts = datetime.now(tz=timezone.utc)
    cs.add_review(score=score, timestamp=ts)

    # Build a training sample for personalisation
    features = cs.as_feature_sequence()
    # Use the current interval as query; label is the observed score binarised
    elapsed = cs.reviews[-1].elapsed_days if cs.reviews else 0.0
    label = float(score >= 0.6)
    sample = build_task_sample(features, max(elapsed, 1.0), label)
    _user_tasks.setdefault(user_id, []).append(sample)

    return jsonify({
        "concept_id": concept_id,
        "review_count": cs.review_count,
        "difficulty": round(cs.difficulty, 4),
        "stability": round(cs.stability, 4),
        "score": score,
    }), 201


@app.route("/api/users/<user_id>/schedule", methods=["GET"])
def get_schedule(user_id: str):
    """Return the recommended next-review schedule for all concepts."""
    if user_id not in _users:
        return jsonify({"error": "user not found"}), 404

    user = _users[user_id]
    if not user.concept_states:
        return jsonify({"schedule": []})

    interleave = request.args.get("interleave", "false").lower() == "true"
    model = _personalise(user_id)
    scheduler = SpacedRepetitionScheduler(model, target_recall=0.9)
    due = scheduler.due_concepts(user, interleave=interleave)

    schedule = []
    for concept_id, interval in due:
        cs = user.concept_states[concept_id]
        schedule.append({
            "concept_id": concept_id,
            "category": cs.category,
            "interval_days": round(interval, 2),
            "difficulty": round(cs.difficulty, 4),
            "stability": round(cs.stability, 4),
            "review_count": cs.review_count,
        })

    return jsonify({"schedule": schedule})


@app.route("/api/users/<user_id>/concepts/<concept_id>/recall-curve", methods=["GET"])
def get_recall_curve(user_id: str, concept_id: str):
    """Return the predicted recall probabilities at several intervals."""
    if user_id not in _users:
        return jsonify({"error": "user not found"}), 404

    user = _users[user_id]
    if concept_id not in user.concept_states:
        return jsonify({"error": "concept not found"}), 404

    model = _personalise(user_id)
    scheduler = SpacedRepetitionScheduler(model, target_recall=0.9)
    cs = user.concept_states[concept_id]

    intervals = [0.5, 1, 2, 3, 5, 7, 14, 21, 30, 60, 90, 180, 365]
    probs = scheduler.recall_curve(cs, intervals)

    return jsonify({
        "concept_id": concept_id,
        "curve": [
            {"interval_days": iv, "recall_probability": round(p, 4)}
            for iv, p in zip(intervals, probs)
        ],
    })


# ---------------------------------------------------------------------------
# Application entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _init_model()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
