"""Tests for :mod:`app` (Flask web application)."""

from __future__ import annotations

import json

import pytest
import torch

from app import app, _init_model, _users, _user_tasks


@pytest.fixture(autouse=True)
def _reset_state():
    """Clear global in-memory stores and re-init the model before each test."""
    _users.clear()
    _user_tasks.clear()
    _init_model()
    yield
    _users.clear()
    _user_tasks.clear()


@pytest.fixture()
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


# ---------------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------------

class TestFrontend:
    def test_index_returns_html(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert b"Spaced Repetition Scheduler" in resp.data


# ---------------------------------------------------------------------------
# User endpoints
# ---------------------------------------------------------------------------

class TestCreateUser:
    def test_create_user(self, client):
        resp = client.post("/api/users",
                           json={"user_id": "alice"})
        assert resp.status_code == 201
        data = resp.get_json()
        assert data["user_id"] == "alice"
        assert data["concept_states"] == {}

    def test_create_user_idempotent(self, client):
        client.post("/api/users", json={"user_id": "alice"})
        resp = client.post("/api/users", json={"user_id": "alice"})
        assert resp.status_code == 201
        assert resp.get_json()["user_id"] == "alice"

    def test_create_user_missing_id(self, client):
        resp = client.post("/api/users", json={})
        assert resp.status_code == 400

    def test_create_user_empty_id(self, client):
        resp = client.post("/api/users", json={"user_id": ""})
        assert resp.status_code == 400


class TestGetUser:
    def test_get_existing_user(self, client):
        client.post("/api/users", json={"user_id": "bob"})
        resp = client.get("/api/users/bob")
        assert resp.status_code == 200
        assert resp.get_json()["user_id"] == "bob"

    def test_get_nonexistent_user(self, client):
        resp = client.get("/api/users/ghost")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Concept endpoints
# ---------------------------------------------------------------------------

class TestAddConcept:
    def test_add_concept(self, client):
        client.post("/api/users", json={"user_id": "alice"})
        resp = client.post("/api/users/alice/concepts",
                           json={"concept_id": "photosynthesis",
                                 "category": "biology"})
        assert resp.status_code == 201
        data = resp.get_json()
        assert data["concept_id"] == "photosynthesis"
        assert data["category"] == "biology"

    def test_add_concept_missing_id(self, client):
        client.post("/api/users", json={"user_id": "alice"})
        resp = client.post("/api/users/alice/concepts", json={})
        assert resp.status_code == 400

    def test_add_concept_creates_user_implicitly(self, client):
        resp = client.post("/api/users/newbie/concepts",
                           json={"concept_id": "algebra"})
        assert resp.status_code == 201


# ---------------------------------------------------------------------------
# Review endpoints
# ---------------------------------------------------------------------------

class TestAddReview:
    def _setup_concept(self, client):
        client.post("/api/users", json={"user_id": "alice"})
        client.post("/api/users/alice/concepts",
                     json={"concept_id": "mitosis", "category": "biology"})

    def test_add_review(self, client):
        self._setup_concept(client)
        resp = client.post(
            "/api/users/alice/concepts/mitosis/review",
            json={"score": 0.8},
        )
        assert resp.status_code == 201
        data = resp.get_json()
        assert data["concept_id"] == "mitosis"
        assert data["review_count"] == 1
        assert data["score"] == 0.8

    def test_add_review_updates_difficulty(self, client):
        self._setup_concept(client)
        resp = client.post(
            "/api/users/alice/concepts/mitosis/review",
            json={"score": 1.0},
        )
        data = resp.get_json()
        # High score should decrease difficulty from the default 0.3
        assert data["difficulty"] < 0.3

    def test_add_review_invalid_score(self, client):
        self._setup_concept(client)
        resp = client.post(
            "/api/users/alice/concepts/mitosis/review",
            json={"score": 2.0},
        )
        assert resp.status_code == 400

    def test_add_review_missing_concept(self, client):
        client.post("/api/users", json={"user_id": "alice"})
        resp = client.post(
            "/api/users/alice/concepts/ghost/review",
            json={"score": 0.5},
        )
        assert resp.status_code == 404

    def test_multiple_reviews_increment_count(self, client):
        self._setup_concept(client)
        client.post("/api/users/alice/concepts/mitosis/review",
                     json={"score": 0.8})
        resp = client.post("/api/users/alice/concepts/mitosis/review",
                           json={"score": 0.6})
        assert resp.get_json()["review_count"] == 2


# ---------------------------------------------------------------------------
# Schedule endpoint
# ---------------------------------------------------------------------------

class TestSchedule:
    def _setup_user(self, client):
        client.post("/api/users", json={"user_id": "alice"})
        for cid, cat in [("c1", "math"), ("c2", "science")]:
            client.post("/api/users/alice/concepts",
                         json={"concept_id": cid, "category": cat})
            client.post(f"/api/users/alice/concepts/{cid}/review",
                         json={"score": 0.8})

    def test_schedule_returns_list(self, client):
        self._setup_user(client)
        resp = client.get("/api/users/alice/schedule")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "schedule" in data
        assert len(data["schedule"]) == 2

    def test_schedule_contains_expected_fields(self, client):
        self._setup_user(client)
        resp = client.get("/api/users/alice/schedule")
        entry = resp.get_json()["schedule"][0]
        assert "concept_id" in entry
        assert "interval_days" in entry
        assert "difficulty" in entry
        assert "stability" in entry

    def test_schedule_interleave_flag(self, client):
        self._setup_user(client)
        resp = client.get("/api/users/alice/schedule?interleave=true")
        assert resp.status_code == 200
        assert len(resp.get_json()["schedule"]) == 2

    def test_schedule_empty_user(self, client):
        client.post("/api/users", json={"user_id": "empty"})
        resp = client.get("/api/users/empty/schedule")
        assert resp.status_code == 200
        assert resp.get_json()["schedule"] == []

    def test_schedule_unknown_user(self, client):
        resp = client.get("/api/users/ghost/schedule")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Recall curve endpoint
# ---------------------------------------------------------------------------

class TestRecallCurve:
    def _setup(self, client):
        client.post("/api/users", json={"user_id": "alice"})
        client.post("/api/users/alice/concepts",
                     json={"concept_id": "gravity", "category": "physics"})
        client.post("/api/users/alice/concepts/gravity/review",
                     json={"score": 0.7})

    def test_recall_curve_returns_points(self, client):
        self._setup(client)
        resp = client.get("/api/users/alice/concepts/gravity/recall-curve")
        assert resp.status_code == 200
        data = resp.get_json()
        assert len(data["curve"]) > 0
        assert "interval_days" in data["curve"][0]
        assert "recall_probability" in data["curve"][0]

    def test_recall_curve_values_in_range(self, client):
        self._setup(client)
        resp = client.get("/api/users/alice/concepts/gravity/recall-curve")
        for pt in resp.get_json()["curve"]:
            assert 0.0 <= pt["recall_probability"] <= 1.0

    def test_recall_curve_unknown_user(self, client):
        resp = client.get("/api/users/ghost/concepts/x/recall-curve")
        assert resp.status_code == 404

    def test_recall_curve_unknown_concept(self, client):
        client.post("/api/users", json={"user_id": "alice"})
        resp = client.get("/api/users/alice/concepts/ghost/recall-curve")
        assert resp.status_code == 404
