import numpy as np
from fastapi.testclient import TestClient

from recsys import evaluate
from server import app
from tests.test_recsys import small_model

client = TestClient(app)


def test_catalog_has_movies_and_eval():
    r = client.get("/api/catalog")
    assert r.status_code == 200
    body = r.json()
    assert len(body["movies"]) > 0
    assert "rmse_test" in body["eval"]


def test_recommend_requires_min_ratings():
    r = client.post("/api/recommend", json={"ratings": {"1": 5}})
    assert r.json().get("error")


def test_recommend_rejects_out_of_range_rating():
    r = client.post("/api/recommend", json={"ratings": {"1": 9, "50": 5, "100": 4}})
    assert r.status_code == 422


def test_recommend_happy_path():
    r = client.post("/api/recommend", json={"ratings": {"1": 5, "50": 5, "100": 4}})
    body = r.json()
    assert len(body["recommendations"]) == 10
    assert all("title" in rec for rec in body["recommendations"])


def test_precision_recall_no_liked_users_returns_zero():
    m = small_model()
    train = (np.array([0]), np.array([0]), np.array([5.0]), {0: 0}, {0: 0})
    test = (np.array([0]), np.array([1]), np.array([1.0]))   # no rating >= relevant
    p, r = evaluate.precision_recall_at_k(m, train, test, k=2)
    assert p == 0.0 and r == 0.0
