import pytest
from fastapi.testclient import TestClient
from src.api.app import app


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "service": "credit-card-fraud-detection"}


def test_single_prediction_endpoint(client):
    payload = {
        "transaction_id": 1001,
        "amount": 25.50,
        "transaction_hour": 14,
        "device_trust_score": 90,
        "velocity_last_24h": 1,
        "cardholder_age": 34,
        "merchant_category": "Grocery",
        "foreign_transaction": 0,
        "location_mismatch": 0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["transaction_id"] == 1001
    assert data["is_fraud"] in [0, 1]
    assert 0.0 <= data["fraud_probability"] <= 1.0


def test_batch_prediction_endpoint(client):
    payload = {
        "instances": [
            {
                "transaction_id": 2001,
                "amount": 10.0,
                "transaction_hour": 10,
                "device_trust_score": 85,
                "velocity_last_24h": 2,
                "cardholder_age": 28,
                "merchant_category": "Food",
                "foreign_transaction": 0,
                "location_mismatch": 0,
            },
            {
                "transaction_id": 2002,
                "amount": 950.0,
                "transaction_hour": 3,
                "device_trust_score": 10,
                "velocity_last_24h": 8,
                "cardholder_age": 45,
                "merchant_category": "Electronics",
                "foreign_transaction": 1,
                "location_mismatch": 1,
            },
        ]
    }
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    assert data[0]["transaction_id"] == 2001
    assert data[1]["transaction_id"] == 2002