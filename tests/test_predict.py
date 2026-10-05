import pytest
from src.models.predict import FraudPredictor
from src.schemas.transaction import MerchantCategory, TransactionInferenceRequest
from src.utils.config import load_config


def test_fraud_predictor_single_payload():
    config = load_config()
    predictor = FraudPredictor(config, threshold=0.5)

    sample_transaction = TransactionInferenceRequest(
        transaction_id=999999,
        amount=250.75,
        transaction_hour=3,
        device_trust_score=15,
        velocity_last_24h=7,
        cardholder_age=45,
        merchant_category=MerchantCategory.ELECTRONICS,
        foreign_transaction=1,
        location_mismatch=1,
    )

    results = predictor.predict([sample_transaction])

    assert len(results) == 1
    assert results[0]["transaction_id"] == 999999
    assert 0.0 <= results[0]["fraud_probability"] <= 1.0
    assert isinstance(results[0]["is_fraud"], bool)