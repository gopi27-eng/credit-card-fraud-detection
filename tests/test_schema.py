import pytest
from pydantic import ValidationError
from src.schemas.transaction import (
    MerchantCategory,TransactionInferenceRequest,
    TransactionRecord,PredictionOutput
)


@pytest.fixture
def valid_record_data() -> dict:
    """Returns a valid transaction record dictionary matching training requirements."""
    return {
        "transaction_id": 1,
        "amount": 84.47,
        "transaction_hour": 22,
        "merchant_category": "Electronics",
        "foreign_transaction": 0,
        "location_mismatch": 0,
        "device_trust_score": 66,
        "velocity_last_24h": 3,
        "cardholder_age": 40,
        "is_fraud": 0,
    }
 
 
def test_valid_transaction_record(valid_record_data: dict):
    """Test successful instantiation of a fully populated training record."""
    record = TransactionRecord(**valid_record_data)
    assert record.transaction_id == 1
    assert record.amount == 84.47
    assert record.merchant_category == MerchantCategory.ELECTRONICS
    assert record.is_fraud == 0
    
def test_valid_inference_request(valid_record_data: dict):
    """Test inference payload where the target column 'is_fraud' is absent."""
    inference_data = valid_record_data.copy()
    del inference_data["is_fraud"]

    req = TransactionInferenceRequest(**inference_data)
    assert req.transaction_id == 1
    assert req.cardholder_age == 40

@pytest.mark.parametrize(
    "invalid_field, invalid_value",
    [
        ("amount", -10.0),            # amount must be strictly > 0
        ("amount", 0.0),              # amount must be strictly > 0
        ("transaction_hour", -1),      # hour must be >= 0
        ("transaction_hour", 24),     # hour must be <= 23
        ("device_trust_score", -1),   # trust score must be >= 0
        ("device_trust_score", 101),  # trust score must be <= 100
        ("velocity_last_24h", -1),    # velocity must be >= 0
        ("cardholder_age", 17),       # cardholder age must be >= 18
        ("cardholder_age", 121),      # cardholder age must be <= 120
        ("foreign_transaction", 2),   # binary flag must be 0 or 1
        ("location_mismatch", -1),    # binary flag must be 0 or 1
        ("merchant_category", "Crypto"),  # non-existent category
    ],
)


def test_schema_field_boundary_failures(valid_record_data: dict, invalid_field: str, invalid_value):
    """Test that boundary violations raise a ValidationError."""
    payload = valid_record_data.copy()
    payload[invalid_field] = invalid_value

    with pytest.raises(ValidationError):
        TransactionRecord(**payload)
        
def test_schema_extra_fields_forbidden(valid_record_data: dict):
    """Test that unmapped extra fields trigger validation failure (extra='forbid')."""
    payload = valid_record_data.copy()
    payload["unexpected_injected_column"] = 999

    with pytest.raises(ValidationError):
        TransactionRecord(**payload)


def test_prediction_output_schema():
    """Test the structure and constraints of the prediction output model."""
    output = PredictionOutput(
        transaction_id=101,
        is_fraud=1,
        fraud_probability=0.875,
    )
    assert output.transaction_id == 101
    assert output.is_fraud == 1
    assert 0.0 <= output.fraud_probability <= 1.0

    # Ensure out-of-range probabilities fail
    with pytest.raises(ValidationError):
        PredictionOutput(transaction_id=102, is_fraud=1, fraud_probability=1.2)
