from enum import Enum
from typing import Literal,List
from pydantic import BaseModel, ConfigDict, Field

class MerchantCategory(str, Enum):
    ELECTRONICS = "Electronics"
    TRAVEL = "Travel"
    GROCERY = "Grocery"
    FOOD = "Food"
    CLOTHING = "Clothing"
    
class TransactionBase(BaseModel):
    """Base transaction attributes received during inference."""

    amount: float = Field(
        ...,
        gt=0.0,
        description="Monetary transaction amount. Must be strictly positive.",
    )
    transaction_hour: int = Field(
        ...,
        ge=0,
        le=23,
        description="Hour of transaction in 24-hour format (0 to 23).",
    )
    merchant_category: MerchantCategory = Field(
        ...,
        description="Category classification of the merchant merchant.",
    )
    foreign_transaction: int = Field(
        ...,
        ge=0,
        le=1,
        description="Indicator flag for cross-border/foreign charge (0 or 1).",
    )
    location_mismatch: int = Field(
        ...,
        ge=0,
        le=1,
        description="Mismatch between billing and physical transaction location (0 or 1).",
    )
    device_trust_score: int = Field(
        ...,
        ge=0,
        le=100,
        description="Device telemetry risk score ranging from 0 (untrusted) to 100 (fully verified).",
    )
    velocity_last_24h: int = Field(
        ...,
        ge=0,
        description="Total count of transactions initiated by this cardholder within past 24 hours.",
    )
    cardholder_age: int = Field(
        ...,
        ge=18,
        le=120,
        description="Age of primary cardholder in years.",
    )

    model_config = ConfigDict(
        extra="forbid",
        use_enum_values=True,
    )
class TransactionInferenceRequest(TransactionBase):
    """Payload sent by clients/gateways for real-time model scoring."""

    transaction_id: int = Field(
        ...,
        description="Unique identifier for the incoming transaction.",
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "transaction_id": 1,
                "amount": 84.47,
                "transaction_hour": 22,
                "merchant_category": "Electronics",
                "foreign_transaction": 0,
                "location_mismatch": 0,
                "device_trust_score": 66,
                "velocity_last_24h": 3,
                "cardholder_age": 40,
            }
        },
    )
    
class TransactionRecord(TransactionInferenceRequest):
    """Full schema for historical records used in training and drift monitoring."""

    is_fraud: int = Field(
        ...,
        ge=0,
        le=1,
        description="Ground truth fraud indicator: 0 (legitimate), 1 (fraud).",
    )
class BatchTransactionInferenceRequest(BaseModel):
    """Container schema for high-throughput batch scoring requests."""

    instances: List[TransactionInferenceRequest]


class PredictionOutput(BaseModel):
    """Standardized prediction response schema for KServe / serving runtimes."""

    transaction_id: int
    is_fraud: int = Field(..., ge=0, le=1)
    fraud_probability: float = Field(..., ge=0.0, le=1.0)
    
TransactionInput = TransactionInferenceRequest