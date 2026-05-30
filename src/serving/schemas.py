from pydantic import BaseModel, Field
from typing import Optional


class PredictRequest(BaseModel):
    """Request schema for prediction endpoint."""
    text: str = Field(
        ...,
        min_length=1,
        description="Input text to classify"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "text": "I feel so hopeless and empty inside"
            }
        }


class PredictResponse(BaseModel):
    """Response schema for prediction endpoint."""
    text: str = Field(description="Input text")
    prediction: int = Field(description="0=Non-Depression, 1=Depression")
    label: str = Field(description="Human readable label")
    confidence: float = Field(description="Model confidence score")


class HealthResponse(BaseModel):
    """Response schema for health endpoint."""
    status: str = Field(description="API status")
    model_name: str = Field(description="Loaded model name")
    model_version: str = Field(description="Loaded model version")
    experiment_name: str = Field(description="MLflow experiment name")


class BatchPredictRequest(BaseModel):
    """Request schema for batch prediction endpoint."""
    texts: list[str] = Field(
        ...,
        min_length=1,
        description="List of texts to classify"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "texts": [
                    "I feel so hopeless and empty inside",
                    "Today was a great day I feel amazing"
                ]
            }
        }


class BatchPredictResponse(BaseModel):
    """Response schema for batch prediction endpoint."""
    predictions: list[PredictResponse]
    total: int = Field(description="Total number of predictions")