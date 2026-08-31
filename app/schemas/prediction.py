"""
Pydantic schemas for AI service request/response models.
"""
from pydantic import BaseModel, Field
from typing import Optional, List


class PredictionResponse(BaseModel):
    """Response schema for disease prediction."""
    disease: str = Field(..., description="Detected disease name")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence (0-1)")
    severity: float = Field(..., ge=0, le=100, description="Disease severity percentage (0-100)")
    affected_area: float = Field(..., ge=0, le=100, description="Estimated affected area percentage")
    risk_level: str = Field(..., description="Risk level: LOW, MODERATE, HIGH, CRITICAL")
    explanation: str = Field(..., description="Human-readable explanation of the diagnosis")
    is_demo: bool = Field(default=True, description="Whether this is a demo/prototype result")


class AnalysisResponse(BaseModel):
    """Response schema for detailed image analysis."""
    prediction: PredictionResponse
    image_stats: dict = Field(default_factory=dict, description="Image statistics from OpenCV analysis")
    recommendations: List[str] = Field(default_factory=list, description="AI-suggested recommendations")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "healthy"
    service: str = "crop-health-ai"
    version: str = "1.0.0"
    model_loaded: bool = False
    demo_mode: bool = True
