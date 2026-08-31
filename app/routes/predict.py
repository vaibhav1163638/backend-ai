"""
Prediction and analysis route handlers.
"""
from fastapi import APIRouter, UploadFile, File, HTTPException
from ..services.image_processing import (
    compute_image_hash,
    load_image_from_bytes,
    analyze_image_colors,
    estimate_affected_area,
)
from ..services.inference import deterministic_inference, get_recommendations
from ..services.model import is_demo_mode, get_model_info
from ..schemas.prediction import PredictionResponse, AnalysisResponse

router = APIRouter()


@router.post("/predict", response_model=PredictionResponse)
async def predict_disease(file: UploadFile = File(...)):
    """
    Predict crop disease from an uploaded image.
    Returns disease name, confidence, severity, and risk level.
    """
    # Validate file type
    if file.content_type not in ["image/jpeg", "image/png", "image/webp", "image/jpg"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Please upload a JPG, PNG, or WebP image."
        )
    
    # Read image bytes
    try:
        image_bytes = await file.read()
    except Exception:
        raise HTTPException(status_code=400, detail="Failed to read uploaded file.")
    
    # Check file size (max 10MB)
    if len(image_bytes) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Maximum size is 10MB.")
    
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")
    
    # Compute hash for deterministic results
    image_hash = compute_image_hash(image_bytes)
    
    # Analyze image with OpenCV
    try:
        img = load_image_from_bytes(image_bytes)
        image_stats = analyze_image_colors(img)
    except Exception:
        image_stats = {}
    
    # Run inference
    result = deterministic_inference(image_hash, image_stats)
    
    return PredictionResponse(
        disease=result["disease"],
        confidence=result["confidence"],
        severity=result["severity"],
        affected_area=result.get("affected_area", result["severity"] * 0.85),
        risk_level=result["risk_level"],
        explanation=result["explanation"],
        is_demo=result.get("is_demo", True),
    )


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_crop(file: UploadFile = File(...)):
    """
    Comprehensive crop analysis: disease prediction + image analysis + recommendations.
    """
    # Validate file type
    if file.content_type not in ["image/jpeg", "image/png", "image/webp", "image/jpg"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Please upload a JPG, PNG, or WebP image."
        )
    
    # Read image bytes
    try:
        image_bytes = await file.read()
    except Exception:
        raise HTTPException(status_code=400, detail="Failed to read uploaded file.")
    
    if len(image_bytes) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Maximum size is 10MB.")
    
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")
    
    # Compute hash for deterministic results
    image_hash = compute_image_hash(image_bytes)
    
    # Analyze image with OpenCV
    try:
        img = load_image_from_bytes(image_bytes)
        image_stats = analyze_image_colors(img)
        affected_pct, area_stats = estimate_affected_area(img)
        image_stats["affected_area_analysis"] = area_stats
    except Exception:
        image_stats = {}
        affected_pct = None
    
    # Run inference
    result = deterministic_inference(image_hash, image_stats)
    
    # Use OpenCV-estimated affected area if available
    if affected_pct is not None and not is_demo_mode():
        result["affected_area"] = affected_pct
    
    # Get recommendations
    recommendations = get_recommendations(result["disease"])
    
    prediction = PredictionResponse(
        disease=result["disease"],
        confidence=result["confidence"],
        severity=result["severity"],
        affected_area=result.get("affected_area", result["severity"] * 0.85),
        risk_level=result["risk_level"],
        explanation=result["explanation"],
        is_demo=result.get("is_demo", True),
    )
    
    return AnalysisResponse(
        prediction=prediction,
        image_stats=image_stats,
        recommendations=recommendations,
    )

from ..services.rice_model import predict_rice_image

@router.post('/predict/rice')
async def predict_rice_route(file: UploadFile = File(...)):
    if file.content_type not in ['image/jpeg', 'image/png', 'image/webp', 'image/jpg']:
        raise HTTPException(status_code=400, detail='Invalid file type. Please upload a JPG, PNG, or WebP image.')
    try:
        image_bytes = await file.read()
    except Exception:
        raise HTTPException(status_code=400, detail='Failed to read uploaded file.')
    if len(image_bytes) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail='File too large.')
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail='Empty file uploaded.')
    try:
        result = predict_rice_image(image_bytes)
        return {'success': True, 'prediction': result['disease'], 'confidence': result['confidence']}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

