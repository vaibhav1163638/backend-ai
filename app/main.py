"""
FastAPI main application for Crop Health AI Service.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes.predict import router as predict_router
from .services.model import load_model, get_model_info
from .services.rice_model import load_rice_model, get_rice_model_info

app = FastAPI(
    title="Crop Health AI Service",
    description="AI-powered crop disease detection and analysis service for SIH 2026",
    version="1.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    """Initialize the AI model on startup."""
    load_model()
    load_rice_model()


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    model_info = get_model_info()
    rice_model_info = get_rice_model_info()
    return {
        "status": "healthy",
        "service": "crop-health-ai",
        "version": "1.0.0",
        **model_info,
        **rice_model_info,
    }


# Include prediction routes
app.include_router(predict_router)
