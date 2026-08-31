"""
Model loading abstraction layer.
In production, this would load a trained PyTorch model.
For the prototype, it provides a clean interface that falls back to demo inference.
"""
import os
from typing import Optional, Dict

# Flag to track if a real model is available
_model = None
_model_loaded = False
_demo_mode = True


def load_model(model_path: Optional[str] = None) -> bool:
    """
    Attempt to load a trained PyTorch model.
    Falls back to demo mode if no model is available.
    
    Args:
        model_path: Path to the trained model file (.pth)
    
    Returns:
        True if a real model was loaded, False if using demo mode
    """
    global _model, _model_loaded, _demo_mode
    
    if model_path is None:
        model_path = os.environ.get("MODEL_PATH", "models/crop_disease_model.pth")
    
    if os.path.exists(model_path):
        try:
            import torch
            _model = torch.load(model_path, map_location="cpu")
            _model.eval()
            _model_loaded = True
            _demo_mode = False
            print(f"[MODEL] Loaded trained model from {model_path}")
            return True
        except Exception as e:
            print(f"[MODEL] Failed to load model: {e}. Using demo mode.")
            _model_loaded = False
            _demo_mode = True
            return False
    else:
        print(f"[MODEL] No model found at {model_path}. Using demo inference.")
        _model_loaded = False
        _demo_mode = True
        return False


def get_model():
    """Get the loaded model, or None if in demo mode."""
    return _model


def is_demo_mode() -> bool:
    """Check if the service is running in demo mode."""
    return _demo_mode


def is_model_loaded() -> bool:
    """Check if a real model is loaded."""
    return _model_loaded


def get_model_info() -> Dict:
    """Get model information for health check."""
    return {
        "model_loaded": _model_loaded,
        "demo_mode": _demo_mode,
        "model_type": "PyTorch CNN" if _model_loaded else "Deterministic Demo",
    }
