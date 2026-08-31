import os
import io
import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image

_model = None
_idx_to_label = {}
_model_loaded = False
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_rice_model():
    """Load the best_rice_model checkpoint."""
    global _model, _idx_to_label, _model_loaded
    
    # Use path relative to the current working directory, or a default based on backend root
    model_path = os.path.join(os.getcwd(), "models", "best_rice_model.pth")
    if not os.path.exists(model_path):
        # Fallback if running from within app directory
        model_path = os.path.join(os.getcwd(), "..", "models", "best_rice_model.pth")
    
    if not os.path.exists(model_path):
        print(f"[RICE MODEL] Not found at {model_path}")
        return False
        
    try:
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=_device, weights_only=False)
        
        # We know it's EfficientNet B0 based on the architecture match
        _model = models.efficientnet_b0(num_classes=10)
        _model.load_state_dict(checkpoint["model_state_dict"])
        _model.to(_device)
        _model.eval()
        
        _idx_to_label = checkpoint.get("idx_to_label", {})
        _model_loaded = True
        print(f"[RICE MODEL] Successfully loaded from {model_path} onto {_device}")
        return True
    except Exception as e:
        print(f"[RICE MODEL] Error loading model: {e}")
        return False

def get_rice_model_info():
    return {
        "rice_model_loaded": _model_loaded,
        "device": str(_device),
        "classes": len(_idx_to_label) if _model_loaded else 0
    }

def predict_rice_image(image_bytes: bytes):
    """
    Run real PyTorch inference on the uploaded image bytes.
    """
    if not _model_loaded:
        raise RuntimeError("Model is not loaded. Cannot run inference.")
        
    # Standard ImageNet preprocessing expected by EfficientNet
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    try:
        # Open image
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        # Preprocess
        input_tensor = preprocess(image)
        # Add batch dimension
        input_batch = input_tensor.unsqueeze(0).to(_device)
        
        with torch.no_grad():
            output = _model(input_batch)
            
        # Get probabilities
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        
        # Get top prediction
        confidence, predicted_idx = torch.max(probabilities, 0)
        confidence = float(confidence.item())
        predicted_idx = predicted_idx.item()
        
        disease_name = _idx_to_label.get(predicted_idx, _idx_to_label.get(str(predicted_idx), "Unknown"))
        
        return {
            "disease": disease_name,
            "confidence": confidence
        }
    except Exception as e:
        raise RuntimeError(f"Inference failed: {e}")
