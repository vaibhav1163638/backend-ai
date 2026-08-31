"""
Image processing service using OpenCV.
Analyzes crop images for color distribution, texture, and estimated affected area.
"""
import hashlib
import io
from typing import Dict, Tuple

import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

from PIL import Image


def compute_image_hash(image_bytes: bytes) -> str:
    """Compute MD5 hash of image bytes for deterministic demo results."""
    return hashlib.md5(image_bytes).hexdigest()


def load_image_from_bytes(image_bytes: bytes) -> np.ndarray:
    """Load image from bytes into OpenCV format (BGR)."""
    if CV2_AVAILABLE:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image")
        return img
    else:
        # Fallback using PIL
        pil_img = Image.open(io.BytesIO(image_bytes))
        pil_img = pil_img.convert("RGB")
        return np.array(pil_img)[:, :, ::-1]  # RGB to BGR


def analyze_image_colors(img: np.ndarray) -> Dict:
    """Analyze color distribution of the crop image."""
    if not CV2_AVAILABLE:
        return {
            "mean_green": 120.0,
            "mean_brown": 80.0,
            "green_ratio": 0.55,
            "brown_ratio": 0.20,
            "yellow_ratio": 0.15,
        }

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Green detection (healthy plant tissue)
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    green_ratio = float(np.count_nonzero(green_mask)) / green_mask.size

    # Brown/yellow detection (diseased tissue)
    lower_brown = np.array([10, 50, 50])
    upper_brown = np.array([25, 255, 200])
    brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)
    brown_ratio = float(np.count_nonzero(brown_mask)) / brown_mask.size

    # Yellow detection
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([35, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    yellow_ratio = float(np.count_nonzero(yellow_mask)) / yellow_mask.size

    # Mean color values
    mean_bgr = img.mean(axis=(0, 1))

    return {
        "mean_blue": round(float(mean_bgr[0]), 2),
        "mean_green": round(float(mean_bgr[1]), 2),
        "mean_red": round(float(mean_bgr[2]), 2),
        "green_ratio": round(green_ratio, 4),
        "brown_ratio": round(brown_ratio, 4),
        "yellow_ratio": round(yellow_ratio, 4),
        "image_width": img.shape[1],
        "image_height": img.shape[0],
    }


def estimate_affected_area(img: np.ndarray) -> Tuple[float, Dict]:
    """
    Estimate the percentage of affected (diseased) area in the image.
    Uses color segmentation to differentiate healthy vs diseased tissue.
    """
    if not CV2_AVAILABLE:
        return 25.0, {"method": "fallback", "healthy_pixels": 0, "diseased_pixels": 0}

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Healthy green tissue mask
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    # Diseased tissue mask (brown, dark spots, yellow)
    lower_brown = np.array([5, 30, 30])
    upper_brown = np.array([25, 255, 200])
    brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)

    # Dark spots (necrotic tissue)
    lower_dark = np.array([0, 0, 0])
    upper_dark = np.array([180, 255, 60])
    dark_mask = cv2.inRange(hsv, lower_dark, upper_dark)

    # Combine disease indicators
    diseased_mask = cv2.bitwise_or(brown_mask, dark_mask)

    # Calculate plant area (green + diseased, excluding background)
    plant_mask = cv2.bitwise_or(green_mask, diseased_mask)
    plant_pixels = max(np.count_nonzero(plant_mask), 1)
    diseased_pixels = np.count_nonzero(diseased_mask)

    affected_pct = min(100.0, round((diseased_pixels / plant_pixels) * 100, 1))

    stats = {
        "method": "opencv_color_segmentation",
        "total_pixels": int(img.shape[0] * img.shape[1]),
        "plant_pixels": int(plant_pixels),
        "healthy_pixels": int(np.count_nonzero(green_mask)),
        "diseased_pixels": int(diseased_pixels),
    }

    return affected_pct, stats
