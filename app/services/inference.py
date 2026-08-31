"""
Deterministic demo inference engine.
Provides consistent disease predictions based on image characteristics.
When a trained model is unavailable, this provides reliable demo results.
"""
from typing import Dict

# Demo disease database with realistic crop disease information
DEMO_DISEASES = [
    {
        "disease": "Tomato Early Blight",
        "confidence": 0.94,
        "severity": 32.0,
        "affected_area": 28.0,
        "risk_level": "MODERATE",
        "explanation": "The image shows concentric ring patterns on lower leaves consistent with Alternaria solani (Early Blight). Brown spots with target-like rings are characteristic of this fungal disease, typically appearing first on older foliage.",
    },
    {
        "disease": "Tomato Late Blight",
        "confidence": 0.91,
        "severity": 55.0,
        "affected_area": 48.0,
        "risk_level": "HIGH",
        "explanation": "The image shows large, irregular water-soaked lesions with pale green to brown coloring, consistent with Phytophthora infestans (Late Blight). This disease progresses rapidly in cool, wet conditions.",
    },
    {
        "disease": "Tomato Leaf Spot",
        "confidence": 0.88,
        "severity": 22.0,
        "affected_area": 18.0,
        "risk_level": "MODERATE",
        "explanation": "Small circular spots with dark margins and lighter centers are visible, consistent with Septoria Leaf Spot. This fungal disease typically starts on lower leaves and progresses upward.",
    },
    {
        "disease": "Powdery Mildew",
        "confidence": 0.92,
        "severity": 40.0,
        "affected_area": 35.0,
        "risk_level": "MODERATE",
        "explanation": "White powdery patches visible on leaf surfaces are characteristic of powdery mildew. This fungal disease thrives in warm, dry conditions with high humidity.",
    },
    {
        "disease": "Tomato Bacterial Spot",
        "confidence": 0.87,
        "severity": 45.0,
        "affected_area": 38.0,
        "risk_level": "HIGH",
        "explanation": "Small, dark, water-soaked spots on leaves and fruit are consistent with bacterial spot caused by Xanthomonas species. The lesions may have a yellow halo.",
    },
    {
        "disease": "Healthy",
        "confidence": 0.96,
        "severity": 0.0,
        "affected_area": 0.0,
        "risk_level": "LOW",
        "explanation": "The plant appears healthy with vibrant green foliage, no visible signs of disease, pest damage, or nutrient deficiency. Continue monitoring regularly.",
    },
    {
        "disease": "Tomato Yellow Leaf Curl",
        "confidence": 0.89,
        "severity": 60.0,
        "affected_area": 52.0,
        "risk_level": "HIGH",
        "explanation": "Upward curling of leaves with yellowing margins is consistent with Tomato Yellow Leaf Curl Virus (TYLCV). This viral disease is transmitted by whiteflies.",
    },
    {
        "disease": "Tomato Mosaic Virus",
        "confidence": 0.85,
        "severity": 35.0,
        "affected_area": 30.0,
        "risk_level": "MODERATE",
        "explanation": "Mottled light and dark green patterns on leaves are consistent with Tomato Mosaic Virus (ToMV). Leaves may appear distorted with areas of lighter coloring.",
    },
]


def get_severity_level(severity: float) -> str:
    """Map severity percentage to level string."""
    if severity <= 20:
        return "LOW"
    elif severity <= 50:
        return "MODERATE"
    elif severity <= 75:
        return "HIGH"
    else:
        return "CRITICAL"


def deterministic_inference(image_hash: str, image_stats: Dict = None) -> Dict:
    """
    Provide deterministic demo inference based on image hash.
    Same image hash always produces the same result.
    
    Args:
        image_hash: MD5 hash of the image bytes
        image_stats: Optional OpenCV analysis stats to refine results
    
    Returns:
        Prediction result dict
    """
    # Use the hash to deterministically select a disease
    hash_int = int(image_hash[:8], 16)
    disease_index = hash_int % len(DEMO_DISEASES)
    
    result = DEMO_DISEASES[disease_index].copy()
    
    # If we have image stats from OpenCV, slightly adjust severity
    # based on actual color analysis (but keep it deterministic)
    if image_stats and "brown_ratio" in image_stats:
        brown_ratio = image_stats.get("brown_ratio", 0)
        green_ratio = image_stats.get("green_ratio", 0)
        
        # If the image is mostly green, lean toward healthy
        if green_ratio > 0.6 and brown_ratio < 0.05:
            result = DEMO_DISEASES[5].copy()  # Healthy
        elif brown_ratio > 0.3:
            # High brown ratio suggests more severe disease
            result["severity"] = min(result["severity"] + 10, 90.0)
            result["risk_level"] = get_severity_level(result["severity"])
    
    result["is_demo"] = True
    return result


# Treatment recommendations mapped to diseases
TREATMENT_RECOMMENDATIONS = {
    "Tomato Early Blight": [
        "Remove and destroy affected lower leaves immediately",
        "Apply copper-based organic fungicide as a preventive measure",
        "Ensure proper spacing between plants for air circulation",
        "Avoid overhead watering — use drip irrigation instead",
        "Mulch around plants to prevent soil splash onto leaves",
        "For chemical treatment, consult local agricultural extension and follow product label instructions",
    ],
    "Tomato Late Blight": [
        "Remove and destroy all affected plant parts immediately — do NOT compost",
        "Apply copper-based fungicide following product label instructions",
        "Improve drainage and air circulation around plants",
        "Avoid watering late in the day",
        "Monitor neighboring plants closely",
        "Consider resistant varieties for future planting",
        "Consult local agricultural extension for approved treatment options",
    ],
    "Tomato Leaf Spot": [
        "Remove affected lower leaves and destroy them",
        "Apply neem oil spray as an organic treatment",
        "Avoid working with plants when foliage is wet",
        "Improve air circulation through proper spacing",
        "Use mulch to prevent soil-borne spore splash",
        "Follow crop rotation practices",
    ],
    "Powdery Mildew": [
        "Apply sulfur-based or potassium bicarbonate spray",
        "Neem oil can be effective as an organic treatment",
        "Improve air circulation — prune dense foliage",
        "Avoid excessive nitrogen fertilization",
        "Water at the base of plants, not on foliage",
        "Consult local guidelines for approved fungicide options",
    ],
    "Tomato Bacterial Spot": [
        "Remove and destroy affected plant parts",
        "Apply copper-based bactericide — follow product label",
        "Avoid overhead irrigation",
        "Do NOT save seed from infected plants",
        "Practice crop rotation (3–4 year cycle)",
        "Use certified disease-free transplants",
    ],
    "Healthy": [
        "Continue regular monitoring every 5–7 days",
        "Maintain proper watering schedule",
        "Apply balanced fertilization as needed",
        "Watch for early signs of pest activity",
        "Ensure good air circulation",
    ],
    "Tomato Yellow Leaf Curl": [
        "Control whitefly population using yellow sticky traps",
        "Apply neem oil to deter whiteflies",
        "Remove and destroy severely affected plants",
        "Use reflective mulch to repel whiteflies",
        "Consider resistant varieties for future planting",
        "Consult local extension for insecticide recommendations",
    ],
    "Tomato Mosaic Virus": [
        "Remove and destroy infected plants — do NOT compost",
        "Disinfect tools and hands after handling infected plants",
        "Control aphid vectors with appropriate measures",
        "Use virus-free certified seed and transplants",
        "Practice crop rotation",
        "Consider resistant varieties for future planting",
    ],
}


def get_recommendations(disease: str) -> list:
    """Get treatment recommendations for a disease."""
    return TREATMENT_RECOMMENDATIONS.get(disease, [
        "Monitor the crop closely for changes",
        "Consult your local agricultural extension officer",
        "Upload follow-up images to track progress",
    ])
