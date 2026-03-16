"""
Unified API Gateway
FastAPI entry point for multimodal AI inferences.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
import shutil

# Local imports
from src.nlp.sentiment_engine import SentimentEngine
from src.cv.vision_classifier import VisionClassifier
from src.ml.predictive_pipeline import PredictivePipeline
from src.utils.logger import StructuredLogger

app = FastAPI(
    title="Aether Advanced Intelligence Suite API",
    description="A unified API for NLP, Computer Vision, and Predictive ML.",
    version="1.0.0"
)

logger = StructuredLogger("api-gateway")

# Initialize models (Lazy loading for better startup)
nlp_engine = SentimentEngine()
cv_classifier = VisionClassifier()

class SentimentRequest(BaseModel):
    text: str

class MLRequest(BaseModel):
    data: List[Dict[str, Any]]

@app.get("/")
def read_root():
    return {"status": "online", "message": "Welcome to Aether AI Suite Gateway."}

@app.post("/nlp/sentiment")
async def analyze_sentiment(request: SentimentRequest):
    try:
        logger.info(f"Analyzing sentiment for text: {request.text[:50]}...")
        result = nlp_engine.analyze(request.text)
        return {"sentiment": result}
    except Exception as e:
        logger.error(f"NLP Inference Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/cv/classify")
async def classify_image(file: UploadFile = File(...)):
    try:
        # Create temp file for processing
        temp_file = f"temp_{file.filename}"
        with open(temp_file, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        logger.info(f"Classifying image: {file.filename}")
        results = cv_classifier.classify(temp_file)
        
        # Cleanup
        os.remove(temp_file)
        return {"predictions": results}
    except Exception as e:
        logger.error(f"CV Inference Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ml/predict")
async def predict_ml(request: MLRequest):
    # This would require a pre-trained model loading mechanism
    return {"message": "Predictive pipeline inference endpoint active.", "input_count": len(request.data)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
