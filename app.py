import os
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import uvicorn

from src.cnnClassifier.pipeline.prediction import PredictionPipeline

# Load environment variables
load_dotenv()


app = FastAPI(title="Chest Cancer Classification API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


@app.get("/", response_class=HTMLResponse)
async def home():
    return Path("templates/index.html").read_text(encoding="utf-8")


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "chest-cancer-classifier"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    file_path = UPLOAD_DIR / file.filename
    
    try:
        with open(file_path, "wb") as f:
            f.write(await file.read())
        
        pipeline = PredictionPipeline(filename=str(file_path))
        result = pipeline.predict()
        
        return {
            "success": True,
            "prediction": result[0]["image"],
            "filename": file.filename
        }
    finally:
        if file_path.exists():
            os.remove(file_path)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
