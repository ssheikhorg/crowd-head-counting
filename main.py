from fastapi import FastAPI, UploadFile, File, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from ultralytics import YOLO
import os
import uuid
from pathlib import Path
import torch
from PIL import Image
import io
import time

app = FastAPI()

# Setup static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Create directories if they don't exist
Path("static/results").mkdir(parents=True, exist_ok=True)

# Load your trained YOLO model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO('runs_optimized/yolov8m_crowd/weights/best.pt').to(device)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    start_time = time.time()
    try:
        # Read and validate image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # Run inference
        results = model(image)

        # Save result image
        result_id = uuid.uuid4()
        result_image_path = f"static/results/{result_id}.jpg"
        results[0].save(filename=result_image_path)

        # Calculate metrics
        count = len(results[0].boxes)
        confidences = [box.conf.item() for box in results[0].boxes]
        avg_confidence = round((sum(confidences) / max(1, len(confidences))) * 100, 1)
        processing_time = round(time.time() - start_time, 2)

        # Determine density
        if count < 10:
            density = "Low"
        elif count < 50:
            density = "Medium"
        else:
            density = "High"

        return {
            "success": True,
            "count": count,
            "image_url": f"/{result_image_path}",
            "density": density,
            "avg_confidence": avg_confidence,
            "processing_time": processing_time,
            "error": None
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@app.get("/results")
async def show_results(
        request: Request,
        count: int,
        image: str,
        density: str,
        confidence: float,
        processing_time: float
):
    return templates.TemplateResponse("results.html", {
        "request": request,
        "count": count,
        "image_url": image,
        "density": density,
        "avg_confidence": confidence,
        "processing_time": processing_time
    })


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=9000)
