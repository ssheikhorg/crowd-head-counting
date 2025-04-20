from fastapi import FastAPI, UploadFile, File, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from ultralytics import YOLO
import uuid
from pathlib import Path
import torch
from PIL import Image
import io
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Setup static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Create directories if they don't exist
Path("static/results").mkdir(parents=True, exist_ok=True)

# Load YOLO model
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")
try:
    model = YOLO("runs_optimized/yolov8m_crowd/weights/best.pt").to(device)
    logger.info("YOLO model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load YOLO model: {str(e)}")
    raise


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    logger.info("Serving home page")
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    start_time = time.time()
    logger.info(f"Received file: {file.filename}, type: {file.content_type}")

    try:
        # Validate file
        if not file.content_type.startswith("image/"):
            logger.error("Invalid file type")
            raise ValueError("Only image files are allowed")

        # Process image
        contents = await file.read()
        logger.info(f"File size: {len(contents)} bytes")
        try:
            image = Image.open(io.BytesIO(contents))
            logger.info("Image opened successfully")
        except Exception as e:
            logger.error(f"Failed to open image: {str(e)}")
            raise ValueError("Invalid image file")

        # Run inference
        logger.info("Running YOLO inference")
        try:
            results = model(image)
            logger.info(f"YOLO results: {len(results)} objects detected")
        except Exception as e:
            logger.error(f"YOLO inference failed: {str(e)}")
            raise ValueError("Model inference failed")

        # Validate results
        if not results or len(results) == 0:
            logger.warning("No objects detected in image")
            raise ValueError("No objects detected in image")

        # Save result
        result_id = uuid.uuid4()
        result_dir = "static/results"
        result_image_path = f"{result_dir}/{result_id}.jpg"
        try:
            results[0].save(filename=result_image_path)
            logger.info(f"Result saved to: {result_image_path}")
        except Exception as e:
            logger.error(f"Failed to save result image: {str(e)}")
            raise ValueError("Failed to save result image")

        # Calculate metrics
        boxes = results[0].boxes
        count = len(boxes)
        confidences = [box.conf.item() for box in boxes] if boxes else []
        avg_confidence = (
            round((sum(confidences) / max(1, len(confidences))) * 100, 1)
            if confidences
            else 0
        )
        processing_time = round(time.time() - start_time, 2)

        # Determine density
        if count < 10:
            density = "Low"
        elif count < 50:
            density = "Medium"
        else:
            density = "High"

        logger.info(
            f"Returning results: count={count}, density={density}, confidence={avg_confidence}, processing_time={processing_time}"
        )

        return {
            "success": True,
            "count": count,
            "image_url": f"/{result_image_path}",
            "density": density,
            "avg_confidence": avg_confidence,
            "processing_time": processing_time,
        }

    except Exception as e:
        logger.error(f"Error in /predict: {str(e)}")
        return {"success": False, "error": str(e)}


@app.get("/results")
async def show_results(
    request: Request,
    count: int,
    image: str,
    density: str,
    confidence: float,
    processing_time: float,
):
    logger.info(
        f"Results page requested: count={count}, image={image}, density={density}, confidence={confidence}, processing_time={processing_time}"
    )
    return templates.TemplateResponse(
        "results.html",
        {
            "request": request,
            "count": count,
            "image_url": image,
            "density": density,
            "avg_confidence": confidence,
            "processing_time": processing_time,
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
