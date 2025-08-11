from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from keras.models import load_model
from PIL import Image
import numpy as np
import io
import os
import asyncio
from fastapi.templating import Jinja2Templates
from fastapi import Request, HTTPException


app = FastAPI()
# Load the pre-trained model
model_path = os.path.join(os.path.dirname(__file__), 'car_colorscnn.keras')
model = load_model(model_path)
labels = ['Black', 'Blue', 'Green', 'Grey', 'Orange', 'Red', 'Silver', 'White', 'Yellow']
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))

def preprocess_image(image_bytes):
    """Convert loaded image bytes to a normalize tensor."""
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.resize((224, 224))
        img_array = np.array(img).astype('float32') / 255.0
        img_array=np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        raise ValueError(f"Failed to preprocess image: {e}")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        input_tensor = preprocess_image(image_bytes)
        if input_tensor is None:
            raise ValueError("Invalid image format or size")
        if input_tensor.shape != (1, 224, 224, 3):
            raise ValueError("Input tensor must be of shape (1, 224, 224, 3)")
        prediction = None
        if model is None:
            raise ValueError("Model is not loaded properly")
        if not labels:
            raise ValueError("Labels are not defined")
        if not isinstance(input_tensor, np.ndarray):
            raise ValueError("Input tensor must be a numpy array")
        if input_tensor.dtype != np.float32:
            raise ValueError("Input tensor must be of type float32")
        if input_tensor.ndim != 4:
            raise ValueError("Input tensor must have 4 dimensions")
        if input_tensor.shape[1:] != (224, 224, 3):
            raise ValueError("Input tensor must have shape (1, 224, 224, 3)")

        # Run prediction in a thread to avoid blocking event loop
        
        prediction = model.predict(input_tensor)
        predicted_label = labels[int(np.argmax(prediction))]
        
        return {"color": predicted_label}

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    """Serve the HTML upload page."""
    try:        
        return templates.TemplateResponse("home.html", {"request": request})
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="home.html not found")