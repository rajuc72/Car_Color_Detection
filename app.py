from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from keras.models import load_model
from PIL import Image
import numpy as np
import io

app = FastAPI()
model = load_model('best_model.keras')
labels = ['Black', 'Blue', 'Green', 'Grey', 'Orange', 'Red', 'Silver', 'White', 'Yellow']

def preprocess_image(image_bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        return np.expand_dims(img_array, axis=0)
    except Exception as e:
        raise ValueError(f"Failed to preprocess image: {e}")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    input_tensor = preprocess_image(image_bytes)
    prediction = model.predict(input_tensor)
    predicted_label = labels[np.argmax(prediction)]
    return {"color": predicted_label}

@app.get("/", response_class=HTMLResponse)
def home():
    with open("home.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)