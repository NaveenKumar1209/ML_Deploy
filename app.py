# app.py

import uvicorn
from fastapi import FastAPI, Form, File, UploadFile, HTTPException
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
from fastapi.responses import JSONResponse
from starlette.requests import Request
import io
from PIL import Image

app = FastAPI()

# Load the pre-trained model
model = tf.keras.models.load_model("Digits_mnist")

class Item(BaseModel):
    image: UploadFile

@app.post("/predict")
async def predict(request: Request, image: UploadFile = File(...)):
    try:
        # Read and preprocess the image
        contents = await image.read()
        image = Image.open(io.BytesIO(contents)).convert("L")  # Convert to grayscale
        image = image.resize((28, 28))
        image_data = np.array(image).reshape(1, 28, 28, 1).astype('float32') / 255.0

        # Make predictions using the loaded model
        predictions = model.predict(image_data)
        predicted_class = np.argmax(predictions)

        return JSONResponse(content={"predicted_class": int(predicted_class)}, status_code=200)

    except Exception as e:
        raise HTTPException(status_code=500, detail="Error processing the image")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)