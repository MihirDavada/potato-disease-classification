from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf


app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = tf.keras.models.load_model("../saved_models/potatoes.keras")

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def read_file_as_image(file_bytes) -> np.ndarray:
    image = np.array(Image.open(BytesIO(file_bytes))) 
    return image

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    # Read The File In Bytes.
    file_bytes = await file.read()
    # Convert Bytes Into Numpy Array.
    image_arr = read_file_as_image(file_bytes)
    # .predict() method does not accept single image. It accepts the batch of image.
    # we have to pass [[256 , 256 , 3]] instead of [256 , 256 , 3].
    # .expand_dims() adds extra dimension into array.
    img_batch = np.expand_dims(image_arr, 0)
    predictions = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
