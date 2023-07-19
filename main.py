
import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model
from fastapi import FastAPI, WebSocket, Request
from fastapi.templating import Jinja2Templates

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Load the pre-trained emotion detection model
model_path = "model.h5"
emotion_model = load_model(model_path, compile=False)

# Define the emotions that the model can detect
EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

# Function to preprocess the image for the emotion model


def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (48, 48))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Function to detect emotions from the video frame


def detect_emotion(frame):
    # Preprocess the frame
    processed_frame = preprocess_image(frame)

    # Predict the emotion
    prediction = emotion_model.predict(processed_frame)
    emotion_label = EMOTIONS[np.argmax(prediction[0])]

    return emotion_label

# WebSocket endpoint to handle video stream and emotion detection


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connection established.")

    # Open the video capture
    video_capture = cv2.VideoCapture(0)

    while True:
        try:
            # Read a frame from the video capture
            ret, frame = video_capture.read()
            if not ret:
                break

            # Detect emotion from the frame
            emotion = detect_emotion(frame)

            # Send the detected emotion to the frontend via WebSocket
            await websocket.send_text('{"emotion": "%s"}' % emotion)
        except Exception as e:
            print("Error:", e)

    video_capture.release()
    await websocket.close()
    print("WebSocket connection closed.")

# Route to serve the index.html page


@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
