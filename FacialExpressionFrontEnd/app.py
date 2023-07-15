from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)
model = load_model('model.h5')

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


def preprocess_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return None, None
    (x, y, w, h) = faces[0]
    face_crop = gray[y:y + w, x:x + h]
    face_crop = cv2.resize(face_crop, (48, 48))
    face_crop = face_crop.astype("float32") / 255.0
    face_crop = np.reshape(face_crop, (1, 48, 48, 1))
    return face_crop, (x, y, w, h)


def get_expression_label(image):
    face_crop, face_coords = preprocess_face(image)
    if face_crop is not None:
        expression_scores = model.predict(face_crop)
        max_index = np.argmax(expression_scores[0])
        label = emotions[max_index]
        return label, face_coords
    else:
        return None, None


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'})

    image = request.files['image'].read()
    nparr = np.frombuffer(image, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    label, face_coords = get_expression_label(frame)

    if label is not None:
        cv2.rectangle(frame, (face_coords[0], face_coords[1]), (
            face_coords[0] + face_coords[2], face_coords[1] + face_coords[3]), (0, 255, 0), 2)
        cv2.putText(frame, label, (face_coords[0], face_coords[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    ret, jpeg = cv2.imencode('.jpg', frame)
    response = jpeg.tobytes()

    return response


if __name__ == '__main__':
    app.run(debug=True)
