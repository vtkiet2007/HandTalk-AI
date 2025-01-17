from flask import Flask, request, jsonify, Response
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import io
import requests

app = Flask(__name__)

model = load_model('action.h5')
actions = ['Xin Chao', 'Toi yeu ban', 'Hai Long']

ESP32_CAM_IP = "192.168.153.155"

@app.route('/')
def index():
    return f'<h1>ESP32-CAM Stream</h1><img src="/stream" width="640">'

@app.route('/stream')
def stream():
    def generate():
        url = f"http://{ESP32_CAM_IP}/stream"
        response = requests.get(url, stream=True)
        for chunk in response.iter_content(chunk_size=1024):
            yield chunk
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/predict', methods=['POST'])
def predict():
    image_file = request.data
    img = Image.open(io.BytesIO(image_file)).resize((224, 224))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

    prediction = model.predict(img_array)
    action = actions[np.argmax(prediction)]
    return jsonify({'action': action})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
