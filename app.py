from flask import Flask, request, jsonify
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from PIL import Image

app = Flask(__name__)

# Khởi tạo Mediapipe và mô hình đã huấn luyện
mp_holistic = mp.solutions.holistic
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(3, activation='softmax'))

# Load trọng số mô hình đã huấn luyện
model.load_weights('action.h5')

# Danh sách các hành động
actions = np.array(['Xin Chao', 'Toi yeu ban', 'Hai Long'])
sequence = []  # Chuỗi lưu trữ keypoints để dự đoán
threshold = 0.5  # Ngưỡng dự đoán

def extract_keypoints(results):
    """Hàm trích xuất keypoints từ kết quả Mediapipe"""
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
    return np.concatenate([pose, face, lh, rh])

@app.route('/predict', methods=['POST'])
def predict():
    """API nhận khung hình từ web, xử lý và trả về kết quả dự đoán"""
    try:
        file = request.files['file']
        img = np.array(Image.open(file.stream).convert('RGB'))

        # Xử lý ảnh với Mediapipe
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = holistic.process(img_rgb)
            keypoints = extract_keypoints(results)

            sequence.insert(0, keypoints)
            sequence = sequence[:30]

            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                if res[np.argmax(res)] > threshold:
                    predicted_action = actions[np.argmax(res)]
                    return jsonify({'prediction': predicted_action})

        return jsonify({'prediction': 'Không phát hiện hành động'})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='192.168.153.69', port=5000, debug=True)
