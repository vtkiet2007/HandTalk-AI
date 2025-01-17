import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp

# Khởi tạo Mediapipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Hàm xử lý ảnh với Mediapipe
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

# Hàm vẽ các điểm đặc trưng trên ảnh
def draw_styled_landmarks(image, results):
    # Vẽ các điểm trên khuôn mặt
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                              mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                              mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1))
    # Vẽ các điểm trên cơ thể
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2))
    # Vẽ các điểm trên tay trái
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=1, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=1, circle_radius=2))
    # Vẽ các điểm trên tay phải
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=1, circle_radius=1),
                              mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=1, circle_radius=1))

# Hàm trích xuất keypoints từ kết quả Mediapipe
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
    return np.concatenate([pose, face, lh, rh])

# Đường dẫn lưu dữ liệu
DATA_PATH = os.path.join('MP_Data')

# Tên hành động khi lưu file (không dấu, không khoảng trắng)
actions = np.array(['Xin_Chao', 'Toi_yeu_ban', 'Do_an'])

# Số lượng video cho mỗi hành động
no_sequences = 30

# Số lượng khung hình cho mỗi video
sequence_length = 30

# Tạo thư mục cho từng hành động và từng video
for action in actions:
    for sequence in range(no_sequences):
        try:
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass

# Bật webcam để thu thập dữ liệu
cap = cv2.VideoCapture(0)

# Khởi tạo mô hình Mediapipe
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    for idx, action in enumerate(actions):  # Duyệt qua danh sách actions không dấu
        for sequence in range(no_sequences):
            for frame_num in range(sequence_length):

                # Đọc dữ liệu từ webcam
                ret, frame = cap.read()

                # Xử lý bằng Mediapipe
                image, results = mediapipe_detection(frame, holistic)
                draw_styled_landmarks(image, results)

                # Hiển thị thông báo thu thập dữ liệu với tên hành động có dấu
                if frame_num == 0:
                    cv2.putText(image, f'Bắt đầu thu thập: {display_actions[idx]} | Video số {sequence}',
                                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.waitKey(2000)
                else:
                    cv2.putText(image, f'Thu thập: {display_actions[idx]} | Video số {sequence}',
                                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                # Trích xuất keypoints và lưu vào thư mục (sử dụng tên không dấu)
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)

                # Hiển thị khung hình
                cv2.imshow('OpenCV feed', image)

                # Nhấn 'q' để thoát
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

cap.release()
cv2.destroyAllWindows()
