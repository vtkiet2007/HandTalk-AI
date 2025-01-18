import cv2
import numpy as np
import mediapipe as mp

# Địa chỉ IP của ESP32-CAM
esp32_cam_url = 'http://192.168.153.107/'

# Thiết lập Mediapipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_styled_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                              mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                              mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1))
    
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2))
    
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=1, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=1, circle_radius=2))
    
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=1, circle_radius=1),
                              mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=1, circle_radius=1))

# Kết nối tới video stream từ ESP32-CAM
cap = cv2.VideoCapture(esp32_cam_url)

# Sử dụng Mediapipe Holistic để xử lý video
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Không thể nhận khung hình. Kiểm tra kết nối ESP32-CAM")
            break

        # Xử lý khung hình bằng Mediapipe
        image, results = mediapipe_detection(frame, holistic)
        
        # Vẽ landmarks lên khung hình
        draw_styled_landmarks(image, results)
        
        # Hiển thị khung hình
        cv2.imshow('ESP32-CAM feed', image)
        
        # Thoát khi nhấn phím 'q'
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()