import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical

# Đường dẫn dữ liệu
DATA_PATH = 'MP_Data'
actions = np.array(['Xin_Chao', 'Toi_yeu_ban', 'Do_an'])
no_sequences = 30
sequence_length = 30

# Khởi tạo danh sách dữ liệu và nhãn
X = []
y = []

# Duyệt qua từng hành động và đọc dữ liệu từ file .npy
for idx, action in enumerate(actions):
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            npy_path = os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy")
            keypoints = np.load(npy_path)
            window.append(keypoints)
        X.append(window)
        y.append(idx)

# Chuyển đổi dữ liệu thành numpy array
X = np.array(X)
y = to_categorical(y).astype(int)

# Khởi tạo mô hình LSTM
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(len(actions), activation='softmax'))

# Biên dịch mô hình
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# Lưu trọng số tốt nhất trong quá trình huấn luyện
checkpoint = ModelCheckpoint('action.h5', save_best_only=True, monitor='categorical_accuracy', mode='max')

# Huấn luyện mô hình
model.fit(X, y, epochs=200, callbacks=[checkpoint])
