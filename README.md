# HandTalk-AI

HandTalk-AI is a real-time hand gesture recognition system designed to empower communication for the deaf or mute community. By combining ESP32-CAM, artificial intelligence (AI), and a modern web application, HandTalk-AI is a practical, socially impactful project at the intersection of hardware and AI for good.

---

## 🌟 Project Highlights

- **For the Community:** Designed to assist people with hearing or speech impairments.
- **Accessible Hardware:** Uses affordable and widely available ESP32-CAM modules.
- **Cutting-Edge AI:** Employs Convolutional Neural Networks (CNN) and LSTM via TensorFlow and Mediapipe.
- **Web Interface:** Real-time gesture visualization and interaction in your browser.
- **Easy to Set Up:** Step-by-step guide from firmware upload to server and web UI.

---

## 👨‍💻 Author

- **Kiệt Võ Tuấn**  
  Researcher | Web Development, Mạng thần kinh tích chập (CNN), ESP32-CAM  
  [LinkedIn](https://www.linkedin.com/in/ki%E1%BB%87t-v%C3%B5-tu%E1%BA%A5n-ab2001346/) • [GitHub](https://github.com/vtkiet2007)

---

## ✨ Features

- Real-time hand gesture recognition with ESP32-CAM streaming
- AI-based gesture inference (Mediapipe + LSTM)
- Web-based dashboard with live results
- Modular: easily extend with new gestures, models, or interface

---

# Gesture Recognition System with ESP32-CAM – Full Setup & Test Guide

## 1. What You Need

**Hardware:**
- 1x [ESP32-CAM Module](https://docs.ai-thinker.com/en/esp32-cam)
- 1x FTDI USB-to-Serial Adapter (3.3V)
- Jumper wires, Breadboard
- USB cable

**Software:**
- Arduino IDE
- Python 3.8+
- Python libraries: Flask, Mediapipe, TensorFlow, OpenCV
- Your trained model file: `action.h5`

---

## 2. Flash ESP32-CAM Firmware (MJPEG Stream)

**Step 1: Arduino Setup**
- Open Arduino IDE
- File > Preferences, paste in "Additional Board URLs":
  ```
  https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json
  ```
- Tools > Board > Boards Manager, search and install **esp32**

**Step 2: Hardware Wiring**

| ESP32-CAM | FTDI Adapter    |
| --------- | --------------- |
| GND       | GND             |
| 5V        | 5V              |
| U0R (RX)  | TX              |
| U0T (TX)  | RX              |
| IO0       | GND (boot mode) |

- Press **RESET** or **EN** to restart ESP32 in flashing mode.

**Step 3: Upload Firmware**
- Open `esp32_cam.ino` (`demo_project/firmware`)
- Edit Wi-Fi credentials:
  ```cpp
  const char* ssid = "YourWiFi";
  const char* password = "YourPassword";
  ```
- Set Tools:
    - Board: ESP32 Wrover Module
    - Flash Frequency: 40 MHz
    - Partition Scheme: Huge APP
    - Upload Speed: 115200
- Click **Upload**
- Disconnect IO0 from GND, **press RESET**

**Step 4: Get IP Address**
- Open Serial Monitor (115200 baud)
- Note the IP address (e.g. `http://192.168.1.107`)

---

## 3. Prepare AI Server

**Step 1: Setup Python Environment**
```bash
cd demo_project/server
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Step 2: Add Model**
- Place trained gesture model (`action.h5`) in `server/`
- Ensure it's trained on 30 frames × 1662 keypoints (see `run.py`)

**Step 3: Run Optimized Server**
```bash
python optimized_server.py
```

---

## 4. Access Video and Results

**Step 1: Start Web Interface**
- Open in browser:  
  `demo_project/web/index.html`

**Step 2: Watch the System Work**
- Video from ESP32-CAM streams on screen
- Every second:
    - Extracts hand/pose/face keypoints
    - Runs AI inference
    - Displays detected gesture (e.g., “Xin chào”)

---

## 5. Common Troubleshooting

| Issue                            | Solution                                                      |
| -------------------------------- | ------------------------------------------------------------- |
| ESP32 doesn’t appear on COM port | Check USB cable, drivers, FTDI jumper wiring                  |
| MJPEG stream not showing         | Check IP, open `http://<ip>:81/stream` in browser             |
| No predictions                   | Make sure `action.h5` is compatible and in correct location   |
| Delay in video                   | Use LAN instead of Wi-Fi for testing                          |
| High latency                     | Lower frame size or switch to WebRTC (advanced)               |

---

## 6. Next Steps / Ideas

- Add more gestures to `action.h5`
- Build a mobile app to access the server
- Replace MJPEG with WebRTC for low latency
- Display 3D avatar to visualize hand signs
- Connect to Text-to-Speech for live communication aid

---
## 🙏 Special Thanks

- Special thanks to **Mr. Phương** for valuable guidance and support.
- Many thanks to [MonzerDev/Real-Time-Sign-Language-Recognition](https://github.com/MonzerDev/Real-Time-Sign-Language-Recognition) for inspiring ideas and technical references that helped shape this project.

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.

---

> “Technology is best when it brings people together.”  
> *— Matt Mullenweg*

---

**HandTalk-AI** is open for collaboration and contributions!  
Feel free to fork, star, and share to advance AI for the community.
