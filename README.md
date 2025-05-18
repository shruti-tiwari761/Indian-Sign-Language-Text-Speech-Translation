## Indian Sign Language Translator (Text/Speech) using YOLOv11 & GTTS

This Django-based web application translates Indian Sign Language (ISL) into text and speech. It uses **YOLOv11** for real-time gesture recognition and **GTTS (Google Text-to-Speech)** for generating speech output.

---

## 🧠 Features

- Real-time Indian Sign Language detection using YOLOv11
- Translates gestures into textual output
- Converts recognized text into speech using GTTS
- Web-based interface built with Django
- Easy to use and extend

---

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone https://https://github.com/shruti-tiwari761/Indian-Sign-Language-Text-Speech-Translation
cd isl-translator
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate
# On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📚 Dependencies

Ensure these libraries are included in `requirements.txt`:

```text
Django>=3.2
opencv-python
torch
torchvision
numpy
gtts
Pillow
```

> **Note**: Make sure you have a working GPU setup for YOLOv11 (PyTorch backend). If not, fallback to CPU by modifying the inference script.

---

## 🚀 Running the App

```bash
python manage.py migrate
python manage.py runserver
```

Visit `http://127.0.0.1:8000/` in your browser to access the application.

---

## 🧠 How It Works

1. The webcam captures live video input.
2. YOLOv11 processes frames to detect and classify hand gestures.
3. The detected gesture is mapped to corresponding text.
4. GTTS converts the text to speech and plays the audio.

---

## 🛠️ How to Use

1. Start the server and open the web interface.
2. Use your webcam to show an ISL gesture.
3. The application detects and displays the text.
4. The speech output will automatically play using GTTS.

---

## 🔮 Future Improvements

- Support for dynamic gestures (sentence formation)
- Add multilingual support for audio output
- Deploy on cloud with GPU acceleration

---

## 🙏 Acknowledgements

- [YOLOv11](https://github.com/ultralytics/yolov11) — for real-time gesture detection.
- [GTTS (Google Text-to-Speech)](https://pypi.org/project/gTTS/) — for converting recognized text to speech.
- [Django](https://www.djangoproject.com/) — for the backend web framework.
- [Indian Sign Language Dataset](https://universe.roboflow.com/niladri-basu-roy-qnrm4/indian-sign-language-detection/dataset/2) — used for training and testing sign recognition models.

