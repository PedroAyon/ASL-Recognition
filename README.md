# American Sign Language (ASL) Alphabet Recognition

## 📌 Project Overview
This project implements an American Sign Language (ASL) alphabet recognition system using deep learning and MediaPipe for real-time hand tracking. The model predicts ASL signs from single-hand landmark data, providing real-time feedback on recognized gestures.

## 🚀 Features
- Real-time hand landmark detection using MediaPipe.
- ASL gesture classification using a trained PyTorch model.
- Confidence-based prediction handling.
- Uses CUDA for acceleration if available.

## 📥 Dataset Download
To train the model, download the ASL datasets from Kaggle:
- **[ASL American Sign Language Alphabet Dataset](https://www.kaggle.com/datasets/debashishsau/aslamerican-sign-language-aplhabet-dataset)**
- **[ASL Alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)**

After downloading, checkout the `merge_database.py` utility to put them together.

## 🔧 Installation & Setup
### 1️⃣ Clone the Repository
```sh
git clone https://github.com/PedroAyon/ASL-Recognition.git
cd ASL-Recognition
```

### 2️⃣ Install Dependencies
```sh
pip install -r requirements.txt
```

### 3️⃣ Train the Model (Optional)
If you need to retrain the model, run:
```sh
python model_training/train_model.py
```

### 4️⃣ Run Real-Time ASL Detection
Ensure your webcam is connected, then execute:
```sh
python asl_webcam_recognition/asl_recognition.py
```

## 🎯 How It Works
1. MediaPipe detects hand landmarks.
2. The trained ASL model processes the landmarks.
3. The model outputs the predicted ASL sign.
4. If the model is uncertain, it prints **"Not sure"**, otherwise, it displays the recognized letter.

## 💡 Future Improvements
- Support multi-hand detection for two-handed signs.
- Implement moving gesture detection.

---
👋 **Created by [Pedro Ayon]**

