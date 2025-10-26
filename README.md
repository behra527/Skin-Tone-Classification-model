# 🎨 Skin Tone Classification Model
> 🧠 *Deep learning model for skin tone detection using VGG16, MTCNN & Flask*

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Flask](https://img.shields.io/badge/Flask-Backend-lightgrey?logo=flask)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange?logo=tensorflow)
![VGG16](https://img.shields.io/badge/Model-VGG16-green)
![Accuracy](https://img.shields.io/badge/Accuracy-98%25-brightgreen)
![License](https://img.shields.io/badge/License-MIT-blue)

---

## 🧩 Overview

The **Skin Tone Classification Model** classifies human skin tones into three categories:  
⚫ **Black** · 🟤 **Brown** · ⚪ **White**  

It uses **VGG16** for feature extraction, **MTCNN** for face detection, and a **Flask backend** to serve predictions via a web interface.  
The system is fully connected with a **frontend UI**, allowing users to upload images and view results in real time.

---

## 🎯 Key Features

- 🧠 **VGG16 Deep Learning Model** — Pretrained CNN fine-tuned for skin tone classification  
- 👤 **MTCNN Face Detection** — Automatically detects and crops faces before classification  
- 🌐 **Flask Backend** — Handles image uploads and returns predictions  
- 💻 **Frontend Integration** — User-friendly web interface for real-time use  
- 🎯 **High Accuracy** — Achieved **98% accuracy** on the custom dataset  
- 🧰 **Custom Dataset Support** — Easy retraining on new datasets  

---

## 🛠️ Tech Stack

| Component       | Technology                     |
|-----------------|--------------------------------|
| Language        | Python                         |
| Backend         | Flask                          |
| Deep Learning   | TensorFlow / Keras (VGG16)    |
| Face Detection  | MTCNN                          |
| Frontend        | HTML, CSS, JavaScript          |
| Accuracy        | 98%                            |
| Dataset         | Custom-labeled face dataset    |

---

## ⚙️ Installation & Setup

### 1️⃣ Clone Repository
```bash
git clone https://github.com/yourusername/skin-tone-classification.git
cd skin-tone-classification
2️⃣ Create Virtual Environment
bash
Copy code
python -m venv venv
source venv/bin/activate      # macOS / Linux
venv\Scripts\activate         # Windows
3️⃣ Install Dependencies
bash
Copy code
pip install -r requirements.txt
4️⃣ Example requirements.txt
ini
Copy code
Flask==3.1.0
tensorflow==2.16.1
mtcnn==0.1.1
opencv-python==4.10.0.84
numpy==1.26.4
pillow==10.4.0
🚀 Run the Project
▶️ Start Flask Server
bash
Copy code
python app.py
Open in your browser:
👉 http://127.0.0.1:5000/

🧪 Test with an Image
Upload an image through the web interface to see the predicted skin tone.

📊 Example Output
Input: User uploads an image via the frontend.

Model Prediction:

json
Copy code
{
  "file_name": "person_01.jpg",
  "detected_faces": 1,
  "predicted_tone": "Brown",
  "confidence": 0.984
}
Frontend Display:

🟤 Predicted Skin Tone: Brown
✅ Confidence: 98.4%

📁 Project Structure
csharp
Copy code
skin-tone-classification/
│
├── app.py                    # Flask backend server
├── model/
│   ├── vgg16_model.h5         # Trained model weights
│   └── model_builder.py       # VGG16 model definition
├── utils/
│   ├── face_detector.py       # MTCNN face detection
│   └── preprocess.py          # Image preprocessing logic
├── static/
│   └── uploads/               # Uploaded images
├── templates/
│   └── index.html             # Frontend HTML page
├── requirements.txt
└── README.md


📈 Model Performance
Metric	Value
Accuracy	98%
Precision	97%
Recall	98%
F1-Score	97.5%

🔮 Future Enhancements
🌍 Add more diverse datasets for global skin tone fairness

🎥 Real-time webcam classification

🧠 Convert to MobileNet for mobile deployment

☁️ Cloud API deployment with Docker + AWS

🤝 Contributing
Contributions and feedback are welcome!
Please open an issue or submit a pull request for improvements.

📜 License
This project is licensed under the MIT License — see the LICENSE file for details.

👨‍💻 Author
Muhammad Behram Hassan
📧 muhammadbehramhassan@gmail.com
🌐 GitHub

⭐ If this project helps you, please give it a star on GitHub!






