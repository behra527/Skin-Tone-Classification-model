# Skin Tone Classification Model

> Deep learning-based skin tone classification using **VGG16**, **MTCNN**, and **Flask**.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Flask](https://img.shields.io/badge/Flask-Backend-lightgrey?logo=flask)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange?logo=tensorflow)
![VGG16](https://img.shields.io/badge/Model-VGG16-green)
![Accuracy](https://img.shields.io/badge/Accuracy-98%25-brightgreen)
![License](https://img.shields.io/badge/License-MIT-blue)


## Overview

The **Skin Tone Classification Model** is a computer vision system designed to classify detected human faces into three skin tone categories:

* Black
* Brown
* White

The system combines:

* **VGG16**  deep learning model used for image feature extraction and classification.
* **MTCNN**  detects faces and extracts the relevant face regions before classification.
* **Flask**  provides the backend API and web application.
* **HTML, CSS, JavaScript**  provides the user interface for image upload and prediction.

The complete pipeline allows users to upload an image and receive a predicted skin tone with a confidence score.



## Key Features

* **VGG16-based classification** using a pretrained CNN architecture.
* **MTCNN face detection** before classification.
* **Automatic face cropping and preprocessing**.
* **Flask backend** for serving predictions.
* **Web-based frontend** for image uploads.
* **Confidence score** returned with each prediction.
* **Custom dataset support** for future retraining.
* Modular project structure for easier development and maintenance.



## System Architecture

```text
Input Image
     │
     ▼
MTCNN Face Detection
     │
     ▼
Face Extraction & Preprocessing
     │
     ▼
VGG16 Classification Model
     │
     ▼
Predicted Skin Tone
     │
     ▼
Confidence Score
     │
     ▼
Flask Backend
     │
     ▼
Web Interface
```



## Tech Stack

| Component            | Technology                  |
| -------------------- | --------------------------- |
| Programming Language | Python                      |
| Deep Learning        | TensorFlow / Keras          |
| CNN Architecture     | VGG16                       |
| Face Detection       | MTCNN                       |
| Backend              | Flask                       |
| Computer Vision      | OpenCV                      |
| Image Processing     | Pillow                      |
| Frontend             | HTML, CSS, JavaScript       |
| Dataset              | Custom-labeled face dataset |



## Model Performance

The model achieved the following results on the evaluated dataset:

| Metric    | Score |
| --------- | ----: |
| Accuracy  |   98% |
| Precision |   97% |
| Recall    |   98% |
| F1-Score  | 97.5% |

> **Note:** These metrics depend on the dataset, preprocessing pipeline, train/test split, and evaluation methodology. They should not be interpreted as evidence of real-world fairness or generalization without evaluation on diverse independent datasets.



## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/skin-tone-classification.git
cd skin-tone-classification
```

### 2. Create a Virtual Environment

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### Example `requirements.txt`

```text
Flask==3.1.0
tensorflow==2.16.1
mtcnn==0.1.1
opencv-python==4.10.0.84
numpy==1.26.4
pillow==10.4.0
```

> Adjust package versions according to your Python and TensorFlow environment.



## Run the Application

Start the Flask server:

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:5000/
```

Upload an image through the web interface to generate a prediction.



## Example Prediction

Example API/application output:

```json
{
  "file_name": "person_01.jpg",
  "detected_faces": 1,
  "predicted_tone": "Brown",
  "confidence": 0.984
}
```

Example frontend result:

```text
Predicted Skin Tone: Brown
Confidence: 98.4%
```



## Project Structure

```text
skin-tone-classification/
│
├── app.py
│
├── model/
│   ├── vgg16_model.h5
│   └── model_builder.py
│
├── utils/
│   ├── face_detector.py
│   └── preprocess.py
│
├── static/
│   └── uploads/
│
├── templates/
│   └── index.html
│
├── requirements.txt
└── README.md
```

### Main Components

| File / Directory         | Purpose                                   |
| ------------------------ | ----------------------------------------- |
| `app.py`                 | Flask application and prediction handling |
| `model/vgg16_model.h5`   | Trained VGG16 model                       |
| `model/model_builder.py` | Model architecture and configuration      |
| `utils/face_detector.py` | MTCNN face detection                      |
| `utils/preprocess.py`    | Image preprocessing                       |
| `templates/index.html`   | Frontend interface                        |
| `static/uploads/`        | Uploaded images                           |
| `requirements.txt`       | Python dependencies                       |

---

## How It Works

### 1. Image Upload

The user uploads an image through the web interface.

### 2. Face Detection

MTCNN identifies faces in the uploaded image.

### 3. Face Extraction

The detected face region is cropped from the original image.

### 4. Preprocessing

The cropped face is resized and prepared according to the model's input requirements.

### 5. Classification

The processed face is passed to the VGG16-based classification model.

### 6. Prediction

The application returns:

* Predicted skin tone
* Confidence score
* Number of detected faces



## Future Improvements

Possible improvements include:

* Evaluate the model on larger and more diverse datasets.
* Analyze performance across different lighting conditions.
* Add fairness and bias evaluation.
* Support real-time webcam prediction.
* Improve face detection and preprocessing.
* Experiment with lightweight architectures such as MobileNet.
* Add Docker-based deployment.
* Deploy the model as a cloud API.
* Add automated model evaluation and testing.
* Optimize inference speed for production environments.



## Limitations

This project is intended as a **computer vision and machine learning project** and should not be used to make decisions about people based solely on predicted skin tone.

Model performance can be affected by:

* Lighting conditions
* Camera quality
* Image resolution
* Face orientation
* Dataset composition
* Dataset labeling quality
* Demographic and geographic diversity

Additional validation is required before using the model in real-world applications.



## Contributing

Contributions are welcome.

To contribute:

1. Fork the repository.
2. Create a new branch.
3. Make your changes.
4. Commit your changes.
5. Open a pull request.

For major changes, please open an issue first to discuss the proposed changes.



## License

This project is licensed under the **MIT License**.

See the `LICENSE` file for more information.



## Author

**Muhammad Behram Hassan**

AI Engineer | Machine Learning | Deep Learning | Generative AI



## Acknowledgments

This project uses:

* TensorFlow / Keras
* VGG16
* MTCNN
* Flask
* OpenCV
* Pillow


