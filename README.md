# Medical Image Processing and Diagnosis Web Application

## Overview
This project is a **Flask-based web application** that provides two main functionalities:
1. **OCR-based Prescription Reader**: Extracts text from uploaded prescription images using Tesseract OCR.
2. **AI-based Medical Image Diagnosis**: Analyzes uploaded medical scan images using a deep learning model (ResNet18) to classify whether a scan is normal or has abnormalities.

## Features
- Web-based UI for uploading images.
- **Optical Character Recognition (OCR)** for extracting text from medical prescriptions.
- **Deep Learning-based Classification** for diagnosing medical scan images.
- RESTful API endpoints for prescription text extraction and medical image diagnosis.
- Uses **PyTorch** and **Tesseract OCR** for processing images.

## Technologies Used
- **Flask** (Backend web framework)
- **OpenCV** (Image processing for OCR)
- **Pytesseract** (OCR for text extraction)
- **PyTorch & torchvision** (Deep learning model for medical image classification)
- **PIL (Pillow)** (Image handling)
- **HTML & JavaScript** (Frontend for uploading images)

## Installation & Setup
### Prerequisites
Ensure you have **Python 3.8+** installed along with the required dependencies.

### Install Dependencies
Run the following command to install necessary Python packages:
```sh
pip install flask opencv-python pytesseract torch torchvision pillow
```

### Run the Flask Application
```sh
python app.py
```
The application will start and be accessible at `http://localhost:5000`.

## Usage
### Uploading Images
1. **Visit `http://localhost:5000`** in your browser.
2. **Upload a Prescription Image**: Extracts and displays the text.
3. **Upload a Medical Scan**: Classifies the image as "Normal Scan" or "Abnormality Detected."

### API Endpoints
#### 1. Prescription OCR
**Endpoint:** `POST /prescription`
- **Input:** Upload a prescription image.
- **Response:** Extracted text in JSON format.

#### 2. Medical Scan Diagnosis
**Endpoint:** `POST /diagnose`
- **Input:** Upload a medical scan image.
- **Response:** Diagnosis result (Normal/Abnormal) in JSON format.

## File Structure
```
.
├── app.py  # Main Flask application
├── requirements.txt  # Dependencies list
├── templates/
│   ├── index.html  # HTML template for UI
├── static/
│   ├── styles.css  # CSS for styling (if applicable)
└── README.md  # Project Documentation
```

## Notes
- Ensure **Tesseract-OCR** is installed and added to the system PATH for OCR to work.
- The deep learning model used is **ResNet18**, pre-trained on ImageNet and fine-tuned for binary classification.

## Future Improvements
- Improve OCR accuracy with better preprocessing techniques.
- Train a more specialized medical image classifier.
- Deploy the model to a cloud-based service for scalability.

## License
This project is open-source under the MIT License.
