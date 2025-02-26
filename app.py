from flask import Flask, request, jsonify, render_template_string
import cv2
import pytesseract
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models

app = Flask(_name_)

# HTML Form Template for Uploading Images
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Upload Image</title>
</head>
<body>
    <h2>Upload an Image</h2>
    <form action="/prescription" method="post" enctype="multipart/form-data">
        <label>Upload Prescription Image:</label>
        <input type="file" name="file" required>
        <button type="submit">Submit</button>
    </form>

    <br>

    <form action="/diagnose" method="post" enctype="multipart/form-data">
        <label>Upload Medical Scan Image:</label>
        <input type="file" name="file" required>
        <button type="submit">Submit</button>
    </form>
</body>
</html>
"""

# OCR-based Prescription Reader
def extract_text_from_prescription(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray)
    return text.strip()

# AI-based Medical Image Analysis
class MedicalImageClassifier:
    def _init_(self):
        self.model = models.resnet18(pretrained=True)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 2)  # Binary classification
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image_path):
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image).unsqueeze(0)
        with torch.no_grad():
            output = self.model(image)
            prediction = torch.argmax(output, dim=1).item()
        return "Abnormality Detected" if prediction == 1 else "Normal Scan"

classifier = MedicalImageClassifier()

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/prescription', methods=['GET', 'POST'])
def process_prescription():
    if request.method == 'GET':
        return render_template_string(HTML_TEMPLATE)

    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    file_path = "temp_prescription.jpg"
    file.save(file_path)
    text = extract_text_from_prescription(file_path)
    return jsonify({"extracted_text": text})

@app.route('/diagnose', methods=['GET', 'POST'])
def diagnose_image():
    if request.method == 'GET':
        return render_template_string(HTML_TEMPLATE)

    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    file_path = "temp_image.jpg"
    file.save(file_path)
    diagnosis = classifier.predict(file_path)
    return jsonify({"diagnosis_result": diagnosis})

if _name_ == "_main_":
    app.run(host='0.0.0.0', port=5000, debug=True)