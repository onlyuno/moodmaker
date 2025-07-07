import os
from flask import Flask, request, render_template
from PIL import Image
import io
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
from flask import jsonify

app = Flask(__name__, template_folder='website/templates', static_folder='website/static')

class_name = [
    'beach_sunny_day', 'beach_night', 'city_sunny_day', 'city_rainy',
    'city_night', 'city_snowy', 'forest_sunny_day', 'forest_snowy_day'
]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.resnet18(weights=None)  # pretrained=False는 deprecated
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(class_name))  # 클래스 개수에 맞게
model.load_state_dict(torch.load('model_trained.pth', map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])  # ← 반드시 'POST' 명시
def predict():
    if 'file' not in request.files:
        return "파일이 없습니다", 400
    file = request.files['file']

    try:
        img = Image.open(file.stream).convert('RGB')
    except:
        return "이미지 파일이 아닙니다", 400

    input_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        pred = torch.argmax(output, 1).item()
        result = class_name[pred]

    return jsonify({"prediction": result})

# 서버 실행
if __name__ == '__main__':
    app.run(debug=True)