import os
from flask import Flask, request, render_template
import torch
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models
import torch.nn as nn

app = Flask(__name__)

# --- 여기에 모델 불러오기 및 예측 함수 넣기 시작 ---

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.resnet18(pretrained=False)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 8)  # 클래스 개수 맞게 설정
model.load_state_dict(torch.load('model_trained.pth', map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
    return predicted.item()

from flask import Flask, request, jsonify
import os

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return "파일이 없습니다.", 400
    file = request.files['file']
    if file.filename == '':
        return "파일 이름이 없습니다.", 400

    upload_path = os.path.join("uploads", file.filename)
    os.makedirs("uploads", exist_ok=True)
    file.save(upload_path)

    pred_idx = predict_image(upload_path)

    keywords_map = {
        0: "city sunny day",
        1: "mountain cloudy",
        2: "beach sunset",
        3: "forest rainy",
        4: "night city lights",
        5: "snow winter",
        6: "desert hot",
        7: "park spring flowers",
    }
    keyword = keywords_map.get(pred_idx, "unknown")

    return render_template("result.html", prediction=keyword)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render는 환경변수 PORT를 사용
    app.run(host="0.0.0.0", port=port)