import os
import gdown
from flask import Flask, request, render_template
import torch
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models
import torch.nn as nn
from flask import jsonify

def download_model():
    MODEL_PATH = 'model_trained.pth'
    DRIVE_FILE_ID = '12CSTGv0Gx8IXoCN4XZ8isSDsNe8wVoF1'  # Google Drive 파일 ID로 변경하세요
    url = f'https://drive.google.com/uc?id={DRIVE_FILE_ID}'

    print("📥 다운로드 시도 중:", url)

    if not os.path.exists(MODEL_PATH):
        try:
            gdown.download(url, MODEL_PATH, quiet=False)
            if os.path.exists(MODEL_PATH):
                print("✅ 다운로드 성공")
            else:
                print("❌ 다운로드 실패: 파일이 존재하지 않음")
                exit(1)
        except Exception as e:
            print(f"❌ 다운로드 중 에러 발생: {e}")
            exit(1)
    else:
        print("📁 모델 파일이 이미 존재함")

download_model()

app = Flask(__name__)

# --- 여기에 모델 불러오기 및 예측 함수 넣기 시작 ---

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.resnet18(weights=None)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 8)  # 클래스 개수 맞게 설정

try:
    model.load_state_dict(torch.load('model_trained.pth', map_location=device))
    model.to(device)
    model.eval()
    print("✅ 모델 로딩 완료.")
except Exception as e:
    print(f"❌ 모델 로딩 실패: {e}")
    exit(1)

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

    upload_dir = os.path.join(os.getcwd(), "tmp")
    os.makedirs(upload_dir, exist_ok=True)
    upload_path = os.path.join(upload_dir, file.filename)
    file.save(upload_path)

    pred_idx = predict_image(upload_path)

    keywords_map = {
        0: "beach_sunny_day",
        1: "beach_night",
        2: "city_sunny_day",
        3: "city_rainy",
        4: "city_night",
        5: "city_snowy",
        6: "forest_sunny_day",
        7: "forest_snowy_day",
    }
    keyword = keywords_map.get(pred_idx, "unknown")

    return jsonify({"prediction": keyword})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render는 환경변수 PORT를 사용
    app.run(host="0.0.0.0", port=port)