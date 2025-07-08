import os
from flask import Flask, request, render_template, jsonify
from PIL import Image
import io
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# Spotify 개발자 페이지에서 받은 키
client_id = 'b52aa9c8561f49b7afb96e82dd43b728'
client_secret = '0918d176603f41879d11adfec5a1304e'
auth_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(auth_manager=auth_manager)

# 예시: forest_snowy_day → 분위기 키워드로 검색
import random

# 키워드별 핵심 단어 리스트
mood_word_lists = {
    "forest_snowy_day": ["dark", "mysterious", "indie", "Sufjan", "Stevens", "ambient", "snowy", "forest", "ethereal", "acoustic"],
    "city_night": ["rhythmic", "city", "pop", "R&B", "smooth", "urban", "chill", "bass", "laid-back", "night"],
    "beach_sunny_day": ["bright", "summer", "pop", "chill", "beach", "waves", "fresh", "acoustic", "sunny", "uplifting"],
    "city_rainy": ["jazzy", "acoustic", "cafe", "soft", "rainy", "day", "cozy", "blues", "melancholy", "indie"],
    "city_sunny_day": ["energetic", "city", "walking", "pop", "powerful", "upbeat", "career", "woman", "anthem", "AJR"],
    "city_snowy": ["warm", "winter", "city", "jazz", "soft", "Christmas", "carol", "holiday", "chill", "snowy"],
    "beach_night": ["mysterious", "ocean", "acoustic", "Moana", "campfire", "chill", "soft", "youth", "ballad", "calm"],
    "forest_sunny_day": ["bright", "sunny", "forest", "indie", "happy", "bird", "song", "fairytale", "warm", "nature"]
}

def get_tracks_by_mood(label):
    words = mood_word_lists.get(label, ["relaxing", "music"])

    # 2~3개 단어를 랜덤 선택해서 조합 (중복 없는 샘플링)
    selected_words = random.sample(words, k=random.randint(2,3))
    query = " ".join(selected_words)

    offset = random.randint(0, 50)
    results = sp.search(q=query, type='track', limit=5, offset=offset)
    
    tracks = []
    for item in results['tracks']['items']:
        tracks.append({
            'name': item['name'],
            'artist': item['artists'][0]['name'],
            'url': item['external_urls']['spotify']
        })
    return tracks


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

    tracks = get_tracks_by_mood(result)

    return jsonify({
        "prediction": result,
        "tracks": tracks
    })

# 서버 실행
if __name__ == '__main__':
    app.run(debug=True)