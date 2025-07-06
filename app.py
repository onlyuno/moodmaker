from flask import Flask, request, jsonify
import os

app = Flask(__name__)

@app.route("/")
def home():
    return "Hello from Render!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    user_input = data.get("input")

    # 예시 예측 로직
    result = f"당신이 보낸 값은 '{user_input}'입니다."
    
    return jsonify({"result": result})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render는 환경변수 PORT를 사용
    app.run(host="0.0.0.0", port=port)