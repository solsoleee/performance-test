from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

# Flask 애플리케이션 초기화
app = Flask(__name__)

# 모델과 벡터라이저 로드
with open("model_and_vectorizer.dump", "rb") as f:
    loaded_data = pickle.load(f)

loaded_model = loaded_data['model']
loaded_vectorizer = loaded_data['vectorizer']

# 메인 페이지 (HTML 입력창)
@app.route('/')
def home():
    return render_template('index.html')

# POST 엔드포인트: 예측
@app.route('/predict', methods=['POST'])
def predict():
    title = request.form.get('title')
    content = request.form.get('content')

    if not title or not content:
        return jsonify({"error": "Invalid input. Provide both 'title' and 'content'."}), 400

    # 제목과 내용을 결합
    combined_text = title + ' ' + content
    new_data = pd.DataFrame({"data": [combined_text]})

    # 벡터화 및 예측
    X_new_tfidf = loaded_vectorizer.transform(new_data['data'])
    res = loaded_model.predict(X_new_tfidf)
    percentage = loaded_model.predict_proba(X_new_tfidf)[0]
    percentage0 = percentage[0]
    percentage1 = percentage[1]

    # 결과 HTML 렌더링
    return render_template('result.html', 
                           prediction=res[0], 
                           percentage0=percentage0, 
                           percentage1=percentage1)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
