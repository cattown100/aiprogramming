from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 모델 로드
model = joblib.load("random_forest_model.pkl")

# 선택 feature 로드
selected_features = joblib.load(
    "selected_features.pkl"
)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    if 'file' not in request.files:
        return "파일이 없습니다."

    file = request.files['file']

    if file.filename == '':
        return "파일 선택 필요"

    filepath = os.path.join(
        app.config['UPLOAD_FOLDER'],
        file.filename
    )

    file.save(filepath)

    # CSV 읽기
    df = pd.read_csv(filepath)

    # hash 제거
    if 'hash' in df.columns:
        df = df.drop(columns=['hash'])

    # malware 제거
    if 'malware' in df.columns:
        df = df.drop(columns=['malware'])

    # 결측치 처리
    df = df.fillna(0)

    # 문자형 인코딩
    non_numeric = df.select_dtypes(
        include=['object', 'category']
    ).columns.tolist()

    if non_numeric:
        df = pd.get_dummies(
            df,
            columns=non_numeric,
            drop_first=True
        )

    # 누락 컬럼 추가
    for col in selected_features:
        if col not in df.columns:
            df[col] = 0

    # 컬럼 순서 맞추기
    df = df[selected_features]

    # 예측
    predictions = model.predict(df)

    result_df = df.copy()

    result_df['Prediction'] = predictions

    malicious_count = (predictions == 1).sum()
    normal_count = (predictions == 0).sum()

    table = result_df.head(20).to_html(
        classes='table',
        index=False
    )

    return render_template(
        "index.html",
        table=table,
        malicious=malicious_count,
        normal=normal_count
    )

if __name__ == "__main__":
    app.run(debug=True)