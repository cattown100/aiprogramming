from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

# Flask 앱 생성
app = Flask(__name__)

# 업로드 폴더
UPLOAD_FOLDER = "uploads"

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 폴더 없으면 생성
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 저장된 RandomForest 모델 로드
model = joblib.load("random_forest_model.pkl")

# 메인 페이지
@app.route("/")
def home():
    return render_template("index.html")


# CSV 업로드 및 예측
@app.route("/predict", methods=["POST"])
def predict():

    # 파일 체크
    if 'file' not in request.files:
        return "파일이 없습니다."

    file = request.files['file']

    if file.filename == '':
        return "파일을 선택하세요."

    # CSV만 허용
    if not file.filename.endswith(".csv"):
        return "CSV 파일만 업로드 가능합니다."

    # 파일 저장
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

    # 문자열 컬럼 인코딩
    non_numeric = df.select_dtypes(
        include=['object', 'category']
    ).columns.tolist()

    if non_numeric:
        df = pd.get_dummies(
            df,
            columns=non_numeric,
            drop_first=True
        )

    # 예측
    predictions = model.predict(df)

    # 결과 추가
    result_df = df.copy()

    result_df['Prediction'] = predictions

    # 악성/정상 개수
    malicious_count = (predictions == 1).sum()
    normal_count = (predictions == 0).sum()

    # HTML 테이블 변환
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


# 서버 실행
if __name__ == "__main__":
    app.run(debug=True)
