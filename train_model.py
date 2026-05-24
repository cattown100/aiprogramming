import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

print("전처리 데이터 로드 중...")

df = pd.read_csv("optimized_data.csv")

# X, y 분리
X = df.drop(columns=['malware'])
y = df['malware']

# 학습 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# RandomForest 모델 생성
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    random_state=42,
    n_jobs=-1
)

print("AI 학습 중...")

# 학습
model.fit(X_train, y_train)

# 예측
predictions = model.predict(X_test)

# 정확도
accuracy = accuracy_score(
    y_test,
    predictions
)

print(f"\n모델 정확도: {accuracy * 100:.2f}%")

# 모델 저장
joblib.dump(
    model,
    "random_forest_model.pkl"
)

print("모델 저장 완료!")