import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier

print("CSV 데이터 로드 중...")

# 데이터 불러오기
df = pd.read_csv("dataset/top_1000_pe_imports.csv")

print(f"데이터 크기: {df.shape}")

# 필수 컬럼 확인
required_cols = {'hash', 'malware'}

missing = required_cols - set(df.columns)

if missing:
    raise Exception(f"필수 컬럼 없음: {missing}")

# X, y 분리
X = df.drop(columns=['hash', 'malware'])
y = df['malware']

# 문자열 라벨 숫자 변환
if y.dtype == object:
    y = y.astype('category').cat.codes

# 결측치 처리
X = X.fillna(0)

# 범주형 인코딩
non_numeric = X.select_dtypes(
    include=['object', 'category']
).columns.tolist()

if non_numeric:
    X = pd.get_dummies(
        X,
        columns=non_numeric,
        drop_first=True
    )

# 랜덤포레스트로 중요도 계산
print("변수 중요도 분석 중...")

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

rf.fit(X, y)

# 중요도 계산
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf.feature_importances_
})

importance_df = importance_df.sort_values(
    by='Importance',
    ascending=False
)

# 상위 30개 변수 선택
top_30_features = importance_df.head(30)['Feature'].tolist()

print("\n상위 중요 변수 TOP 5")

for i, row in importance_df.head(5).iterrows():
    print(f"{row['Feature']} : {row['Importance']:.4f}")

# 최적화 데이터 생성
optimized_df = pd.concat([
    X[top_30_features],
    y.reset_index(drop=True)
], axis=1)

# 저장
optimized_df.to_csv(
    "optimized_data.csv",
    index=False
)

# 선택 변수 저장
joblib.dump(
    top_30_features,
    "selected_features.pkl"
)

print("\n최적화 완료!")
print("optimized_data.csv 저장 완료")