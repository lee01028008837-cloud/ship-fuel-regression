import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# ==========================================
# 1. 데이터 불러오기
# ==========================================

df = pd.read_csv("tanker.csv", sep=";")

print("===== 데이터 정보 =====")
print("데이터 개수:", len(df))
print("변수 이름:")
print(df.columns.tolist())
print()


# ==========================================
# 2. 분석 변수 설정
# ==========================================

X = df[["Speed"]]
y = df["fuelcons"]


# ==========================================
# 3. 원자료 산점도
# ==========================================

plt.figure(figsize=(10, 6))

plt.scatter(
    df["Speed"],
    df["fuelcons"],
    alpha=0.7
)

plt.xlabel("Ship Speed (knots)")
plt.ylabel("Fuel Consumption (g/s)")
plt.title("Ship Speed vs Fuel Consumption")

plt.grid(True)
plt.show()


# ==========================================
# 4. 학습 데이터와 테스트 데이터 분리
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("===== 데이터 분할 =====")
print("전체 데이터:", len(df))
print("학습 데이터:", len(X_train))
print("테스트 데이터:", len(X_test))
print()


# ==========================================
# 5. 선형회귀
# ==========================================

linear_model = LinearRegression()

linear_model.fit(
    X_train,
    y_train
)

linear_pred = linear_model.predict(
    X_test
)

linear_r2 = r2_score(
    y_test,
    linear_pred
)

linear_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        linear_pred
    )
)

linear_mae = mean_absolute_error(
    y_test,
    linear_pred
)

print("===== 선형회귀 평가 결과 =====")
print("R²:", linear_r2)
print("RMSE:", linear_rmse)
print("MAE:", linear_mae)
print()


# ==========================================
# 6. 비선형회귀
#    2차 다항회귀
# ==========================================

poly_model = make_pipeline(
    PolynomialFeatures(
        degree=2,
        include_bias=False
    ),
    LinearRegression()
)

poly_model.fit(
    X_train,
    y_train
)

poly_pred = poly_model.predict(
    X_test
)

poly_r2 = r2_score(
    y_test,
    poly_pred
)

poly_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        poly_pred
    )
)

poly_mae = mean_absolute_error(
    y_test,
    poly_pred
)

print("===== 2차 다항회귀 평가 결과 =====")
print("R²:", poly_r2)
print("RMSE:", poly_rmse)
print("MAE:", poly_mae)
print()


# ==========================================
# 7. 선형회귀와 비선형회귀 그래프 비교
# ==========================================

x_range = np.linspace(
    df["Speed"].min(),
    df["Speed"].max(),
    300
).reshape(-1, 1)

linear_curve = linear_model.predict(
    x_range
)

poly_curve = poly_model.predict(
    x_range
)

plt.figure(figsize=(10, 6))

plt.scatter(
    df["Speed"],
    df["fuelcons"],
    alpha=0.5,
    label="Actual Data"
)

plt.plot(
    x_range,
    linear_curve,
    linewidth=2,
    label="Linear Regression"
)

plt.plot(
    x_range,
    poly_curve,
    linewidth=2,
    label="Polynomial Regression"
)

plt.xlabel("Ship Speed (knots)")
plt.ylabel("Fuel Consumption (g/s)")
plt.title("Linear vs Nonlinear Regression")

plt.legend()
plt.grid(True)

plt.show()


# ==========================================
# 8. 모델 성능 비교
# ==========================================

result = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Polynomial Regression (degree=2)"
    ],
    "R2": [
        linear_r2,
        poly_r2
    ],
    "RMSE": [
        linear_rmse,
        poly_rmse
    ],
    "MAE": [
        linear_mae,
        poly_mae
    ]
})

print("===== 모델 성능 비교 =====")
print(result)
print()


# ==========================================
# 9. 선박 속도를 입력하여 연료소비량 예측
# ==========================================

speed_input = float(
    input(
        "예측할 선박 속도(knots)를 입력하세요: "
    )
)

input_data = pd.DataFrame({
    "Speed": [speed_input]
})

linear_result = linear_model.predict(
    input_data
)[0]

poly_result = poly_model.predict(
    input_data
)[0]

print()
print("===== 연료소비량 예측 결과 =====")
print(
    f"입력한 선박 속도: "
    f"{speed_input:.2f} knots"
)

print(
    f"선형회귀 예측값: "
    f"{linear_result:.2f} g/s"
)

print(
    f"비선형회귀 예측값: "
    f"{poly_result:.2f} g/s"
)
