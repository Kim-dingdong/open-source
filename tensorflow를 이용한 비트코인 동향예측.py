import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 한글 폰트 설정 (Matplotlib에서 한글이 깨지지 않도록)
font_path = "C:/Windows/Fonts/malgun.ttf"  # 윈도우 기본 한글 폰트 (Malgun Gothic)
font_prop = font_manager.FontProperties(fname=font_path)
plt.rc('font', family=font_prop.get_name())  # 한글 폰트 적용

# CSV 파일 불러오기
df = pd.read_csv("btcusd_1-min_data.csv")

# datetime 처리 및 정렬
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.sort_values('datetime')
df = df.set_index('datetime')

# Close 컬럼만 사용
df = df[['Close']].dropna()

# 정규화
scaler = MinMaxScaler()
scaled_close = scaler.fit_transform(df[['Close']])

# LSTM 시퀀스 생성 함수
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i])
        y.append(data[i])
    return np.array(X), np.array(y)

# 시퀀스 길이 설정
SEQ_LEN = 60
X, y = create_sequences(scaled_close, SEQ_LEN)

# 학습 / 테스트 분리
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# LSTM 입력 차원 맞추기
X_train = X_train.reshape((-1, SEQ_LEN, 1))
X_test = X_test.reshape((-1, SEQ_LEN, 1))

# LSTM 모델 생성
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(SEQ_LEN, 1)),
    LSTM(50),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')

# 모델 학습
model.fit(X_train, y_train, epochs=5, batch_size=64)

# 학습 후 모델 저장
model.save('bitcoin_model.h5')  # 모델을 'bitcoin_model.h5'로 저장
print("모델 저장 완료!")  # 저장 완료 메시지 출력

# 예측
predicted = model.predict(X_test)
predicted_price = scaler.inverse_transform(predicted)
real_price = scaler.inverse_transform(y_test.reshape(-1, 1))

# 시각화
plt.figure(figsize=(12,6))
plt.plot(real_price, label='실제 비트코인 가격')
plt.plot(predicted_price, label='예측 가격 (LSTM)', linestyle='--')
plt.xlabel('시간')
plt.ylabel('가격 (USD)')
plt.title('LSTM 기반 비트코인 가격 예측 결과', fontsize=14)  # 한글 제목 지원
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 향후 5년 예측 (예측 기간 5년으로 확장)
future = model.make_future_dataframe(df, periods=5*365*24*60)  # 5년 (5년치 데이터)
forecast = model.predict(future)
forecast_price = scaler.inverse_transform(forecast)

# 5년 예측 시각화
plt.figure(figsize=(12,6))
plt.plot(future['ds'], forecast_price, label='예측 가격 (5년)', linestyle='--')
plt.xlabel('날짜')
plt.ylabel('가격 (USD)')
plt.title('LSTM 기반 5년 후 비트코인 가격 예측', fontsize=14)  # 한글 제목 지원
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
