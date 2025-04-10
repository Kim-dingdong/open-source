import yfinance as yf
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# 🔧 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


# yfinance로 전체 기간 이더리움 가격 데이터 가져오기
eth = yf.Ticker("ETH-USD")
df = eth.history(period="max")
df.reset_index(inplace=True)

# 🔧 타임존 제거 (중요!)
df['Date'] = df['Date'].dt.tz_localize(None)

# Prophet용 컬럼 이름 변경
df = df[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})

# Prophet 모델 생성 및 학습
model = Prophet()
model.fit(df)

# 향후 30일 예측
future = model.make_future_dataframe(periods=365)
forecast = model.predict(future)

# 예측 정확도 측정
df_forecast = forecast.set_index('ds').join(df.set_index('ds'), how='left')
df_forecast = df_forecast.dropna(subset=['y'])

mae = mean_absolute_error(df_forecast['y'], df_forecast['yhat'])
rmse = np.sqrt(mean_squared_error(df_forecast['y'], df_forecast['yhat']))

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

# 시각화: 실제 + 예측
plt.figure(figsize=(12, 6))
plt.plot(df['ds'], df['y'], label='실제 이더리움 가격')
plt.plot(forecast['ds'], forecast['yhat'], label='예측 가격 (Prophet)', linestyle='--')
plt.xlabel('날짜')
plt.ylabel('가격 (USD)')
plt.title('이더리움 전체 기간 가격 예측')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 트렌드 및 계절성 시각화
model.plot_components(forecast)
plt.tight_layout()
plt.show()
