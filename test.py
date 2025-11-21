import pandas as pd
from utils.model import LSTMForecastModel
url = "https://raw.githubusercontent.com/nghiahsgs/iqair-dataset/refs/heads/main/result/ho-chi-minh-city/aqi_ho-chi-minh-city_2025_may.csv"

df = pd.read_csv(url)
print(df.head())


# Giả sử bạn đã có DataFrame 'df_history' và đường dẫn thư mục chứa model
MODEL_PATH = './models/'

try:
    # Khởi tạo model
    forecaster = LSTMForecastModel(df, MODEL_PATH)

    # Dự báo 24h
    forecast_24h = forecaster.do_forecast_aqi_24h()
    print("Pred 24h:", forecast_24h[:2]) # In thử 2 dòng

    # Dự báo 7 ngày
    forecast_7d = forecaster.do_forecast_aqi_7day()
    print("Pred 7 ngày:", forecast_7d)

except Exception as e:
    print(f"Error: {e}")