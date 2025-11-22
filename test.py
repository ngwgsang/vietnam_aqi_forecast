import os
import pandas as pd

from utils.model import (
    LSTMForecastModel,
    ForecastModel,
    MockForcastModel,
)

URL = (
    "https://raw.githubusercontent.com/nghiahsgs/iqair-dataset/refs/heads/main/"
    "result/ho-chi-minh-city/aqi_ho-chi-minh-city_2025_may.csv"
)

# Thư mục chứa model LSTM
MODEL_DIR = "./models/"

# ---- NHỎ GỌN: registry các model có thể dùng ----
def create_lstm(df: pd.DataFrame):
    # Nếu cần đổi tên file model / scaler / config, chỉnh ở đây
    artifacts = {
        "model": "best_aqi_lstm.h5",
        "scaler_X": "scaler_X.pkl",
        "scaler_y": "scaler_y.pkl",
        "config": "model_config.json",
    }
    return LSTMForecastModel(df, MODEL_DIR, artifacts=artifacts)

def create_baseline(df: pd.DataFrame):
    return ForecastModel(df)

def create_mock(df: pd.DataFrame):
    # Ví dụ demo / test
    return MockForcastModel(seed=42, base_aqi=80)

MODEL_REGISTRY = {
    "lstm": create_lstm,
    "baseline": create_baseline,
    "mock": create_mock,
}

def get_forecaster(df: pd.DataFrame):
    """
    Chọn model dựa trên biến môi trường AQI_MODEL.
    Mặc định: 'lstm'
    Ví dụ:
      - AQI_MODEL=baseline
      - AQI_MODEL=mock
      - AQI_MODEL=lstm
    """
    model_name = os.getenv("AQI_MODEL", "lstm")
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown AQI_MODEL='{model_name}'. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[model_name](df)

def main():
    df = pd.read_csv(URL)
    print(df.head())

    try:
        forecaster = get_forecaster(df)

        # Dự báo 24h
        forecast_24h = forecaster.do_forecast_aqi_24h()
        print("Pred 24h (first 2):", forecast_24h[:2])

        # Dự báo 7 ngày
        forecast_7d = forecaster.do_forecast_aqi_7day()
        print("Pred 7 ngày:", forecast_7d)

        # Nếu model là LSTM và config có metrics:
        if hasattr(forecaster, "metrics") and forecaster.metrics is not None:
            print("Metrics (from config):", forecaster.metrics)

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
