import os
import pandas as pd

# 1. Import thêm TCNForecastModel
from utils.model import (
    ForecastModel,
    MockForcastModel,
    TCNForecastModel  # <--- Class mới
)

URL = (
    "https://raw.githubusercontent.com/nghiahsgs/iqair-dataset/refs/heads/main/"
    "result/ho-chi-minh-city/aqi_ho-chi-minh-city_2025_may.csv"
)

# Thư mục chứa model (Bạn cần copy file .h5 và .pkl vào đây)
MODEL_DIR = "./models/"

# ---- CÁC HÀM TẠO MODEL ----
def create_tcn(df: pd.DataFrame):
    """
    Hàm khởi tạo cho TCN Model.
    Nó sẽ load 2 file model .h5 và scaler .pkl từ MODEL_DIR.
    """
    # Định nghĩa đường dẫn file (khớp với tên file bạn đã lưu)
    path_24h = os.path.join(MODEL_DIR, "tcn_GLOBAL_task_24h_global.h5")
    path_7d = os.path.join(MODEL_DIR, "tcn_GLOBAL_task_7d_global.h5")
    path_scaler = os.path.join(MODEL_DIR, "scaler.pkl")

    # Kiểm tra file tồn tại để báo lỗi rõ ràng hơn
    if not os.path.exists(path_24h):
        raise FileNotFoundError(f"Missing model file: {path_24h}")

    # Load artifacts (Class TCNForecastModel có cơ chế cache, gọi nhiều lần không sao)
    # Hàm này sẽ load model vào RAM nếu chưa load
    TCNForecastModel.load_artifacts(path_24h, path_7d, path_scaler)
    
    # Trả về instance đã sẵn sàng sử dụng
    return TCNForecastModel(df)

def create_baseline(df: pd.DataFrame):
    return ForecastModel(df)

def create_mock(df: pd.DataFrame):
    return MockForcastModel(seed=42, base_aqi=80)

# ---- ĐĂNG KÝ MODEL ----
MODEL_REGISTRY = {
    # "lstm": create_lstm,
    "tcn": create_tcn,        # <--- Đăng ký key 'tcn'
    "baseline": create_baseline,
    "mock": create_mock,
}

def get_forecaster(df: pd.DataFrame):
    """
    Chọn model dựa trên biến môi trường AQI_MODEL.
    Mặc định mình đổi sang 'tcn' để test luôn.
    """
    # Bạn có thể set biến môi trường hoặc sửa default ở đây
    model_name = os.getenv("AQI_MODEL", "tcn") 
    
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown AQI_MODEL='{model_name}'. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )
    print(f"🚀 Initializing model: {model_name.upper()}")
    return MODEL_REGISTRY[model_name](df)

def main():
    # Load dữ liệu mẫu
    print(f"📥 Downloading data from {URL}...")
    df = pd.read_csv(URL)
    print(f"✅ Data loaded: {df.shape}")

    try:
        # Khởi tạo forecaster (Tự động load model TCN nếu AQI_MODEL='tcn')
        forecaster = get_forecaster(df)

        print("\n--- FORECAST RESULT ---")
        
        # Dự báo 24h
        forecast_24h = forecaster.do_forecast_aqi_24h()
        print(f"⏱️ Pred 24h:")
        for item in forecast_24h:
            print(f"   {item}")

        # Dự báo 7 ngày
        forecast_7d = forecaster.do_forecast_aqi_7day()
        print(f"📅 Pred 7 Days:")
        for item in forecast_7d:
            print(f"   {item}")

    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()