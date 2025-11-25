import os
import pandas as pd
import sys
import datetime  # <--- Cần thêm thư viện này để lấy ngày tháng

# Thêm thư mục hiện tại vào sys.path để import được utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.model import (
    ForecastModel,
    MockForcastModel,
    TCNForecastModel
)

# ==========================================
# 1. CẤU HÌNH URL (COPY TỪ MAIN.PY)
# ==========================================
CITY_SLUG = "ho-chi-minh-city"
BASE_URL = (
    "https://raw.githubusercontent.com/HiAmNear/iqair-crawling"
    "/refs/heads/main/result/{city}/aqi_{city}_{year}_{month}.csv"
)

# Map số tháng -> suffix trong tên file
MONTH_SLUGS = [
    "jan", "feb", "mar", "apr", "may", "jun",
    "jul", "aug", "sep", "oct", "nov", "dec"
]

def build_month_urls(city_slug: str, year: int):
    """Tạo list URL từ tháng 1 đến tháng hiện tại của năm `year`."""
    today = datetime.date.today()
    current_month = today.month

    urls = []
    for m in range(1, current_month + 1):
        month_slug = MONTH_SLUGS[m - 1]
        url = BASE_URL.format(
            city=city_slug,
            year=year,
            month=month_slug,
        )
        urls.append(url)
    return urls

# TẠO URL DỮ LIỆU
YEAR = 2025
all_urls = build_month_urls(CITY_SLUG, YEAR)

# Lấy URL tháng hiện tại (phần tử cuối cùng trong list)
# Đây chính là logic mà main.py đang dùng
URL = all_urls[-1]

print(f"🔗 Target Data URL: {URL}")

# ==========================================
# 2. CẤU HÌNH MODEL
# ==========================================
MODEL_DIR = "./models/"

def create_tcn(df: pd.DataFrame):
    """
    Hàm khởi tạo cho TCN Model (Deep Learning).
    """
    path_24h = os.path.join(MODEL_DIR, "tcn_GLOBAL_task_24h_global.h5")
    path_7d = os.path.join(MODEL_DIR, "tcn_GLOBAL_task_7d_global.h5")
    path_scaler = os.path.join(MODEL_DIR, "scaler.pkl")

    # Kiểm tra file tồn tại
    if not os.path.exists(path_24h):
        raise FileNotFoundError(f"Missing model file: {path_24h}")
    if not os.path.exists(path_7d):
        raise FileNotFoundError(f"Missing model file: {path_7d}")
    if not os.path.exists(path_scaler):
        raise FileNotFoundError(f"Missing scaler file: {path_scaler}")

    print("📥 Loading TCN Artifacts into RAM...")
    TCNForecastModel.load_artifacts(path_24h, path_7d, path_scaler)
    
    return TCNForecastModel(df)

def create_baseline(df: pd.DataFrame):
    return ForecastModel(df)

def create_mock(df: pd.DataFrame):
    return MockForcastModel(seed=42, base_aqi=80)

MODEL_REGISTRY = {
    "tcn": create_tcn,
    "baseline": create_baseline,
    "mock": create_mock,
}

def get_forecaster(df: pd.DataFrame, model_type: str = "tcn"):
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model_type='{model_type}'")
    
    print(f"🚀 Initializing model type: {model_type.upper()}")
    return MODEL_REGISTRY[model_type](df)

def main():
    print("="*50)
    print("🛠️  TESTING FORECAST LOGIC (DYNAMIC URL)")
    print("="*50)

    # 1. Load dữ liệu từ URL động
    print(f"📥 Downloading CSV data...")
    try:
        df = pd.read_csv(URL)
        print(f"✅ Data loaded successfully. Shape: {df.shape}")
        
        # In thử vài dòng cuối để kiểm tra ngày tháng
        print("\n--- Latest Data Points ---")
        print(df[['timestamp', 'aqi']].tail(3))
    except Exception as e:
        print(f"❌ Failed to load CSV from {URL}")
        print(f"   Error: {e}")
        return

    # 2. Khởi tạo Forecaster
    try:
        forecaster = get_forecaster(df, model_type="tcn")
        
        print("\n" + "-"*30)
        print("🔮 RUNNING PREDICTIONS")
        print("-" * 30)

        # 3. Test dự báo 24h
        print("\n[1] Testing do_forecast_aqi_24h()...")
        forecast_24h = forecaster.do_forecast_aqi_24h()
        
        if forecast_24h:
            print(f"   ✅ Success! Got {len(forecast_24h)} hourly points.")
            print("   SAMPLE (First 12):")
            for item in forecast_24h[:12]:
                print(f"     - {item['timestamp']}: AQI {item['aqi']}")
        else:
            print("   ⚠️ Result is empty.")

        # 4. Test dự báo 7 ngày
        print("\n[2] Testing do_forecast_aqi_7day()...")
        forecast_7d = forecaster.do_forecast_aqi_7day()
        
        if forecast_7d:
            print(f"   ✅ Success! Got {len(forecast_7d)} daily points.")
            print("   SAMPLE:")
            for item in forecast_7d:
                print(f"     - {item['date']}: AQI {item['aqi']}")
        else:
            print("   ⚠️ Result is empty.")

    except FileNotFoundError as e:
        print(f"\n❌ FILE MISSING: {e}")
        print("   👉 Hãy đảm bảo bạn đã copy file .h5 và .pkl vào thư mục ./models/")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()