from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
import datetime
import re
import os
from apscheduler.schedulers.background import BackgroundScheduler
from contextlib import asynccontextmanager

from utils.cron_job import CronJob
from utils.model import ForecastModel, MockForcastModel, TCNForecastModel

# ==============================================================================
# [NEW] CẤU HÌNH THÀNH PHỐ
# ==============================================================================
SUPPORTED_CITIES = {
    "ho-chi-minh-city": "Hồ Chí Minh",
    "hanoi": "Hà Nội",
    "can-tho": "Cần Thơ",      # [NEW]
    "nha-trang": "Nha Trang",  # [NEW]
    "hue": "Huế",              # [NEW]
    "vinh": "Vinh"             # [NEW]
}

# Template URL chung, {city} sẽ được thay thế
BASE_URL_TEMPLATE = (
    "https://raw.githubusercontent.com/HiAmNear/iqair-crawling"
    "/refs/heads/main/result/{city}/aqi_{city}_{year}_{month}.csv"
)

# Đường dẫn model
MODEL_DIR = "./models"
PATH_TCN_24H = os.path.join(MODEL_DIR, "tcn_GLOBAL_task_24h_global.h5")
PATH_TCN_7D = os.path.join(MODEL_DIR, "tcn_GLOBAL_task_7d_global.h5")
PATH_SCALER = os.path.join(MODEL_DIR, "scaler.pkl")

MONTH_SLUGS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]

# ==============================================================================
# [MODIFIED] GLOBAL DATA STORE (Lưu theo từng city)
# ==============================================================================
# Cấu trúc: global_data["hanoi"] = { "current": ..., "history": ... }
global_data = {}

# Khởi tạo khung chứa dữ liệu cho tất cả thành phố
for slug in SUPPORTED_CITIES.keys():
    global_data[slug] = {
        "current": {},
        "history": [],
        "forecast_24h": [],
        "forecast_7d": [],
        "heatmap_daily": [],
        "last_updated": None
    }


def extract_weather_vietnamese(icon_str):
    """[MODIFIED] Hàm dịch thời tiết chuẩn (đã sửa ở bước trước)"""
    match = re.search(r"ic-weather-(\d{2})[dn]", str(icon_str))
    if match:
        code_id = match.group(1)
        weather_map = {
            "01": "Trời quang",
            "02": "Ít mây",
            "03": "Mây rải rác",
            "04": "Nhiều mây",
            "09": "Mưa rào",
            "10": "Trời mưa",
            "11": "Giông bão",
            "13": "Tuyết rơi",
            "50": "Sương mù"
        }
        return weather_map.get(code_id, "Không xác định")
    return "Không xác định"


def build_urls_for_city(city_slug: str, year: int):
    """[MODIFIED] Tạo URL cho 1 city cụ thể"""
    urls = []
    today = datetime.date.today()
    for m in range(1, today.month + 1):
        month_slug = MONTH_SLUGS[m - 1]
        url = BASE_URL_TEMPLATE.format(city=city_slug, year=year, month=month_slug)
        urls.append(url)
    return urls


def process_city_data(city_slug: str, scope: str = "current"):
    """
    [MODIFIED] Worker xử lý dữ liệu cho 1 thành phố cụ thể.
    """
    target_city_name = SUPPORTED_CITIES.get(city_slug, "Unknown")
    print(f"🔄 [UPDATE] Đang xử lý: {target_city_name} ({scope})")
    
    # 1. Tạo URL và CronJob cho city này
    urls = build_urls_for_city(city_slug, 2025)
    cron_job = CronJob(urls[-1], history_urls=urls)
    
    try:
        # 2. Fetch data
        df_current = cron_job.fetch()
        
        history_df = None
        if scope == "history":
            # Lưu file riêng cho từng city để tránh ghi đè
            history_df, _ = cron_job.build_history_csv(filename=f"history_{city_slug}.csv")
        
        df_source = history_df if history_df is not None else df_current
        
        if df_source is None or df_source.empty:
            print(f"⚠️ [WARN] Không có dữ liệu cho {city_slug}")
            return

        # 3. Lọc dữ liệu theo tên thành phố (Relative filter)
        # Lưu ý: Data CSV cột 'city' có thể là 'Ho Chi Minh City' hoặc 'Hồ Chí Minh'
        # Ta lọc lỏng lẻo để bắt được dữ liệu
        # Nếu file CSV chỉ chứa 1 city thì lấy hết cũng được
        
        # Chuẩn hoá timestamp
        df_source["timestamp"] = pd.to_datetime(df_source["timestamp"].astype(str), utc=True, errors="coerce")
        df_source["timestamp"] = df_source["timestamp"].dt.tz_convert("Asia/Ho_Chi_Minh").dt.tz_localize(None)
        
        # Sắp xếp và lấy dòng có AQI
        valid_rows = df_source.dropna(subset=["aqi", "timestamp"]).sort_values("timestamp")
        
        # 4. Xử lý Current Info
        current_info = {}
        if not valid_rows.empty:
            latest = valid_rows.iloc[-1]
            aqi = int(latest["aqi"])
            
            current_info = {
                "location": f"{target_city_name}, Vietnam",
                "aqi": aqi,
                "windspeed": latest.get("wind_speed", "--"),
                "humidity": latest.get("humidity", "--"),
                "weather": extract_weather_vietnamese(latest.get("weather_icon", "")),
                "status": "Kém" if aqi > 100 else "Tốt",
                "updated": latest["timestamp"].strftime("%H:%M %d/%m"),
                "pollutants": {} # Nếu có data chi tiết thì map vào đây
            }
        
        global_data[city_slug]["current"] = current_info
        
        # 5. Xử lý History & Heatmap
        if scope == "history":
            global_data[city_slug]["history"] = valid_rows.to_dict("records")
            daily = valid_rows.assign(date=valid_rows["timestamp"].dt.date).groupby("date")["aqi"].mean().reset_index()
            global_data[city_slug]["heatmap_daily"] = daily.rename(columns={"aqi": "avg_aqi"}).to_dict("records")

        # 6. Forecast
        # [NOTE] Chỉ HCM mới dùng TCN (nếu model train cho HCM), các city khác dùng Baseline để tránh lỗi
        clean_df = valid_rows
        model = None
        
        if not clean_df.empty:
            try:
                # Nếu muốn dùng TCN cho mọi nơi (cần retrain hoặc chấp nhận sai số):
                # model = TCNForecastModel(clean_df)
                
                # Hiện tại fallback về Baseline cho an toàn
                model = ForecastModel(clean_df) 
            except Exception:
                model = MockForcastModel()
            
            global_data[city_slug]["forecast_24h"] = model.do_forecast_aqi_24h()
            global_data[city_slug]["forecast_7d"] = model.do_forecast_aqi_7day()
        
        global_data[city_slug]["last_updated"] = datetime.datetime.now()

    except Exception as e:
        print(f"❌ [ERROR] Lỗi update {city_slug}: {e}")


def update_all_cities(scope="current"):
    """[NEW] Chạy vòng lặp qua tất cả city"""
    for slug in SUPPORTED_CITIES.keys():
        process_city_data(slug, scope)


# --- SCHEDULER & LIFESPAN ---
scheduler = BackgroundScheduler()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load Model AI (Giữ nguyên)
    if os.path.exists(PATH_TCN_24H) and os.path.exists(PATH_TCN_7D):
        try:
            print("📥 Đang load TCN Model...")
            TCNForecastModel.load_artifacts(PATH_TCN_24H, PATH_TCN_7D, PATH_SCALER)
            print("✅ Load model thành công.")
        except Exception as e:
            print(f"❌ Lỗi load model: {e}")

    # [MODIFIED] Update history cho TẤT CẢ city khi khởi động
    print("🚀 Khởi động: Đang tải dữ liệu lịch sử cho tất cả thành phố...")
    update_all_cities(scope="history")

    # [MODIFIED] Schedule job loop qua tất cả city
    scheduler.add_job(update_all_cities, "interval", hours=1, args=["current"], id="hourly_update")
    scheduler.start()
    yield
    scheduler.shutdown()

app = FastAPI(lifespan=lifespan)
templates = Jinja2Templates(directory="templates")

# --- ROUTES ---
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/air-quality")
async def get_air_quality(city: str = "ho-chi-minh-city"):
    """
    [MODIFIED] API nhận tham số city (mặc định là HCM)
    Ví dụ: /api/air-quality?city=hanoi
    """
    # Validate city
    if city not in global_data:
        # Fallback về default hoặc báo lỗi
        return {"status": "error", "message": f"City '{city}' not supported"}
    
    data = global_data[city]
    
    # Nếu chưa có dữ liệu current (đang load lần đầu)
    if not data["current"]:
         return {"status": "loading", "message": "Đang tải dữ liệu..."}

    return {
        "current": data["current"],
        "forecast_24h": data["forecast_24h"],
        "forecast_7d": data["forecast_7d"],
        "heatmap_daily": data["heatmap_daily"],
        "last_updated": data["last_updated"],
    }