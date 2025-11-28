from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
import datetime
import re
import os
import random  # [NEW] Import random
from apscheduler.schedulers.background import BackgroundScheduler
from contextlib import asynccontextmanager

from utils.cron_job import CronJob
from utils.model import ForecastModel, MockForcastModel, TCNForecastModel


APP_SEED = random.randint(0, 1000000)
print(f"🎲 App Seed initialized: {APP_SEED}")

# ==============================================================================
# CẤU HÌNH THÀNH PHỐ
# ==============================================================================
SUPPORTED_CITIES = {
    "ho-chi-minh-city": "Hồ Chí Minh",
    "hanoi": "Hà Nội",
    "can-tho": "Cần Thơ",
    "nha-trang": "Nha Trang",
    "hue": "Huế",
    "vinh": "Vinh"
}

# Template URL chung
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
# GLOBAL DATA STORE
# ==============================================================================
global_data = {}

# Khởi tạo khung chứa dữ liệu
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
    """Hàm dịch thời tiết chuẩn"""
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
    """Tạo URL cho 1 city cụ thể"""
    urls = []
    today = datetime.date.today()
    for m in range(1, today.month + 1):
        month_slug = MONTH_SLUGS[m - 1]
        url = BASE_URL_TEMPLATE.format(city=city_slug, year=year, month=month_slug)
        urls.append(url)
    return urls


def process_city_data(city_slug: str, scope: str = "current"):
    """
    Worker xử lý dữ liệu cho 1 thành phố cụ thể.
    """
    target_city_name = SUPPORTED_CITIES.get(city_slug, "Unknown")
    print(f"🔄 [UPDATE] Đang xử lý: {target_city_name} ({scope})")
    
    # 1. Tạo URL và CronJob
    urls = build_urls_for_city(city_slug, 2025)
    cron_job = CronJob(urls[-1], history_urls=urls)
    
    try:
        # 2. Fetch data
        df_current = cron_job.fetch()
        
        history_df = None
        if scope == "history":
            history_df, _ = cron_job.build_history_csv(filename=f"history_{city_slug}.csv")
        
        df_source = history_df if history_df is not None else df_current
        
        if df_source is None or df_source.empty:
            print(f"⚠️ [WARN] Không có dữ liệu cho {city_slug}")
            return

        # 3. Xử lý DataFrame
        df_source["timestamp"] = pd.to_datetime(df_source["timestamp"].astype(str), utc=True, errors="coerce")
        df_source["timestamp"] = df_source["timestamp"].dt.tz_convert("Asia/Ho_Chi_Minh").dt.tz_localize(None)
        
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
                "pollutants": {}
            }
        
        global_data[city_slug]["current"] = current_info
        
        # 5. Xử lý History & Heatmap
        if scope == "history":
            global_data[city_slug]["history"] = valid_rows.to_dict("records")
            daily = valid_rows.assign(date=valid_rows["timestamp"].dt.date).groupby("date")["aqi"].mean().reset_index()
            global_data[city_slug]["heatmap_daily"] = daily.rename(columns={"aqi": "avg_aqi"}).to_dict("records")

        # 6. Forecast & Random Noise Application
        clean_df = valid_rows
        model = None
        
        if not clean_df.empty:
            try:
                # Fallback về Baseline cho an toàn với các city chưa có data train
                model = ForecastModel(clean_df) 
            except Exception:
                model = MockForcastModel()
            
            # Lấy kết quả dự báo gốc
            raw_forecast_24h = model.do_forecast_aqi_24h()
            raw_forecast_7d = model.do_forecast_aqi_7day()

            # ==================================================================
            # [NEW] LOGIC THÊM NHIỄU RANDOM [-6, +6] CỐ ĐỊNH THEO SESSION
            # ==================================================================
            # Tạo bộ sinh số ngẫu nhiên riêng biệt cho thành phố này
            # Seed = APP_SEED (cố định lúc start app) + Tên thành phố
            # -> Đảm bảo mỗi lần vào lại thành phố này, dãy số random vẫn y hệt
            
            # 1. Xử lý cho 24h
            rng_24h = random.Random(f"{APP_SEED}_{city_slug}_24h")
            for item in raw_forecast_24h:
                if "aqi" in item:
                    noise = rng_24h.randint(-6, 6)
                    # Cộng nhiễu, đảm bảo không âm
                    item["aqi"] = max(0, item["aqi"] + noise)

            # 2. Xử lý cho 7 ngày
            rng_7d = random.Random(f"{APP_SEED}_{city_slug}_7d")
            for item in raw_forecast_7d:
                if "aqi" in item:
                    noise = rng_7d.randint(-6, 6)
                    item["aqi"] = max(0, item["aqi"] + noise)

            # Lưu vào global data
            global_data[city_slug]["forecast_24h"] = raw_forecast_24h
            global_data[city_slug]["forecast_7d"] = raw_forecast_7d
        
        global_data[city_slug]["last_updated"] = datetime.datetime.now()

    except Exception as e:
        print(f"❌ [ERROR] Lỗi update {city_slug}: {e}")


def update_all_cities(scope="current"):
    """Chạy vòng lặp qua tất cả city"""
    for slug in SUPPORTED_CITIES.keys():
        process_city_data(slug, scope)


# --- SCHEDULER & LIFESPAN ---
scheduler = BackgroundScheduler()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load Model AI
    if os.path.exists(PATH_TCN_24H) and os.path.exists(PATH_TCN_7D):
        try:
            print("📥 Đang load TCN Model...")
            TCNForecastModel.load_artifacts(PATH_TCN_24H, PATH_TCN_7D, PATH_SCALER)
            print("✅ Load model thành công.")
        except Exception as e:
            print(f"❌ Lỗi load model: {e}")

    print("🚀 Khởi động: Đang tải dữ liệu lịch sử cho tất cả thành phố...")
    update_all_cities(scope="history")

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
    # Validate city
    if city not in global_data:
        return {"status": "error", "message": f"City '{city}' not supported"}
    
    data = global_data[city]
    
    if not data["current"]:
         return {"status": "loading", "message": "Đang tải dữ liệu..."}

    return {
        "current": data["current"],
        "forecast_24h": data["forecast_24h"],
        "forecast_7d": data["forecast_7d"],
        "heatmap_daily": data["heatmap_daily"],
        "last_updated": data["last_updated"],
    }