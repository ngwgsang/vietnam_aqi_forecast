from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
import datetime

from apscheduler.schedulers.background import BackgroundScheduler
from contextlib import asynccontextmanager

from utils.cron_job import CronJob
from utils.model import ForecastModel, MockForcastModel


# ---- CONFIG ----
CITY_SLUG = "ho-chi-minh-city"
BASE_URL = (
    "https://raw.githubusercontent.com/nghiahsgs/iqair-dataset"
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


# ---- DÙNG HÀM Ở TRÊN ĐỂ TẠO URL ----
YEAR = 2025

all_urls = build_month_urls(CITY_SLUG, YEAR)

# URL tháng hiện tại (cuối danh sách)
CSV_URL = all_urls[-1]

# Các URL lịch sử (từ Jan đến trước tháng hiện tại)
HISTORY_CSV_URLS = all_urls[:-1]

TARGET_CITY = "Hồ Chí Minh"  # vẫn giữ như cũ

# Biến toàn cục lưu trữ dữ liệu trong RAM
global_data = {
    "current": {},
    "history": [],
    "forecast_24h": [],
    "forecast_7d": [],
    "heatmap_daily": [],
    "last_updated": None
}

cron_job = CronJob(CSV_URL, history_urls=HISTORY_CSV_URLS)


def fetch_and_process_data(scope: str = "current"):
    """
    Hàm worker để tải và xử lý CSV.
    scope='history': Lấy toàn bộ và gộp các tháng cũ.
    scope='current': Lấy dữ liệu mới nhất.
    """
    try:
        # 1. Lấy dataset hiện tại (sẽ refresh mỗi giờ)
        df_current = cron_job.fetch()

        # 2. Nếu cần lịch sử thì gộp tất cả tháng và ghi ra cache/history.csv
        history_df = None
        if scope == "history":
            history_df, history_path = cron_job.build_history_csv()
            if history_df is not None:
                print(f"[HISTORY] Đã gộp và lưu tại {history_path}")

        # 3. Chọn dataframe nguồn cho các bước tiếp theo
        df_source = history_df if history_df is not None else df_current

        if df_source is None or df_source.empty:
            print("[WARN] df_source rỗng, không có dữ liệu nào.")
            return

        # 4. Lọc dữ liệu TP.HCM và chuẩn hoá timestamp
        hcm_df = df_source[df_source["city"] == TARGET_CITY].copy()

        # Ép hết sang string rồi parse về UTC để tránh mixed tz
        hcm_df["timestamp"] = pd.to_datetime(
            hcm_df["timestamp"].astype(str),
            utc=True,
            errors="coerce",
        )
        # Chuyển về giờ Việt Nam rồi bỏ timezone (tz-naive, thống nhất)
        hcm_df["timestamp"] = (
            hcm_df["timestamp"]
            .dt.tz_convert("Asia/Ho_Chi_Minh")
            .dt.tz_localize(None)
        )

        hcm_df = hcm_df.dropna(subset=["timestamp"]).sort_values("timestamp")

        # 5. Lấy bản ghi mới nhất có aqi hợp lệ (non-NaN)
        valid_rows = hcm_df.dropna(subset=["aqi"])
        if not valid_rows.empty:
            latest_valid = valid_rows.iloc[-1]
            latest_aqi = latest_valid["aqi"]
            aqi_value = int(latest_aqi)
            status = "Kém" if aqi_value > 100 else "Tốt"
            updated_time = latest_valid["timestamp"].strftime("%H:%M %d/%m")
        else:
            # Không có aqi thật, chỉ có timestamp giả
            latest_any = hcm_df.iloc[-1]
            aqi_value = None
            status = "Không có dữ liệu"
            updated_time = latest_any["timestamp"].strftime("%H:%M %d/%m")

        global_data["current"] = {
            "location": f"{TARGET_CITY}, Vietnam",
            "aqi": aqi_value,
            "status": status,
            "updated": updated_time,
        }

        # 6. Lưu lịch sử (sử dụng full history nếu có, nếu không dùng hiện tại)
        if scope == "history":
            global_data["history"] = hcm_df.to_dict("records")
            # Trung bình theo ngày để vẽ heatmap, chỉ dùng aqi thật
            daily = (
                valid_rows.assign(date=valid_rows["timestamp"].dt.date)
                .groupby("date")["aqi"]
                .mean()
                .dropna()
                .reset_index()
                .rename(columns={"aqi": "avg_aqi"})
                .sort_values("date")
            )
            global_data["heatmap_daily"] = daily.to_dict("records")

        # 7. Forecasts: dùng dữ liệu đã làm sạch (chỉ non-NaN)
        try:
            clean_df = valid_rows
            if clean_df.empty:
                raise ValueError("Không có giá trị AQI hợp lệ để train model.")
            model = ForecastModel(clean_df)
        except Exception as e:
            print(f"[FORECAST] Lỗi khi khởi tạo ForecastModel: {e}")
            model = MockForcastModel()

        try:
            global_data["forecast_24h"] = model.do_forecast_aqi_24h()
        except Exception as e:
            print(f"[FORECAST] Lỗi khi dự báo 24h: {e}")
            global_data["forecast_24h"] = []

        try:
            global_data["forecast_7d"] = model.do_forecast_aqi_7day()
        except Exception as e:
            print(f"[FORECAST] Lỗi khi dự báo 7 ngày: {e}")
            global_data["forecast_7d"] = []

        global_data["last_updated"] = datetime.datetime.now()
        print(f"[{scope.upper()}] Đã cập nhật dữ liệu lúc {global_data['last_updated']}")

    except Exception as e:
        print(f"Lỗi khi fetch data: {e}")



# --- SCHEDULER SETUP ---
scheduler = BackgroundScheduler()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Chạy ngay khi app khởi động (Lấy lịch sử + hiện tại)
    fetch_and_process_data(scope="history")

    # 2. Thêm job: Chạy mỗi tiếng một lần (interval)
    scheduler.add_job(
        fetch_and_process_data,
        "interval",
        hours=1,
        args=["current"],
        id="hourly_update",
    )

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
async def get_air_quality():
    # Trả về dữ liệu thực tế từ biến toàn cục thay vì hardcode
    if not global_data["current"]:
        return {"status": "loading", "message": "Đang tải dữ liệu..."}

    repsonse = {
        "current": global_data["current"],
        "forecast_24h": global_data["forecast_24h"],
        "forecast_7d": global_data["forecast_7d"],
        "heatmap_daily": global_data["heatmap_daily"],
        "last_updated": global_data["last_updated"],
    }
    print(repsonse)
    return repsonse
