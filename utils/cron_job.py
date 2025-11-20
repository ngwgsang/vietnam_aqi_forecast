# utils/cron_job.py
import pandas as pd
import datetime
import requests

MONTH_SLUGS = [
    "jan", "feb", "mar", "apr", "may", "jun",
    "jul", "aug", "sep", "oct", "nov", "dec"
]
MONTH_MAP = {slug: i + 1 for i, slug in enumerate(MONTH_SLUGS)}

TARGET_CITY = "Hồ Chí Minh"  # hoặc import từ config nếu muốn

class CronJob:
    def __init__(self, url, history_urls=None):
        self.url = url
        self.history_urls = history_urls or []

    # --- Hàm đọc CSV an toàn ---
    def _safe_read_csv(self, url: str) -> pd.DataFrame:
        try:
            # Kiểm tra tồn tại nhanh (HEAD)
            resp = requests.head(url, timeout=5)
            if resp.status_code != 200:
                print(f"[WARN] {url} trả về {resp.status_code}. Sinh DataFrame giả.")
                return self._empty_month_df_from_url(url)

            return pd.read_csv(url)
        except Exception as e:
            print(f"[WARN] Không đọc được {url}: {e}. Sinh DataFrame giả.")
            return self._empty_month_df_from_url(url)

    # --- Sinh DataFrame giả nếu không có CSV ---
    def _empty_month_df_from_url(self, url: str) -> pd.DataFrame:
        """
        Parse url dạng: .../aqi_ho-chi-minh-city_2025_may.csv
        → tạo hourly timestamp trong tháng đó, aqi = None
        """
        fname = url.split("/")[-1]  # aqi_ho-chi-minh-city_2025_may.csv
        base = fname.replace(".csv", "")
        parts = base.split("_")     # ['aqi', 'ho-chi-minh-city', '2025', 'may']

        if len(parts) < 4:
            # fallback: DataFrame rỗng nhưng đúng schema
            return pd.DataFrame(columns=["city", "timestamp", "aqi"])

        year = int(parts[-2])
        month_slug = parts[-1]
        month = MONTH_MAP.get(month_slug)

        if month is None:
            return pd.DataFrame(columns=["city", "timestamp", "aqi"])

        start = datetime.datetime(year, month, 1)
        # tính ngày đầu tháng tiếp theo
        if month == 12:
            end = datetime.datetime(year + 1, 1, 1)
        else:
            end = datetime.datetime(year, month + 1, 1)

        # tạo timestamp theo giờ
        idx = pd.date_range(start, end - datetime.timedelta(hours=1), freq="1H")

        df = pd.DataFrame({
            "city": [TARGET_CITY] * len(idx),
            "timestamp": idx,
            "aqi": [None] * len(idx),  # aqi = None như yêu cầu
        })
        return df

    def fetch(self) -> pd.DataFrame:
        return self._safe_read_csv(self.url)

    def build_history_csv(self):
        dfs = []

        for url in self.history_urls:
            df = self._safe_read_csv(url)
            # vẫn append cả DataFrame “giả” (aqi=None) để giữ timeline đầy đủ
            if df is not None and not df.empty:
                dfs.append(df)

        if not dfs:
            print("[HISTORY] Không có lịch sử (tất cả file đều lỗi).")
            return None, None

        history_df = pd.concat(dfs, ignore_index=True)

        history_path = "cache/history.csv"
        history_df.to_csv(history_path, index=False)

        return history_df, history_path
