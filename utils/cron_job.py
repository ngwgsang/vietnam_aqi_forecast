import pandas as pd
import requests
import os
import io
import datetime

class CronJob:
    def __init__(self, current_url, history_urls=None, cache_dir="cache"):
        self.current_url = current_url
        self.history_urls = history_urls or []
        self.cache_dir = cache_dir
        
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    def fetch(self):
        """Lấy dữ liệu hiện tại từ URL"""
        try:
            print(f"Fetching: {self.current_url}")
            response = requests.get(self.current_url)
            if response.status_code == 200:
                df = pd.read_csv(io.StringIO(response.text))
                return df
            else:
                print(f"[WARN] {self.current_url} trả về {response.status_code}")
                return pd.DataFrame()
        except Exception as e:
            print(f"Lỗi fetch: {e}")
            return pd.DataFrame()

    def build_history_csv(self, filename="history.csv"):
        """
        [MODIFIED] Hàm này đã được cập nhật để nhận tham số filename
        Gộp tất cả các file lịch sử lại và lưu vào cache với tên file chỉ định.
        """
        all_dfs = []
        
        # 1. Duyệt qua list URL lịch sử
        for url in self.history_urls:
            try:
                # print(f"   -> Gộp lịch sử: {url}")
                resp = requests.get(url)
                if resp.status_code == 200:
                    temp_df = pd.read_csv(io.StringIO(resp.text))
                    all_dfs.append(temp_df)
                else:
                    # Nếu 404 (thường gặp với city mới chưa có data tháng cũ)
                    # print(f"[WARN] Bỏ qua {url} (404)")
                    pass
            except Exception as e:
                print(f"Lỗi gộp {url}: {e}")

        # 2. Nếu không có dữ liệu nào (City mới tinh hoặc lỗi mạng toàn bộ)
        if not all_dfs:
            print(f"[INFO] Không tìm thấy dữ liệu lịch sử nào hợp lệ. Tạo data giả để không crash app.")
            return self._create_dummy_dataframe(), None

        # 3. Gộp và lưu file
        try:
            final_df = pd.concat(all_dfs, ignore_index=True)
            
            # Xử lý sơ bộ: Xóa trùng lặp thời gian
            if "timestamp" in final_df.columns:
                final_df["timestamp"] = pd.to_datetime(final_df["timestamp"], errors='coerce', utc=True)
                final_df = final_df.dropna(subset=["timestamp"])
                final_df = final_df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")

            save_path = os.path.join(self.cache_dir, filename)
            final_df.to_csv(save_path, index=False)
            return final_df, save_path
            
        except Exception as e:
            print(f"Lỗi khi concat/save history: {e}")
            return None, None

    def _create_dummy_dataframe(self):
        """Tạo DataFrame giả để app không bị crash khi gặp 404"""
        # Tạo 24 giờ dữ liệu giả gần nhất
        end = datetime.datetime.now()
        start = end - datetime.timedelta(hours=24)
        # [FIX] Sửa '1H' thành '1h' để tránh FutureWarning
        idx = pd.date_range(start, end, freq="1h") 
        
        df = pd.DataFrame({
            "timestamp": idx,
            "city": ["Unknown"] * len(idx),
            "aqi": [50] * len(idx), # Mặc định tốt
            "wind_speed": ["0 km/h"] * len(idx),
            "humidity": ["0%"] * len(idx),
            "weather_icon": ["/dl/assets/svg/weather/ic-weather-01d.svg"] * len(idx)
        })
        return df