import os
import json
import joblib
import numpy as np
import pandas as pd
import re
import datetime as dt
from typing import List, Dict
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import random


class ForecastModel:
    """
    Lightweight forecasting helper that infers trends from the
    most recent AQI observations. This is intentionally simple
    so it can run inside the web container without heavy deps.
    """

    def __init__(self, df: pd.DataFrame):
        if df is None or df.empty:
            raise ValueError("ForecastModel expects a non-empty DataFrame.")

        prepared = df.copy()
        prepared["timestamp"] = pd.to_datetime(prepared["timestamp"], errors="coerce")
        prepared["aqi"] = pd.to_numeric(prepared["aqi"], errors="coerce")
        prepared = prepared.dropna(subset=["timestamp", "aqi"]).sort_values("timestamp")

        if prepared.empty:
            raise ValueError("No valid timestamp/aqi rows found for forecasting.")

        self.df = prepared

    def _hourly_trend(self) -> float:
        recent = self.df.tail(min(len(self.df), 48))
        if len(recent) < 2:
            return 0.0
        first, last = recent["aqi"].iloc[0], recent["aqi"].iloc[-1]
        horizon = max(len(recent) - 1, 1)
        return (last - first) / horizon

    def do_forecast_aqi_24h(self) -> List[Dict]:
        """
        Project the next 24 hours using a simple linear trend derived
        from recent observations, smoothed by the last day's average.
        """
        start_time = self.df["timestamp"].iloc[-1]
        recent = self.df.tail(min(len(self.df), 24))
        avg_recent = recent["aqi"].mean()
        slope = self._hourly_trend()
        current = recent["aqi"].iloc[-1]

        forecasts = []
        for i in range(1, 25):
            projected = current + slope * i
            smoothed = 0.6 * avg_recent + 0.4 * projected
            forecasts.append(
                {
                    "timestamp": (start_time + pd.Timedelta(hours=i)).isoformat(),
                    "aqi": max(0, round(smoothed)),
                }
            )
        return forecasts

    def do_forecast_aqi_7day(self) -> List[Dict]:
        """
        Estimate the next 7 days using daily averages and a mild trend.
        """
        daily = (
            self.df.set_index("timestamp")
            .resample("D")["aqi"]
            .mean()
            .dropna()
        )

        if daily.empty:
            raise ValueError("Not enough daily data to forecast.")

        tail = daily.tail(min(len(daily), 7))
        baseline = tail.mean()
        trend = 0.0
        if len(tail) > 1:
            trend = (tail.iloc[-1] - tail.iloc[0]) / max(len(tail) - 1, 1)

        start_day = daily.index[-1].normalize()
        forecasts = []
        for i in range(1, 8):
            value = baseline + trend * i
            forecasts.append(
                {
                    "date": (start_day + pd.Timedelta(days=i)).date().isoformat(),
                    "aqi": max(0, round(value)),
                }
            )
        return forecasts


class MockForcastModel:
    """
    Quick mock that generates deterministic random values for demos or tests.
    """

    def __init__(self, seed: int = 42, base_aqi: int = 80):
        self.rng = random.Random(seed)
        self.base_aqi = base_aqi

    def _noise(self, scale: int = 20) -> int:
        return self.rng.randint(-scale, scale)

    def do_forecast_aqi_7day(self) -> List[Dict]:
        today = dt.date.today()
        return [
            {
                "date": (today + dt.timedelta(days=i)).isoformat(),
                "aqi": max(0, self.base_aqi + self._noise(25)),
            }
            for i in range(1, 8)
        ]
    
    def do_forecast_aqi_24h(self) -> List[Dict]:
        now = dt.datetime.now()
        return [
            {
                "timestamp": (now + dt.timedelta(hours=i)).isoformat(),
                "aqi": max(0, self.base_aqi + self._noise(15)),
            }
            for i in range(1, 25)
        ]

class TCNForecastModel:
    """
    Advanced forecasting helper that uses pre-trained TCN models (Deep Learning).
    
    Architecture:
    - Uses Class-level attributes to cache models (load once, predict many times).
    - Re-implements feature engineering pipeline to match training data.
    - Performs recursive multi-step forecasting.
    """

    # --- GLOBAL CACHE (Tránh load lại model mỗi lần init) ---
    _model_24h = None
    _model_7d = None
    _scaler = None
    
    # Định nghĩa thứ tự feature bắt buộc (phải khớp 100% với lúc train)
    INPUT_FEATURES = [
        'year', 'month', 'day', 'hour', 'day_of_week', 'is_weekend', 
        'time_of_day', 'wind_speed_value', 'humidity_ratio', 'weather_type', 
        'is_peak_hour', 'season', 'day_progress', 'humidity_level', 
        'wind_category', 'aqi'
    ]
    
    # Mapping đơn giản cho các cột Category (Thay vì phải load LabelEncoder)
    # Lưu ý: Cần khớp với logic LabelEncoder lúc train. 
    # Tốt nhất là save LabelEncoder ra file .pkl, nhưng ở đây mình hardcode demo.
    CAT_MAPPING = {
        'season': {'Spring': 0, 'Summer': 1, 'Autumn': 2, 'Winter': 3},
        'time_of_day': {'Night': 0, 'Morning': 1, 'Afternoon': 2, 'Evening': 3, 'Late Night': 4},
        'weather_type': {'Clear': 0, 'Cloudy': 1, 'Rain': 2}, # Cần map đầy đủ theo data của bạn
        'humidity_level': {'Low': 0, 'Medium': 1, 'High': 2},
        'wind_category': {'Calm': 0, 'Breezy': 1, 'Windy': 2, 'Strong': 3}
    }

    @classmethod
    def load_artifacts(cls, path_24h: str, path_7d: str, scaler_path: str):
        """
        Call this method ONCE at the application startup.
        """
        if cls._model_24h is None:
            print(f"📥 Loading TCN Models & Scaler...")
            cls._model_24h = load_model(path_24h)
            cls._model_7d = load_model(path_7d)
            cls._scaler = joblib.load(scaler_path)
            print("✅ Artifacts loaded successfully.")

    def __init__(self, df: pd.DataFrame, city_name: str = 'ho-chi-minh'):
        """
        Args:
            df: DataFrame containing at least 'timestamp' and 'aqi'.
            city_name: Used for potential city-specific encoding if needed.
        """
        if self._model_24h is None:
            raise RuntimeError("Models not loaded! Call TCNForecastModel.load_artifacts() first.")

        if df is None or df.empty:
            raise ValueError("ForecastModel expects a non-empty DataFrame.")

        self.city_name = city_name
        self.raw_df = self._clean_input_df(df)
        
        # Tạo features đầy đủ từ raw data
        self.processed_df = self._feature_engineering(self.raw_df)

    def _clean_input_df(self, df: pd.DataFrame) -> pd.DataFrame:
        prepared = df.copy()
        prepared["timestamp"] = pd.to_datetime(prepared["timestamp"], errors="coerce")
        prepared["aqi"] = pd.to_numeric(prepared["aqi"], errors="coerce")
        # Fill các cột thiếu nếu cần thiết cho feature engineering
        if "wind_speed" not in prepared.columns: prepared["wind_speed"] = "0 km/h"
        if "humidity" not in prepared.columns: prepared["humidity"] = "80%"
        if "weather_icon" not in prepared.columns: prepared["weather_icon"] = "unknown"
        
        prepared = prepared.dropna(subset=["timestamp", "aqi"]).sort_values("timestamp")
        return prepared

    def _feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Tái tạo lại logic get_aug_df từ notebook"""
        df = df.copy()
        
        # 1. Time Features
        df["year"] = df["timestamp"].dt.year
        df["month"] = df["timestamp"].dt.month
        df["day"] = df["timestamp"].dt.day
        df["hour"] = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
        df["day_progress"] = df["hour"] / 24.0
        
        # 2. Advanced Time Features (Manual Mapping)
        # Time of Day
        bins = [-1, 5, 11, 17, 21, 24]
        labels = ["Late Night", "Morning", "Afternoon", "Evening", "Night"]
        # Lưu ý: pd.cut trả về category, cần map sang số ngay
        time_labels = pd.cut(df["hour"], bins=bins, labels=labels)
        df["time_of_day"] = time_labels.map(lambda x: self.CAT_MAPPING['time_of_day'].get(x, 0)).astype(int)

        # Season
        def get_season(month):
            if month in [2, 3, 4]: return "Spring"
            elif month in [5, 6, 7]: return "Summer"
            elif month in [8, 9, 10]: return "Autumn"
            else: return "Winter"
        df["season"] = df["month"].apply(get_season).map(lambda x: self.CAT_MAPPING['season'].get(x, 0)).astype(int)

        df["is_peak_hour"] = (df["hour"].between(6, 9) | df["hour"].between(17, 20)).astype(int)

        # 3. Weather Features (Xử lý chuỗi -> số)
        # Wind Speed
        df["wind_speed_value"] = (
            df["wind_speed"].astype(str)
            .apply(lambda x: re.findall(r"[\d.]+", x))
            .apply(lambda x: float(x[0]) if x else 0.0)
        )
        # Wind Category
        df["wind_category"] = pd.cut(
            df["wind_speed_value"], 
            bins=[-0.1, 5, 15, 25, np.inf], 
            labels=["Calm", "Breezy", "Windy", "Strong"]
        ).map(lambda x: self.CAT_MAPPING['wind_category'].get(x, 0)).astype(int)

        # Humidity
        df["humidity_ratio"] = (
            df["humidity"].astype(str).str.replace("%", "", regex=False).astype(float) / 100
        )
        df["humidity_level"] = pd.cut(
            df["humidity_ratio"], 
            bins=[-0.01, 0.4, 0.7, 1.0], 
            labels=["Low", "Medium", "High"]
        ).map(lambda x: self.CAT_MAPPING['humidity_level'].get(x, 0)).astype(int)

        # Weather Type (Placeholder - map đơn giản để tránh lỗi)
        # Trong thực tế bạn nên save LabelEncoder của weather_type
        df["weather_type"] = 0 

        # Đảm bảo đúng thứ tự cột
        return df[self.INPUT_FEATURES]

    def _predict_recursive(self, model, current_window, steps) -> List[float]:
        """Chạy dự báo cuộn (Recursive)"""
        predictions = []
        curr_seq = current_window.copy() # (window_size, n_features)
        
        target_idx = self.INPUT_FEATURES.index('aqi')

        for _ in range(steps):
            # Reshape for model (1, window, features)
            input_data = curr_seq.reshape(1, curr_seq.shape[0], curr_seq.shape[1])
            
            # Predict (scaled value)
            pred_scaled = model.predict(input_data, verbose=0)[0, 0]
            predictions.append(pred_scaled)
            
            # Update sequence
            next_row = curr_seq[-1].copy()
            next_row[target_idx] = pred_scaled
            # (Optional: Cập nhật giờ/ngày cho next_row tại đây để chính xác hơn)
            
            curr_seq = np.vstack([curr_seq[1:], next_row])
            
        return predictions

    def _inverse_transform_result(self, scaled_preds: List[float]) -> List[int]:
        """Chuyển đổi kết quả scaled về AQI thực tế"""
        target_idx = self.INPUT_FEATURES.index('aqi')
        dummy = np.zeros((len(scaled_preds), len(self.INPUT_FEATURES)))
        dummy[:, target_idx] = scaled_preds
        
        real_values = self._scaler.inverse_transform(dummy)[:, target_idx]
        return [max(0, int(round(val))) for val in real_values]

    def do_forecast_aqi_24h(self) -> List[Dict]:
        """Dự báo 24 giờ tới bằng Model TCN 24h"""
        window_size = 48
        
        # 1. Scale dữ liệu đầu vào
        data_values = self.processed_df.values
        if len(data_values) < window_size:
            # Fallback nếu không đủ dữ liệu: Trả về lỗi hoặc dùng heuristic cũ
             raise ValueError(f"Not enough data for TCN (Need {window_size} hours)")

        scaled_data = self._scaler.transform(data_values)
        last_window = scaled_data[-window_size:]

        # 2. Predict
        preds_scaled = self._predict_recursive(self._model_24h, last_window, steps=24)
        
        # 3. Inverse & Format
        preds_real = self._inverse_transform_result(preds_scaled)
        
        start_time = self.raw_df["timestamp"].iloc[-1]
        forecasts = []
        for i, val in enumerate(preds_real):
            forecasts.append({
                "timestamp": (start_time + pd.Timedelta(hours=i+1)).isoformat(),
                "aqi": val
            })
            
        return forecasts

    def do_forecast_aqi_7day(self) -> List[Dict]:
        """Dự báo 7 ngày tới bằng Model TCN 7d"""
        window_size = 168 # 1 tuần
        steps_hours = 168 # 7 ngày * 24h
        
        # 1. Scale
        data_values = self.processed_df.values
        if len(data_values) < window_size:
             # Fallback: có thể padding hoặc raise error
             raise ValueError(f"Not enough data for TCN (Need {window_size} hours)")

        scaled_data = self._scaler.transform(data_values)
        last_window = scaled_data[-window_size:]

        # 2. Predict (Dự báo từng giờ cho 7 ngày)
        preds_scaled = self._predict_recursive(self._model_7d, last_window, steps=steps_hours)
        preds_real = self._inverse_transform_result(preds_scaled)

        # 3. Aggregate thành Daily (Trung bình ngày)
        start_time = self.raw_df["timestamp"].iloc[-1]
        
        # Tạo DataFrame tạm để resample theo ngày
        future_dates = [start_time + pd.Timedelta(hours=i+1) for i in range(steps_hours)]
        df_forecast = pd.DataFrame({'timestamp': future_dates, 'aqi': preds_real})
        
        # Group by Date lấy trung bình
        daily_forecast = df_forecast.resample('D', on='timestamp')['aqi'].mean().round().astype(int)
        
        forecasts = []
        for date, aqi in daily_forecast.items():
            # Bỏ qua ngày hiện tại nếu resample tính cả phần còn lại của hôm nay
            if date.date() <= start_time.date():
                continue
            forecasts.append({
                "date": date.date().isoformat(),
                "aqi": aqi
            })
            
        return forecasts