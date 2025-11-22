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

class LSTMForecastModel:
    """
    Forecasting helper using a pre-trained LSTM model (.h5).
    It loads model artifacts and performs recursive prediction.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        model_dir: str,
        artifacts: Dict[str, str] | None = None,
    ):
        """
        Args:
            df: Raw dataframe (input data).
            model_dir: Path to the folder containing model artifacts.
            artifacts: Dict map tên logic -> tên file thực tế, ví dụ:
                {
                    "model": "best_aqi_lstm.h5",
                    "scaler_X": "scaler_X.pkl",
                    "scaler_y": "scaler_y.pkl",
                    "config": "model_config.json"
                }
            Nếu không truyền thì dùng default như hiện tại.
        """
        if df is None or df.empty:
            raise ValueError("LSTMForecastModel expects a non-empty DataFrame.")

        self.model_dir = model_dir

        # ---- NEW: cho phép cấu hình tên file model / scaler / config ----
        if artifacts is None:
            artifacts = {
                "model": "best_aqi_lstm.h5",
                "scaler_X": "scaler_X.pkl",
                "scaler_y": "scaler_y.pkl",
                "config": "model_config.json",
            }
        self.artifacts = artifacts

        # 1. Load Artifacts
        try:
            model_path = os.path.join(model_dir, artifacts["model"])
            scaler_X_path = os.path.join(model_dir, artifacts["scaler_X"])
            scaler_y_path = os.path.join(model_dir, artifacts["scaler_y"])
            config_path = os.path.join(model_dir, artifacts["config"])

            self.model = load_model(model_path, compile=False)
            self.scaler_X = joblib.load(scaler_X_path)
            self.scaler_y = joblib.load(scaler_y_path)
            with open(config_path, 'r', encoding="utf-8") as f:
                self.config = json.load(f)
        except KeyError as e:
            raise ValueError(f"Missing key in artifacts mapping: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model artifacts from {model_dir}: {e}")

        # 2. Preprocess Data (Feature Engineering + Scaling)
        self.raw_df = df.copy()
        self.raw_df["timestamp"] = pd.to_datetime(self.raw_df["timestamp"], errors="coerce")

        self.processed_df = self._feature_engineering(self.raw_df)

        # 3. Validate Data Length + lấy config
        self.win_size = self.config['window_size']
        self.features = self.config['features']

        # ---- NEW: load metrics từ config (nếu có) để không hard-code ở code khác ----
        # Ví dụ trong model_config.json:
        # {
        #   "window_size": 24,
        #   "features": [...],
        #   "metrics": ["mae", "rmse", "mape"]
        # }
        self.metrics = self.config.get("metrics", None)

        if len(self.processed_df) < self.win_size:
            raise ValueError(f"Not enough data. Need at least {self.win_size} rows.")

    def _feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Re-implements the training feature engineering logic."""
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.sort_values(by=['city', 'timestamp']).reset_index(drop=True)
        
        # Time features
        df["hour"] = df["timestamp"].dt.hour
        df["month"] = df["timestamp"].dt.month
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
        df["is_peak_hour"] = (df["hour"].between(6, 9) | df["hour"].between(17, 20)).astype(int)
        df["day_progress"] = df["hour"] / 24.0
        
        # Text to Number
        df["wind_speed_value"] = df["wind_speed"].astype(str).apply(lambda x: re.findall(r"[\d.]+", x)).apply(lambda x: float(x[0]) if x else 0.0)
        df["humidity_ratio"] = df["humidity"].astype(str).str.replace("%", "", regex=False).astype(float) / 100
        
        # Weather & Season extraction
        def extract_weather(icon_str):
            match = re.search(r"ic-w-\d{2}-([a-z-]+)-full", str(icon_str))
            return match.group(1) if match else "unknown"
        df["weather_type"] = df["weather_icon"].apply(extract_weather)
        
        def get_season(month):
            if month in [2, 3, 4]: return "Spring"
            elif month in [5, 6, 7]: return "Summer"
            elif month in [8, 9, 10]: return "Autumn"
            else: return "Winter"
        df["season"] = df["month"].apply(get_season)

        city_std_val = df["aqi"].std() if len(df) > 1 else 0
        df["city_aqi_std"] = city_std_val

        df["humidity_level"] = pd.cut(df["humidity_ratio"], bins=[-0.01, 0.4, 0.7, 1.0], labels=["Low", "Medium", "High"])
        df["wind_category"] = pd.cut(df["wind_speed_value"], bins=[-0.1, 5, 15, 25, np.inf], labels=["Calm", "Breezy", "Windy", "Strong"])
        df["time_of_day"] = pd.cut(df["hour"], bins=[-1, 5, 11, 17, 21, 24], labels=["Late Night", "Morning", "Afternoon", "Evening", "Night"])

        le = LabelEncoder()
        cols_to_encode = ['weather_type', 'season', 'humidity_level', 'wind_category', 'time_of_day']
        for col in cols_to_encode:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            
        return df

    def _predict_logic(self, steps: int) -> pd.DataFrame:
        """Core recursive prediction loop."""
        recent_data = self.processed_df[self.features].tail(self.win_size).values
        current_seq = self.scaler_X.transform(recent_data).reshape(1, self.win_size, -1)
        
        preds = []
        last_time = self.raw_df['timestamp'].max()
        future_times = []

        for i in range(steps):
            pred_scaled = self.model.predict(current_seq, verbose=0)
            pred_val = self.scaler_y.inverse_transform(pred_scaled)[0][0]
            preds.append(pred_val)
            
            next_time = last_time + pd.Timedelta(hours=i+1)
            future_times.append(next_time)

            new_step_features = current_seq[:, -1, :].copy()
            new_step_features = new_step_features.reshape(1, 1, -1)
            current_seq = np.concatenate([current_seq[:, 1:, :], new_step_features], axis=1)

        return pd.DataFrame({'timestamp': future_times, 'aqi': preds})

    def do_forecast_aqi_24h(self) -> List[Dict]:
        df_pred = self._predict_logic(steps=24)
        results = []
        for _, row in df_pred.iterrows():
            results.append({
                "timestamp": row['timestamp'].isoformat(),
                "aqi": max(0, round(row['aqi']))
            })
        return results

    def do_forecast_aqi_7day(self) -> List[Dict]:
        steps = 7 * 24
        df_pred = self._predict_logic(steps=steps)
        df_pred['date'] = df_pred['timestamp'].dt.date
        df_daily = df_pred.groupby('date')['aqi'].mean().reset_index()
        results = []
        for _, row in df_daily.iterrows():
            results.append({
                "date": row['date'].isoformat(),
                "aqi": max(0, round(row['aqi']))
            })
        return results
