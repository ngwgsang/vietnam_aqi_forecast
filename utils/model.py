import random
import datetime as dt
from typing import List, Dict

import pandas as pd


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
