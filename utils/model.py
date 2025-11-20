class ForecastModel:
    def __init__(self):
        NotImplementedError

    def do_forecast_aqi_7day(self):
        NotImplementedError
    
    def do_forecast_aqi_24h(self):
        NotImplementedError
        

class MockForcastModel:
    def do_forecast_aqi_7day(self):
        "TODO RANDOM"
        NotImplementedError
    
    def do_forecast_aqi_24h(self):
        "TODO RANDOM"
        NotImplementedError