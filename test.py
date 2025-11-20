import pandas as pd

url = "https://raw.githubusercontent.com/nghiahsgs/iqair-dataset/refs/heads/main/result/ho-chi-minh-city/aqi_ho-chi-minh-city_2025_may.csv"

df = pd.read_csv(url)
print(df.head())