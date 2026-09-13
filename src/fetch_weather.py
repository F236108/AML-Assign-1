"""
Task 3 – Download public weather data from Open-Meteo API & verify plant location.

Fetches historical weather for Plant 1 (Lat 14.82, Lon 78.28, Gandikota AP)
for the period 2020-05-15 to 2020-06-17.
Merges with plant1_hourly.csv and plots 3-day verification.
"""
from pathlib import Path
import json
import requests
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

LATITUDE = 14.82
LONGITUDE = 78.28
START_DATE = "2020-05-15"
END_DATE = "2020-06-17"

def fetch_open_meteo():
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "hourly": "shortwave_radiation,temperature_2m,cloud_cover",
        "timezone": "Asia/Kolkata"
    }
    
    print(f"Fetching Open-Meteo weather data for Lat {LATITUDE}, Lon {LONGITUDE}...")
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    
    hourly_data = data["hourly"]
    df_weather = pd.DataFrame({
        "datetime": pd.to_datetime(hourly_data["time"]),
        "sw_radiation": hourly_data["shortwave_radiation"],
        "temp_2m": hourly_data["temperature_2m"],
        "cloud_cover": hourly_data["cloud_cover"]
    })
    
    out_path = DATA_DIR / "plant1_openmeteo.csv"
    df_weather.to_csv(out_path, index=False)
    print(f"Saved weather data to {out_path} ({len(df_weather)} rows)")
    return df_weather

def verify_location(df_merged):
    # Select 3 days: May 16, 17, 18
    mask = (df_merged['datetime'] >= '2020-05-16') & (df_merged['datetime'] < '2020-05-19')
    sub = df_merged[mask].copy()
    
    plt.figure(figsize=(12, 5), dpi=300)
    
    # Primary axis: sensor irradiation (kW/m² or W/m²)
    ax1 = plt.gca()
    p1 = ax1.plot(sub['datetime'], sub['irradiation'], color='#1f77b4', lw=2, label='Sensor Irradiation (kW/m²)')
    ax1.set_xlabel('Date & Time', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Sensor Irradiation (kW/m²)', color='#1f77b4', fontsize=11, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='#1f77b4')
    
    # Secondary axis: Open-Meteo sw_radiation (W/m²)
    ax2 = ax1.twinx()
    p2 = ax2.plot(sub['datetime'], sub['sw_radiation'], color='#ff7f0e', linestyle='--', lw=2, label='Open-Meteo Shortwave (W/m²)')
    ax2.set_ylabel('Open-Meteo Shortwave Radiation (W/m²)', color='#ff7f0e', fontsize=11, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='#ff7f0e')
    
    # Title and layout
    plt.title('Location Verification: On-Site Sensor vs Open-Meteo Weather (May 16–18, 2020)', fontsize=13, fontweight='bold', pad=12)
    
    # Combined legend
    lines = p1 + p2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right', frameon=True)
    
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    
    out_fig = RESULTS_DIR / "fig5_weather_verification.png"
    plt.savefig(out_fig)
    plt.close()
    print(f"Saved location verification plot to {out_fig}")
    
    # Calculate correlation
    corr = df_merged['irradiation'].corr(df_merged['sw_radiation'])
    print(f"Correlation between sensor irradiation & Open-Meteo sw_radiation: {corr:.4f}")
    
    # Peak hour verification
    sensor_peak_hour = df_merged.groupby(df_merged['datetime'].dt.hour)['irradiation'].mean().idxmax()
    meteo_peak_hour = df_merged.groupby(df_merged['datetime'].dt.hour)['sw_radiation'].mean().idxmax()
    print(f"Peak hour — Sensor: {sensor_peak_hour}:00, Open-Meteo: {meteo_peak_hour}:00")

def main():
    # Load plant hourly data
    plant_path = DATA_DIR / "plant1_hourly.csv"
    if not plant_path.exists():
        raise FileNotFoundError(f"{plant_path} missing! Run prepare.py first.")
    df_plant = pd.read_csv(plant_path)
    df_plant['datetime'] = pd.to_datetime(df_plant['datetime'])
    
    # Fetch weather
    df_weather = fetch_open_meteo()
    
    # Merge
    df_merged = pd.merge(df_plant, df_weather, on='datetime', how='inner')
    merged_path = DATA_DIR / "plant1_merged.csv"
    df_merged.to_csv(merged_path, index=False)
    print(f"Saved merged dataset to {merged_path} ({len(df_merged)} rows)")
    
    # Verify location
    verify_location(df_merged)

if __name__ == '__main__':
    main()
