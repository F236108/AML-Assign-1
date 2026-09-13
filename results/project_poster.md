# Research Presentation Poster: Solar Power Plant Output Forecasting

![Project Research Poster](fig11_project_poster.png)

---

## 📌 Executive Poster Summary Panel

```
========================================================================================================
                      PREDICTING SOLAR POWER PLANT OUTPUT FROM WEATHER DATA
                            Linear Regression Built Strictly From Scratch
                          Applied Machine Learning Project — Plant 1 (India)
========================================================================================================
```

---

### PANEL 1: Background & Problem Statement
- **Objective**: Predict plant-level hourly AC power generation ($P_{AC}$) for a 30 MW solar plant located in Gandikota, AP (14.82°N, 78.28°E).
- **Core Research Question**: How much accuracy is sacrificed when replacing expensive on-site weather sensors with free public satellite reanalysis data (Open-Meteo API)?
- **Strict Constraint**: All algorithms implemented strictly from scratch using `numpy` and `pandas` only (no ML libraries).

---

### PANEL 2: Data Engineering & Location Verification
- **Preprocessing Pipeline**:
  - Resampled 15-minute inverter generation & weather sensor data to **796 hourly mean rows**.
  - Merged Open-Meteo API ERA5 reanalysis weather data (816 rows).
  - Spatial Verification: Sensor irradiation vs Open-Meteo radiation achieved **0.9333 correlation** and identical **12:00 PM solar peak hour**.

---

### PANEL 3: Solver Performance & Model Comparison

| Feature Set | Solver Method | Train RMSE | Test RMSE (All) | Test RMSE (Daytime) | Relative Error % |
|---|---|---|---|---|---|
| **Set A (On-Site Sensors)** | **Normal Equation** | **537.62 kW** | **539.46 kW** | **704.46 kW** | **2.35%** |
| **Set A (On-Site Sensors)** | **Batch GD** ($\alpha=0.001$) | **741.84 kW** | **880.25 kW** | **1,144.76 kW** | **3.82%** |
| **Set A (On-Site Sensors)** | **SGD** ($\alpha=0.01$) | **539.93 kW** | **549.47 kW** | **714.47 kW** | **2.38%** |
| **Set B (Public Weather)** | **Normal Equation** | **2,699.92 kW** | **2,620.94 kW** | **3,409.50 kW** | **11.36%** |
| **Set B (Public Weather)** | **Batch GD** ($\alpha=0.1$) | **2,699.92 kW** | **2,620.94 kW** | **3,409.50 kW** | **11.36%** |
| **Set B (Public Weather)** | **SGD** ($\alpha=0.01$) | **2,744.92 kW** | **2,699.92 kW** | **3,453.25 kW** | **11.51%** |

---

### PANEL 4: Physics & Technical Takeaways
1. **On-Site Sensors achieve ~2.35% daytime error**, whereas **public weather API data causes ~11.36% daytime error** (nearly 4x higher RMSE).
2. **PV Semiconductor Thermal Loss**: PV panel glass temperature (`module_temp`) rises up to 65°C under direct sun, driving open-circuit voltage drop ($\theta_{\text{module\_temp}} = -108.12$). Public air temperature (`temp_2m`) fails to represent panel thermal heating.
3. **Solver Convergence**: Analytical Normal Equation provides exact global minimum parameters instantaneously for $m \approx 800$.

---

### PANEL 5: Enterprise Web Deployment
- Exported trained weights into `results/model_weights.json`.
- Modern web application interface (`app/index.html`) featuring real-time Z-score scaling, diode output clipping, and diurnal curve visualizer.
- GitHub Code Repository: [https://github.com/F236108/AML-Assign-1](https://github.com/F236108/AML-Assign-1)
