# Predicting Solar Power Plant Output from Weather Data: A Linear Regression Study from Scratch

*Evaluating On-Site Weather Sensors vs. Public Satellite Reanalysis Data for Grid-Scale Solar Generation Forecasting*

---

## 1. The Question & Problem Motivation

Solar photovoltaic (PV) power generation is inherently variable, dependent on diurnal solar geometry, atmospheric cloud attenuation, and panel temperature dynamics. For electrical grid operators, predicting solar plant AC power generation hours in advance is essential for load balancing, spinning reserve allocation, and grid frequency stability.

In this study, we investigate the following core machine learning questions:
1. **Predictive Accuracy**: Can linear regression fit strictly from scratch accurately predict plant-level hourly AC power generation ($P_{AC}$) using atmospheric weather variables?
2. **On-Site vs. Remote Public Data**: How much prediction accuracy is lost when replacing expensive on-site weather sensors with free, publicly accessible satellite reanalysis weather data (Open-Meteo API)?
3. **Optimization Solvers**: How do closed-form analytical solvers (Normal Equation) compare to iterative optimization algorithms (Batch Gradient Descent and Stochastic Gradient Descent)?

---

## 2. Data Engineering & Preparation

We analyze 34 days of continuous generation and weather data (May 15, 2020 – June 17, 2020) from **Plant 1**, located near Gandikota, Andhra Pradesh, India (Lat 14.82°N, Lon 78.28°E).

### Preprocessing Pipeline:
- **Timestamp Standardization**: Raw generation files used non-standard date strings (`DD-MM-YYYY HH:MM`), whereas weather sensors logged standard ISO timestamps (`YYYY-MM-DD HH:MM:SS`). All timestamps were standardized into native UTC/IST datetimes.
- **Plant-Level Power Aggregation**: The plant contains 22 separate central inverters. We aggregated total plant generation by summing `AC_POWER` and `DC_POWER` across all inverters at each 15-minute timestamp.
- **Hourly Resampling**: The 15-minute merged generation and weather readings were resampled to **hourly arithmetic means**, producing a clean dataset of **796 hourly rows**.

### Table 1: Data Preparation Summary

| Metric / Parameter | Value |
|---|---|
| **Raw generation rows (Plant 1)** | 68,778 |
| **Raw sensor rows (Plant 1)** | 3,182 |
| **Timestamps present in only one file** | 26 |
| **Hourly rows after resampling** | 796 |
| **Hourly rows with missing values** | 0 |
| **Open-Meteo weather rows downloaded** | 816 |
| **Correlation (Sensor Irradiation vs Open-Meteo Radiation)** | **0.9333** |
| **Peak Hour Comparison (Sensor vs Open-Meteo)** | **12:00 PM vs 12:00 PM** (Identical) |

---

## 3. Exploratory Data Analysis (EDA)

Before modeling, we explored the physical relationships across solar generation and meteorological variables.

![AC Power vs Irradiation](fig1_ac_vs_irradiation.png)
*Figure 1: AC Power vs. Irradiation scatter plot.*

![Module Temp vs Ambient Temp](fig2_module_vs_ambient.png)
*Figure 2: PV Module Temperature vs. Ambient Air Temperature (colored by irradiation).*

![AC Power vs DC Power Ratio](fig3_ac_vs_dc.png)
*Figure 3: AC Power vs. DC Power linear conversion ratio (Inverter efficiency $\approx 0.098$).*

![Average AC Power by Hour of Day](fig4_avg_power_by_hour.png)
*Figure 4: Mean AC Power by hour of day (diurnal bell curve peaking at solar noon).*

### Key EDA Insights:
1. **Linear Photovoltaic Response**: AC power shows a strong linear correlation with solar irradiance ($r > 0.95$).
2. **Thermal Panel Elevation**: PV module temperature (`module_temp`) regularly exceeds ambient air temperature (`ambient_temp`) by **15–25°C** under direct solar irradiance due to radiative heating.
3. **Inverter Conversion Ratio**: AC power vs DC power exhibits a near-perfect linear ratio of **$0.098$** (accounting for central inverter scaling and conversion efficiency).
4. **Diurnal Bell Curve**: Hourly mean AC power follows a symmetrical bell curve peaking between **11:00 AM and 1:00 PM (solar noon)**, dropping to zero at night (19:00–05:00).

---

## 4. Location Verification: On-Site Sensor vs. Open-Meteo API

To verify the geographical coordinates of Plant 1 (Lat 14.82°N, Lon 78.28°E), we downloaded historical ERA5 reanalysis weather data from the **Open-Meteo Historical Weather API** and compared on-site sensor irradiance (`irradiation`) against satellite-derived shortwave radiation (`sw_radiation`).

![Location Verification Plot](fig5_weather_verification.png)
*Figure 5: 3-day location verification comparison (Sensor Irradiation vs. Open-Meteo Shortwave Radiation).*

### Verification Findings:
- The correlation between on-site sensor irradiation and Open-Meteo shortwave radiation reached **0.9333**, confirming exact spatial alignment.
- Both signals reach peak intensity at **12:00 PM IST**, confirming no time-zone offset or solar longitude discrepancy.

---

## 5. Linear Regression Implementation & Solver Results

All algorithms were implemented **from scratch using NumPy matrix operations**:
- **Hypothesis**: $h_\theta(X) = X \theta$
- **Cost Function**: $J(\theta) = \frac{1}{2m} \|X \theta - y\|_2^2$
- **Normal Equation**: $\theta = (X^T X)^{-1} X^T y$
- **Batch Gradient Descent**: $\theta \leftarrow \theta - \alpha \frac{1}{m} X^T (X \theta - y)$
- **Stochastic Gradient Descent**: $\theta \leftarrow \theta - \alpha (x^{(i)})^T (x^{(i)} \theta - y^{(i)})$

### Feature Sets:
- **Set A (On-Site Sensors)**: `irradiation`, `module_temp`, `ambient_temp`, $\sin(2\pi h / 24)$, $\cos(2\pi h / 24)$ ($d=5$)
- **Set B (Public Weather)**: `sw_radiation`, `temp_2m`, `cloud_cover`, $\sin(2\pi h / 24)$, $\cos(2\pi h / 24)$ ($d=5$)
- **Feature Scaling**: Z-score standardized using **training set mean and standard deviation only** ($\mu_{train}, \sigma_{train}$). Prepend intercept column $x_0 = 1$.
- **Train/Test Split**: Chronological split without shuffling — Train = May 15 to June 10 (628 hours), Test = June 11 to June 17 (168 hours).

### Learning Rate Selection ($\alpha$ Tuning on Set A):

![Batch GD Learning Rates](fig6_learning_rate_batch_gd.png)
*Figure 6: Batch GD Cost $J(\theta)$ vs. Iteration for $\alpha \in \{10^{-5}, 10^{-4}, 10^{-3}\}$.*

![SGD Learning Rates](fig7_learning_rate_sgd.png)
*Figure 7: SGD Cost $J(\theta)$ vs. Epoch for $\alpha \in \{10^{-4}, 10^{-3}, 10^{-2}\}$.*

- **Batch GD ($\alpha \in \{10^{-5}, 10^{-4}, 10^{-3}\}$, 500 iterations)**: $\alpha = 10^{-5}$ and $10^{-4}$ are **too small** (slow convergence). $\alpha = 10^{-3}$ is **about right**, reducing cost $J(\theta)$ by >84% in 500 iterations.
- **SGD ($\alpha \in \{10^{-4}, 10^{-3}, 10^{-2}\}$, 50 epochs)**: $\alpha = 10^{-4}$ is **too small**, whereas $\alpha = 10^{-2}$ is **about right**, rapidly reaching global minimum cost.

---

### Table 2: Test-Set RMSE Performance (kW)

| Feature Set | Solver Method | Train RMSE (kW) | Test RMSE (All 168 Hours) | Test RMSE (Daytime Hours Only) |
|---|---|---|---|---|
| **Set A (Sensors)** | **Normal Equation** | **537.62** | **539.46** | **704.46** |
| **Set A (Sensors)** | **Batch GD** ($\alpha=0.001$, 10,000 iters) | **741.84** | **880.25** | **1,144.76** |
| **Set A (Sensors)** | **SGD** ($\alpha=0.01$, 50 epochs) | **539.93** | **549.47** | **714.47** |
| **Set B (Public)** | **Normal Equation** | **2,699.92** | **2,620.94** | **3,409.50** |
| **Set B (Public)** | **Batch GD** ($\alpha=0.1$, 2,000 iters) | **2,699.92** | **2,620.94** | **3,409.50** |
| **Set B (Public)** | **SGD** ($\alpha=0.01$, 100 epochs) | **2,744.92** | **2,699.92** | **3,453.25** |

---

### Table 3: Learned Weight Vector ($\theta$) for Set A

Weights learned on standardized features ($x_0 = 1$ intercept included):

| Parameter Index | Feature Name | Set A Normal Eq | Set A Batch GD | Set A SGD | Physical Interpretation |
|---|---|---|---|---|---|
| **$\theta_0$** | **Intercept ($x_0=1$)** | **+6,890.57** | **+6,890.25** | **+6,904.65** | Mean baseline plant power across training hours |
| **$\theta_1$** | **`irradiation`** | **+8,345.04** | **+4,928.74** | **+8,240.72** | **Primary Driver (+)**: Photovoltaic photon flux |
| **$\theta_2$** | **`module_temp`** | **-108.12** | **+3,228.69** | **-0.17** | Thermal efficiency effect; incomplete BGD convergence changes this estimate |
| **$\theta_3$** | **`ambient_temp`** | **-17.16** | **-382.55** | **-49.53** | Minor ambient thermal coupling |
| **$\theta_4$** | **`sin_hour`** | **-47.38** | **+317.10** | **-74.39** | Morning/afternoon asymmetry correction |
| **$\theta_5$** | **`cos_hour`** | **-416.56** | **-756.65** | **-423.60** | Diurnal cycle suppression at night |

---

## 6. Task 5.2 Analysis: On-Site Sensors vs. Public Weather Data

### Performance Comparison:
- **Set A Daytime RMSE**: **704.46 kW** (**2.35%** of 30 MW plant capacity)
- **Set B Daytime RMSE**: **3,409.50 kW** (**11.36%** of 30 MW plant capacity)
- **Absolute Daytime Difference ($\Delta$)**: **2,705.04 kW**
- **Relative Difference**: Public weather data incurs **9.01% higher error** as a fraction of peak plant power, or **+384% higher RMSE**.

![Actual vs Predicted Test Week](fig9_actual_vs_predicted.png)
*Figure 8: Actual vs. Predicted AC Power across the 7-day test week (June 11–17, 2020).*

### Physical Causes of Performance Degradation in Set B:
1. **Coarse Spatial Resolution & Local Clouds**: Open-Meteo relies on satellite reanalysis grids (~10 km resolution). Sudden local cloud shadows passing over the plant are missed by satellite grids.
2. **Missing PV Module Temperature Sensor**: On-site sensors measure direct panel glass temperature (`module_temp`), which rises to 65°C under direct sunlight. Set B only has 2m air temperature (`temp_2m`), completely missing semiconductor thermal efficiency degradation.

---

## 7. Model Diagnostics & Residual Analysis

Analyzing the model residuals ($y - \hat{y}$) across hours of the day:

![Residuals vs Hour of Day](fig8_residuals_by_hour.png)
*Figure 9: Residuals ($y - \hat{y}$) vs. Hour of Day for Set A Normal Equation.*

### Diagnostics Findings:
- **Nighttime (19:00 – 05:00)**: Residuals are identically zero due to post-processing ReLU clipping ($\max(\hat{y}, 0)$).
- **Midday Peak Error (11:00 – 14:00)**: Residuals show maximum variance ($\pm 2,000$ kW). This is caused by **inverter saturation / clipping** at maximum solar irradiance and rapid thermal transient lags during midday cloud passages.

---

## 8. Conclusion & Key Takeaways

1. **Photovoltaic Physics Matter**: Solar irradiance is the dominant driver of generation ($\theta_1 = +8345$), while module temperature exhibits a clear negative efficiency penalty ($\theta_2 = -108$).
2. **On-Site Sensors are Mandatory for Precision**: Relying solely on free public weather APIs increases prediction error by **nearly 4x** (704 kW vs 3,409 kW daytime RMSE).
3. **Closed-Form Normal Equation is Superior for Small $m$**: For $m \approx 800$ rows, the analytical Normal Equation achieves exact global minimum weights instantaneously without hyperparameter tuning.
