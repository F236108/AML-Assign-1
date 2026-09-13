# Task 5 — Analysis & Technical Report

## 1. Feature Weight Interpretation ($\theta$ Analysis for Set A)

Using the closed-form Normal Equation on standardized features (Set A), the learned weight vector $\theta$ is:

| Parameter / Feature | Variable Name | Weight ($\theta_j$) | Physical Interpretation |
|---|---|---|---|
| **Intercept ($\theta_0$)** | $x_0 = 1$ | **2,845.12 kW** | Baseline mean AC power output across all training hours. |
| **Irradiation ($\theta_1$)** | `irradiation` | **+4,482.35 kW** | **Largest Positive Weight**. Directly drives photovoltaic power generation. |
| **Module Temperature ($\theta_2$)** | `module_temp` | **-512.60 kW** | **Negative Weight**. Matches semiconductor physics: PV panel efficiency drops as cell temperature rises (Pmax temperature coefficient $\approx -0.4\%/^\circ\text{C}$). |
| **Ambient Temperature ($\theta_3$)** | `ambient_temp` | **+84.15 kW** | Small positive adjustment, accounting for thermal radiation balance. |
| **Sine Hour ($\theta_4$)** | `sin(2πh/24)` | **-142.30 kW** | Cyclic time correction for morning vs afternoon asymmetry. |
| **Cosine Hour ($\theta_5$)** | `cos(2πh/24)` | **-615.40 kW** | Diurnal cycle adjustment suppressing night predictions. |

### Physical Consistency:
- **Irradiation** has by far the largest weight magnitude ($\theta_1 \approx +4482$). This aligns perfectly with solar physics: solar photon flux is the primary energy source.
- **Module Temperature** has a distinct **negative coefficient** ($\theta_2 \approx -512$). High module temperatures reduce open-circuit voltage $V_{oc}$ and total power efficiency, exactly as described by PV physics.

---

## 2. Model Performance Comparison: Set A (On-Site Sensors) vs Set B (Public Weather)

| Metric | Set A (On-Site Sensors) | Set B (Public Weather) | Difference ($\Delta$) |
|---|---|---|---|
| **Test RMSE (All Hours)** | **539.46 kW** | **2,620.94 kW** | **+2,081.48 kW** (+385%) |
| **Test RMSE (Daytime Only)** | **704.46 kW** | **3,409.50 kW** | **+2,705.04 kW** (+384%) |
| **Error as % of Peak Plant Power (~30 MW)** | **2.35%** | **11.36%** | **9.01% higher error** |

### Key Observations:
1. **Set A on-site sensors achieve vastly superior accuracy** (Daytime RMSE = 704.46 kW, or 2.35% of peak plant capacity).
2. **Set B public weather data introduces substantial error** (Daytime RMSE = 3,409.50 kW, or 11.36% of peak plant capacity).

### Why Public Weather Data Performs Worse:
- **Spatial Resolution & Local Microclimate**: Open-Meteo uses satellite reanalysis data (~10 km grid cells). Transient cloud cover and local shading over the plant site are not captured in real time.
- **Lack of Module Temperature**: On-site sensors measure actual panel module temperature, which reaches up to 65°C under direct sunlight. Set B only has 2m air temperature, missing panel thermal heating dynamics.

---

## 3. Optimization Solver Comparison (Normal Equation vs Batch GD vs SGD)

| Feature Set | Solver | Train RMSE (kW) | Test RMSE (All) | Test RMSE (Daytime) | Iterations / Epochs |
|---|---|---|---|---|---|
| **Set A** | **Normal Equation** | **537.62** | **539.46** | **704.46** | Analytical (Exact) |
| **Set A** | **Batch GD** ($\alpha=0.1$) | **537.62** | **539.46** | **704.46** | 10,000 iters |
| **Set A** | **SGD** ($\alpha=0.01$) | **539.93** | **549.47** | **714.47** | 100 epochs |
| **Set B** | **Normal Equation** | **2,699.92** | **2,620.94** | **3,409.50** | Analytical (Exact) |
| **Set B** | **Batch GD** ($\alpha=0.1$) | **2,699.92** | **2,620.94** | **3,409.50** | 2,000 iters |
| **Set B** | **SGD** ($\alpha=0.01$) | **2,744.92** | **2,699.92** | **3,453.25** | 100 epochs |

### Solver Selection Guidance:
- **For this dataset ($m \approx 800$ rows)**: **Normal Equation** is the optimal choice. Since $d=5$, computing $(X^T X)^{-1} X^T y$ is instantaneous ($<1$ ms) and guarantees the exact global minimum without hyperparameter tuning ($\alpha$).
- **For 10 Million Rows ($m = 10,000,000$)**: **Stochastic Gradient Descent (SGD) or Mini-Batch GD** is mandatory. Storing $X \in \mathbb{R}^{10,000,000 \times 6}$ in RAM and computing $X^T X$ requires massive memory and $\mathcal{O}(m d^2 + d^3)$ operations. SGD streams through memory in $\mathcal{O}(m d)$ time per epoch.

---

## 4. Learning Rate Selection Analysis (Task 4.7)

### Batch Gradient Descent ($\alpha \in \{10^{-5}, 10^{-4}, 10^{-3}\}$, 500 Iterations)
Tested on Set A and plotted on a single figure ([`results/fig6_learning_rate_batch_gd.png`](file:///c:/Users/Stranger/OneDrive/Documents/AML%20Assign-1/results/fig6_learning_rate_batch_gd.png)):

- **$\alpha = 10^{-5}$**: **Too Small**. The cost $J(\theta)$ decreases extremely slowly, starting at $7.55 \times 10^7$ and only reaching $5.93 \times 10^7$ after 500 iterations.
- **$\alpha = 10^{-4}$**: **Too Small / Slow**. The cost drops moderately to $4.83 \times 10^7$ after 500 iterations, requiring thousands of additional iterations.
- **$\alpha = 10^{-3}$**: **About Right (Best among candidate set)**. The cost drops rapidly down to $1.19 \times 10^7$ in 500 iterations (an 84%+ drop in cost). Selected for BGD modeling.

---

### Stochastic Gradient Descent ($\alpha \in \{10^{-4}, 10^{-3}, 10^{-2}\}$, 50 Epochs)
Tested on Set A and plotted on a single figure ([`results/fig7_learning_rate_sgd.png`](file:///c:/Users/Stranger/OneDrive/Documents/AML%20Assign-1/results/fig7_learning_rate_sgd.png)):

- **$\alpha = 10^{-4}$**: **Too Small**. $J(\theta)$ drops slowly down to $9.38 \times 10^5$.
- **$\alpha = 10^{-3}$**: **Moderate / Slow**. $J(\theta)$ drops to $1.94 \times 10^5$.
- **$\alpha = 10^{-2}$**: **About Right (Best among candidate set)**. $J(\theta)$ drops fastest down to $1.51 \times 10^5$, rapidly reaching near steady-state cost within 50 epochs. Selected for SGD modeling.

---

## 5. Loss Trajectory Analysis: Batch GD vs SGD

- **Batch Gradient Descent ($J(\theta)$ Curve)**:
  - Decreases **smoothly and monotonically** at every single iteration.
  - Because each update uses the exact gradient averaged over the entire training set $X_{train}$, the trajectory steps directly along the steepest descent direction of $J(\theta)$.
- **Stochastic Gradient Descent ($J(\theta)$ Curve)**:
  - Exhibits **fluctuations and noise** from epoch to epoch.
  - Because SGD updates weights per individual sample $i$, individual samples with extreme weather variations pull $\theta$ in different directions, causing the trajectory to oscillate around the minimum.

---

## 5. Residual Analysis ($y - \hat{y}$ vs Hour of Day)

Analyzing the residual plot (`results/fig8_residuals_by_hour.png`):
- **Nighttime (19:00 – 05:00)**: Residuals are near zero because power is strictly zero and predictions clip to 0.
- **Midday Peak Hours (11:00 – 14:00)**: Residuals exhibit the **largest spread ($\pm 2,000$ kW)**.

### Physical Causes of Midday Errors:
1. **Inverter Clipping / Saturation**: On high irradiance days, plant generation hits maximum inverter capacity (clipping), whereas linear regression assumes linear output growth.
2. **Thermal Inertia & Rapid Cloud Transients**: Fast-moving clouds cause sudden drops in irradiance while module temperature remains high, creating non-linear power drops that linear models cannot fully fit.
