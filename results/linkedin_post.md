# LinkedIn Post — Solar Power Plant Generation Forecasting

*(Copy & paste the text below directly onto LinkedIn)*

---

 **Can we accurately forecast grid-scale Solar Power Generation strictly using weather data?**

As renewable energy expands across power grids worldwide, predicting photovoltaic (PV) generation hours in advance is critical for load balancing and grid stability. 

In our latest **Applied Machine Learning Project**, my team and I built a **Linear Regression engine strictly from scratch** (using `numpy` and `pandas` only) to forecast hourly AC power generation for a **30 MW solar plant in Gandikota, India**.

Here are our key findings and engineering takeaways:

🔬 **Key Machine Learning & Physics Insights:**
1. **On-Site Sensors vs. Public Weather APIs**: On-site weather sensors achieved an impressive **2.35% error rate** (704 kW Daytime RMSE), compared to **11.36% error** (3,409 kW Daytime RMSE) when using satellite reanalysis data (Open-Meteo API). Public satellite weather data incurs **nearly 4x higher prediction error** due to spatial grid coarseness (~10 km resolution).
2. **Semiconductor Thermal Efficiency Penalty**: Our model automatically discovered the negative temperature coefficient of PV panels ($\theta_{\text{module\_temp}} = -108.12$). High module temperatures elevate thermal losses and drop output voltage.
3. **Solver Comparison**: The Analytical Normal Equation $(X^T X)^{-1} X^T y$ provided the exact global minimum instantaneously, while Batch Gradient Descent with $\alpha=10^{-3}$ made stable progress toward the same solution.
4. **Interactive Enterprise Web App**: We exported the trained model weights into a modern web predictor with real-time Z-score transformations, diurnal solar curve visualizer, and capacity utilization gauges.

---

📖 **Read the full Technical Blog Post**:
[Link to your Medium Blog Post / results/blog_post.md]

💻 **Explore the GitHub Repository & Code**:
https://github.F236108/AML-Assign-1

---

#MachineLearning #SolarEnergy #RenewableEnergy #DataScience #Python #LinearRegression #AI #GridDecarbonization #CleanTech #AppliedAI
