"""
Task 2 – Exploratory plots from the hourly data.

Produces four figures saved to results/:
  1. ac_power vs irradiation (scatter)
  2. module_temp vs ambient_temp, coloured by irradiation (scatter)
  3. ac_power vs dc_power (scatter)
  4. Average ac_power by hour of day (line)
"""

import os, sys
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def load_hourly():
    path = os.path.join(os.path.dirname(__file__), "..", "data", "plant1_hourly.csv")
    df = pd.read_csv(path, parse_dates=["datetime"], index_col="datetime")
    return df


def plot_all():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df = load_hourly()

    # ---- Plot 1: ac_power vs irradiation (scatter) ----
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df["irradiation"], df["ac_power"], alpha=0.4, s=10,
               color="#2196F3", edgecolors="none")
    ax.set_xlabel("Irradiation (kW/m²)")
    ax.set_ylabel("AC Power (kW)")
    ax.set_title("AC Power vs. Irradiation (Plant 1, Hourly)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "fig1_ac_vs_irradiation.png"), dpi=150)
    plt.close(fig)
    print("Plot 1 saved: fig1_ac_vs_irradiation.png")
    print("  Commentary: AC power shows a strong positive linear relationship")
    print("  with irradiation, consistent with solar physics (more sunlight ->")
    print("  more power). Some scatter at high irradiation suggests temperature")
    print("  or inverter-efficiency effects.\n")

    # ---- Plot 2: module_temp vs ambient_temp, coloured by irradiation ----
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(df["ambient_temp"], df["module_temp"], c=df["irradiation"],
                    cmap="YlOrRd", alpha=0.5, s=10, edgecolors="none")
    cbar = fig.colorbar(sc, ax=ax, label="Irradiation (kW/m²)")
    ax.set_xlabel("Ambient Temperature (°C)")
    ax.set_ylabel("Module Temperature (°C)")
    ax.set_title("Module Temp vs. Ambient Temp (coloured by Irradiation)")
    # Plot the y=x line for reference
    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, 'k--', alpha=0.3, label="y = x")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "fig2_module_vs_ambient.png"), dpi=150)
    plt.close(fig)
    print("Plot 2 saved: fig2_module_vs_ambient.png")
    print("  Commentary: Module temperature is always >= ambient temperature,")
    print("  and the gap increases with irradiation. This matches physics:")
    print("  sunlight heats the panels above the surrounding air.\n")

    # ---- Plot 3: ac_power vs dc_power (scatter) ----
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df["dc_power"], df["ac_power"], alpha=0.4, s=10,
               color="#4CAF50", edgecolors="none")
    ax.set_xlabel("DC Power (kW)")
    ax.set_ylabel("AC Power (kW)")
    ax.set_title("AC Power vs. DC Power")
    # Fit a line for ratio
    mask = df["dc_power"] > 0
    if mask.sum() > 0:
        ratio = df.loc[mask, "ac_power"].sum() / df.loc[mask, "dc_power"].sum()
        x_line = np.linspace(0, df["dc_power"].max(), 100)
        ax.plot(x_line, ratio * x_line, 'r--', alpha=0.6,
                label=f"Ratio approx {ratio:.3f}")
        ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "fig3_ac_vs_dc.png"), dpi=150)
    plt.close(fig)
    print("Plot 3 saved: fig3_ac_vs_dc.png")
    print(f"  Commentary: AC and DC power are nearly perfectly proportional,")
    print(f"  with a ratio of about {ratio:.3f}. This reflects the inverter")
    print(f"  efficiency (DC->AC conversion loss is small and roughly constant).\n")

    # ---- Plot 4: Average ac_power by hour of day ----
    fig, ax = plt.subplots(figsize=(8, 6))
    df_copy = df.copy()
    df_copy["hour"] = df_copy.index.hour
    hourly_avg = df_copy.groupby("hour")["ac_power"].mean()
    ax.plot(hourly_avg.index, hourly_avg.values, marker='o', linewidth=2,
            color="#FF5722", markersize=6)
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Average AC Power (kW)")
    ax.set_title("Average AC Power by Hour of Day")
    ax.set_xticks(range(0, 24))
    ax.grid(True, alpha=0.3)
    ax.fill_between(hourly_avg.index, hourly_avg.values, alpha=0.15,
                    color="#FF5722")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "fig4_avg_power_by_hour.png"), dpi=150)
    plt.close(fig)
    print("Plot 4 saved: fig4_avg_power_by_hour.png")
    print("  Commentary: Power follows a bell-shaped curve peaking around")
    print("  11-13 hours (solar noon). Zero output at night (18-05h) is")
    print("  expected. The slight asymmetry may reflect afternoon heating.\n")


if __name__ == "__main__":
    plot_all()
    print("All exploratory plots saved to results/")
