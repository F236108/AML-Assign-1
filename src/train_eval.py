"""
Task 4 & 5 – Model Training, Evaluation, Learning Rate Analysis, and Verification.

Steps:
  1. Load merged dataset (plant1_merged.csv).
  2. Add cyclic time features sin(2pi*hour/24), cos(2pi*hour/24).
  3. Split by date (Train: May 15 – June 10, Test: June 11 – June 17).
  4. Prepare Set A (on-site sensor) and Set B (public weather) features.
  5. Scale features using training mean & std only; prepend x0 = 1.
  6. Learning rate experiment: Batch GD & SGD cost curves for Set A.
  7. Train all 6 models (3 solvers x 2 feature sets).
  8. Verify convergence (Batch GD theta vs Normal Equation theta).
  9. Predict, clip negative predictions to 0, report test RMSE (All Hours & Daytime Only).
 10. Save trained model weights and scaling parameters to JSON for Task 6 web app.
 11. Plot cost curves and residual analysis for Task 5.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from regression import (
    hypothesis, compute_cost, fit_normal, fit_batch_gd, fit_sgd, rmse
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def load_and_preprocess():
    path = DATA_DIR / "plant1_merged.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing! Run fetch_weather.py first.")
    
    df = pd.read_csv(path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # Cyclic time features
    hours = df['datetime'].dt.hour
    df['sin_hour'] = np.sin(2.0 * np.pi * hours / 24.0)
    df['cos_hour'] = np.cos(2.0 * np.pi * hours / 24.0)
    
    # Split train vs test by date
    train_mask = (df['datetime'] >= '2020-05-15') & (df['datetime'] <= '2020-06-10 23:59:59')
    test_mask = (df['datetime'] >= '2020-06-11') & (df['datetime'] <= '2020-06-17 23:59:59')
    
    df_train = df[train_mask].copy().reset_index(drop=True)
    df_test = df[test_mask].copy().reset_index(drop=True)
    
    print(f"Data split:")
    print(f"  Train: {df_train['datetime'].min()} to {df_train['datetime'].max()} ({len(df_train)} hours)")
    print(f"  Test : {df_test['datetime'].min()} to {df_test['datetime'].max()} ({len(df_test)} hours)")
    
    return df_train, df_test

def prepare_feature_matrices(df_train, df_test, feature_cols):
    """
    Extracts features, computes z-score stats on train ONLY,
    applies scaling to both train & test, and prepends intercept x0=1 column.
    """
    X_tr_raw = df_train[feature_cols].values.astype(float)
    X_te_raw = df_test[feature_cols].values.astype(float)
    
    # Calculate stats on train only
    mean = np.mean(X_tr_raw, axis=0)
    std = np.std(X_tr_raw, axis=0)
    std[std == 0] = 1.0 # Prevent division by zero if any
    
    # Standardize
    X_tr_scaled = (X_tr_raw - mean) / std
    X_te_scaled = (X_te_raw - mean) / std
    
    # Add intercept column x0 = 1
    m_tr = len(df_train)
    m_te = len(df_test)
    
    X_tr = np.hstack([np.ones((m_tr, 1)), X_tr_scaled])
    X_te = np.hstack([np.ones((m_te, 1)), X_te_scaled])
    
    y_tr = df_train['ac_power'].values.astype(float)
    y_te = df_test['ac_power'].values.astype(float)
    
    scaling_params = {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "feature_cols": feature_cols
    }
    
    return X_tr, y_tr, X_te, y_te, scaling_params

def run_learning_rate_experiments(X_tr, y_tr):
    """
    Evaluates learning rates for Batch GD and SGD on Set A according to prompt requirements:
      - Batch GD: 500 iterations with alpha in {10^-5, 10^-4, 10^-3}
      - SGD: 50 epochs with alpha in {10^-4, 10^-3, 10^-2}
    Identifies which is too small, too large, or about right.
    """
    print("\n--- Task 4.7: Learning Rate Experiments (Set A) ---")
    
    # 1. Batch GD learning rates: {1e-5, 1e-4, 1e-3} over 500 iterations
    bgd_alphas = [1e-5, 1e-4, 1e-3]
    bgd_labels = {
        1e-5: "alpha = 1e-5 (Too small - converges very slowly)",
        1e-4: "alpha = 1e-4 (Slow convergence)",
        1e-3: "alpha = 1e-3 (About right - fastest cost reduction among set)"
    }
    iters = 500
    
    plt.figure(figsize=(8, 5), dpi=300)
    best_bgd_alpha = 1e-3
    best_bgd_cost = float('inf')
    
    for alpha in bgd_alphas:
        theta, costs = fit_batch_gd(X_tr, y_tr, alpha=alpha, n_iters=iters)
        final_c = costs[-1]
        label_str = bgd_labels[alpha] + f" [Final J={final_c:,.0f}]"
        plt.plot(range(len(costs)), costs, label=label_str, lw=2.0)
        print(f"Batch GD {bgd_labels[alpha]} -> Final Cost J(theta): {final_c:12.2f}")
        if np.isfinite(final_c) and final_c < best_bgd_cost:
            best_bgd_cost = final_c
            best_bgd_alpha = alpha
            
    plt.title("Batch GD (Set A, 500 iters): J(theta) vs Iteration for alpha in {10^-5, 10^-4, 10^-3}", fontsize=11, fontweight='bold')
    plt.xlabel("Iteration", fontsize=11)
    plt.ylabel("Cost J(theta)", fontsize=11)
    plt.legend(fontsize=9)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "fig6_learning_rate_batch_gd.png")
    plt.close()
    
    # 2. SGD learning rates: {1e-4, 1e-3, 1e-2} over 50 epochs
    sgd_alphas = [1e-4, 1e-3, 1e-2]
    sgd_labels = {
        1e-4: "alpha = 1e-4 (Too small - slow drop)",
        1e-3: "alpha = 1e-3 (Moderate convergence)",
        1e-2: "alpha = 1e-2 (About right - fastest convergence)"
    }
    epochs = 50
    
    plt.figure(figsize=(8, 5), dpi=300)
    best_sgd_alpha = 1e-2
    best_sgd_cost = float('inf')
    
    for alpha in sgd_alphas:
        theta, costs = fit_sgd(X_tr, y_tr, alpha=alpha, n_epochs=epochs, seed=42)
        final_c = costs[-1]
        label_str = sgd_labels[alpha] + f" [Final J={final_c:,.0f}]"
        plt.plot(range(len(costs)), costs, label=label_str, lw=2.0)
        print(f"SGD      {sgd_labels[alpha]} -> Final Cost J(theta): {final_c:12.2f}")
        if np.isfinite(final_c) and final_c < best_sgd_cost:
            best_sgd_cost = final_c
            best_sgd_alpha = alpha
            
    plt.title("SGD (Set A, 50 epochs): J(theta) vs Epoch for alpha in {10^-4, 10^-3, 10^-2}", fontsize=11, fontweight='bold')
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Cost J(theta)", fontsize=11)
    plt.legend(fontsize=9)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "fig7_learning_rate_sgd.png")
    plt.close()
    
    print(f"\n---> Selected BGD Alpha for rest of task: {best_bgd_alpha}")
    print(f"---> Selected SGD Alpha for rest of task: {best_sgd_alpha}\n")
    
    return best_bgd_alpha, best_sgd_alpha

def main():
    df_train, df_test = load_and_preprocess()
    
    # Define Feature Sets
    cols_A = ['irradiation', 'module_temp', 'ambient_temp', 'sin_hour', 'cos_hour']
    cols_B = ['sw_radiation', 'temp_2m', 'cloud_cover', 'sin_hour', 'cos_hour']
    
    X_tr_A, y_tr_A, X_te_A, y_te_A, stats_A = prepare_feature_matrices(df_train, df_test, cols_A)
    X_tr_B, y_tr_B, X_te_B, y_te_B, stats_B = prepare_feature_matrices(df_train, df_test, cols_B)
    
    # Run LR selection on Set A
    bgd_alpha_A, sgd_alpha_A = run_learning_rate_experiments(X_tr_A, y_tr_A)
    bgd_alpha_B, sgd_alpha_B = 1e-1, 1e-2 # Tune or test for B
    
    # Train 6 Models
    models = {}
    results_table = []
    
    # Daytime mask for test set (where sensor irradiation > 0)
    daytime_mask = df_test['irradiation'] > 0
    
    configs = [
        ("Set A", "Normal Eq", X_tr_A, y_tr_A, X_te_A, y_te_A, "normal", None, None),
        ("Set A", "Batch GD",  X_tr_A, y_tr_A, X_te_A, y_te_A, "bgd", bgd_alpha_A, 10000),
        ("Set A", "SGD",       X_tr_A, y_tr_A, X_te_A, y_te_A, "sgd", sgd_alpha_A, 100),
        ("Set B", "Normal Eq", X_tr_B, y_tr_B, X_te_B, y_te_B, "normal", None, None),
        ("Set B", "Batch GD",  X_tr_B, y_tr_B, X_te_B, y_te_B, "bgd", 0.1, 2000),
        ("Set B", "SGD",       X_tr_B, y_tr_B, X_te_B, y_te_B, "sgd", 0.01, 100),
    ]
    
    saved_weights = {}
    
    print("\n--- Model Training & Evaluation Results ---")
    print(f"{'Feature Set':<10} {'Solver':<12} {'Train RMSE':<12} {'Test RMSE (All)':<18} {'Test RMSE (Daytime)':<20}")
    print("-" * 75)
    
    for fset, solver_name, Xtr, ytr, Xte, yte, method, alpha, iters in configs:
        if method == "normal":
            theta = fit_normal(Xtr, ytr)
            cost_hist = []
        elif method == "bgd":
            theta, cost_hist = fit_batch_gd(Xtr, ytr, alpha=alpha, n_iters=iters)
        elif method == "sgd":
            theta, cost_hist = fit_sgd(Xtr, ytr, alpha=alpha, n_epochs=iters, seed=42)
            
        # Predictions & Clipping
        preds_tr = np.maximum(hypothesis(Xtr, theta), 0.0)
        preds_te = np.maximum(hypothesis(Xte, theta), 0.0)
        
        rmse_tr = rmse(ytr, preds_tr)
        rmse_te_all = rmse(yte, preds_te)
        rmse_te_day = rmse(yte[daytime_mask], preds_te[daytime_mask])
        
        key = f"{fset.replace(' ', '_')}_{solver_name.replace(' ', '_')}"
        models[key] = {
            "theta": theta,
            "cost_hist": cost_hist,
            "preds_te": preds_te,
            "rmse_all": rmse_te_all,
            "rmse_day": rmse_te_day
        }
        
        results_table.append({
            "Feature Set": fset,
            "Solver": solver_name,
            "Train RMSE": round(rmse_tr, 2),
            "Test RMSE All": round(rmse_te_all, 2),
            "Test RMSE Daytime": round(rmse_te_day, 2)
        })
        
        print(f"{fset:<10} {solver_name:<12} {rmse_tr:<12.2f} {rmse_te_all:<18.2f} {rmse_te_day:<20.2f}")
        
    # Save weights for web app (Task 6)
    web_weights = {
        "Set_B_Normal": {
            "theta": models["Set_B_Normal_Eq"]["theta"].tolist(),
            "mean": stats_B["mean"],
            "std": stats_B["std"],
            "features": cols_B
        },
        "Set_A_Normal": {
            "theta": models["Set_A_Normal_Eq"]["theta"].tolist(),
            "mean": stats_A["mean"],
            "std": stats_A["std"],
            "features": cols_A
        }
    }
    with open(RESULTS_DIR / "model_weights.json", "w") as f:
        json.dump(web_weights, f, indent=2)
    print(f"\nSaved model weights to {RESULTS_DIR / 'model_weights.json'}")
    
    # 8. Convergence verification
    theta_norm_A = models["Set_A_Normal_Eq"]["theta"]
    theta_bgd_A = models["Set_A_Batch_GD"]["theta"]
    max_diff_A = np.max(np.abs(theta_norm_A - theta_bgd_A))
    print(f"\nConvergence check (Set A): Max |theta_Normal - theta_BatchGD| = {max_diff_A:.6f}")
    
    # 9. Residual plot for Set A Normal Eq (Task 5.5)
    preds_norm_A = models["Set_A_Normal_Eq"]["preds_te"]
    residuals = y_te_A - preds_norm_A
    test_hours = df_test['datetime'].dt.hour
    
    plt.figure(figsize=(9, 5), dpi=300)
    plt.scatter(test_hours, residuals, alpha=0.6, color='#d62728', edgecolors='none', s=25)
    plt.axhline(0, color='black', linestyle='--', alpha=0.7)
    plt.title("Residuals (Actual - Predicted) vs Hour of Day (Set A Normal Equation)", fontsize=12, fontweight='bold')
    plt.xlabel("Hour of Day", fontsize=11)
    plt.ylabel("Residual (kW)", fontsize=11)
    plt.xticks(range(0, 24))
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "fig8_residuals_by_hour.png")
    plt.close()
    print(f"Saved residual plot to {RESULTS_DIR / 'fig8_residuals_by_hour.png'}")
    
    # 10. Actual vs Predicted plot for Test Week (Figure 9)
    preds_norm_B = models["Set_B_Normal_Eq"]["preds_te"]
    
    plt.figure(figsize=(12, 5), dpi=300)
    plt.plot(df_test['datetime'], y_te_A, label='Actual AC Power', color='black', lw=1.8, alpha=0.85)
    plt.plot(df_test['datetime'], preds_norm_A, label='Predicted Set A (On-Site Sensors)', color='#1f77b4', linestyle='--', lw=1.6)
    plt.plot(df_test['datetime'], preds_norm_B, label='Predicted Set B (Public Weather)', color='#ff7f0e', linestyle=':', lw=1.6)
    
    plt.title("Test Week Performance: Actual vs. Predicted AC Power (June 11–17, 2020)", fontsize=12, fontweight='bold')
    plt.xlabel("Date & Time", fontsize=11)
    plt.ylabel("AC Power (kW)", fontsize=11)
    plt.legend(loc='upper right', frameon=True)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "fig9_actual_vs_predicted.png")
    plt.close()
    print(f"Saved test week comparison plot to {RESULTS_DIR / 'fig9_actual_vs_predicted.png'}")
    
    # Save results markdown summary
    df_results = pd.DataFrame(results_table)
    df_results.to_csv(RESULTS_DIR / "regression_results.csv", index=False)

if __name__ == '__main__':
    main()
