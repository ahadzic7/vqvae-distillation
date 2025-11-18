import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit, least_squares
from sklearn.metrics import r2_score


# ============================================================
# CONFIGURATION
# ============================================================

BPD_TARGET = 2.762
# BPD_TARGET = 6.4
FID_TARGET = 143.368
# FID_TARGET = 273.9

PLOTS_DIR = "plots_7"
# CSV_PATH = "grid_search_all_models.csv"
# CSV_PATH = "grid_search_summary6.csv"
CSV_PATH = "merged_results_MNIST.csv"

sns.set(style="whitegrid", context="talk")
os.makedirs(PLOTS_DIR, exist_ok=True)


# ============================================================
# DATA LOADING & CLEANING
# ============================================================

def clear_outliers(df, z_thresh=3):
    """Remove rows where either BPD or FID are more than z_thresh std from the mean."""
    X, Y = df["BPD"].to_numpy(float), df["FID"].to_numpy(float)
    z_x = np.abs(stats.zscore(X, nan_policy='omit'))
    z_y = np.abs(stats.zscore(Y, nan_policy='omit'))
    return df[(z_x < z_thresh) & (z_y < z_thresh)]


def load_data(path):
    """Load and preprocess the grid search data."""
    df = pd.read_csv(path)

    # Sort and save
    df_res = df.sort_values("BPD", ascending=True)
    df_res.to_csv(path, index=False)

    # Extract numeric dimension from "16x16" → 16
    # df["DIM"] = df["DIM"].str.extract(r"(\d+)").astype(int)
    if "DIM" in df.columns:
        df["DIM"] = (
            df["DIM"]
            .astype(str)
            .str.extract(r"(\d+)")
            .squeeze()  # get the Series from the DataFrame returned by extract()
        )
        # Convert to numeric safely (NaNs preserved if no digits found)
        df["DIM"] = pd.to_numeric(df["DIM"], errors="coerce")
    else:
        df["DIM"] = np.nan

    # Convert numeric columns
    numeric_cols = ["BPD", "FID", "DIM", "CH", "BETA"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)

    # Remove outliers
    df = clear_outliers(df)

    # df = df[df["BPD"] < 13]
    df = df[df["BPD"] > 0]

    return df


# ============================================================
# ANALYSIS
# ============================================================

def compute_best_configs(df):
    """Compute best configurations by different metrics."""
    best_bpd = df.loc[df["BPD"].idxmin()]
    best_fid = df.loc[df["FID"].idxmin()]

    # Combined rank
    df["rank_bpd"] = df["BPD"].rank()
    df["rank_fid"] = df["FID"].rank()
    df["rank_sum"] = df["rank_bpd"] + df["rank_fid"]
    best_rank = df.loc[df["rank_sum"].idxmin()]

    # Normalized score
    norm = (df[["BPD", "FID"]] - df[["BPD", "FID"]].min()) / (df[["BPD", "FID"]].max() - df[["BPD", "FID"]].min())
    df["norm_score"] = norm["BPD"] + norm["FID"]
    best_norm = df.loc[df["norm_score"].idxmin()]

    return best_bpd, best_fid, best_rank, best_norm


# ============================================================
# PLOTTING FUNCTIONS
# ============================================================

def plot_correlation_heatmap(df):
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Between Hyperparameters and Metrics")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/correlation_heatmap.png", dpi=300)
    plt.close()


def plot_scatter_by_category(df, best_models):
    """Scatter plots of BPD vs FID colored by various categorical variables."""
    categories = ["BETA", "DIM", "DT", "CH"]
    label_map = {1: "1x1", 2: "2x2", 4: "4x4", 8: "8x8", 16: "16x16"}

    for cat in categories:
        plt.figure(figsize=(9, 8))
        sns.scatterplot(data=df, x="BPD", y="FID", hue=cat, palette="tab10", s=100, edgecolor="black", alpha=0.8)

        # Highlight best models
        for (label, row), color in zip(best_models.items(), plt.cm.Set1.colors):
            plt.scatter(row["BPD"], row["FID"], s=250, facecolor="none",
                        edgecolor=color, linewidth=2, label=label, zorder=5)

        # Add target reference
        # plt.scatter(BPD_TARGET, FID_TARGET, color="red", s=120, edgecolor="black", label="CM Target", zorder=5)
        plt.axvline(BPD_TARGET, color="red", linestyle="--", label="CM reported BPD")


        # Legend formatting
        handles, labels = plt.gca().get_legend_handles_labels()
        if cat == "DIM":
            labels = [label_map.get(int(l), l) if l.isdigit() else l for l in labels]

        plt.legend(handles, labels, title=cat, loc="upper right")
        plt.title(f"BPD vs FID (labeled by {cat})")
        plt.xlabel("BPD")
        plt.ylabel("FID")
        plt.tight_layout()
        plt.savefig(f"{PLOTS_DIR}/bpd_fid_labeled_by_{cat.lower()}.png", dpi=300)
        plt.close()


def plot_metric_vs_param(df, metric, param, target, target_label, ylim=None):
    """Generic bar plot for visualizing a metric vs parameter."""
    plt.figure(figsize=(7, 4))
    sns.barplot(data=df.sort_values(param), x=param, y=metric, hue=param,
                dodge=False, palette="crest", legend=False)
    plt.axhline(target, color="red", linestyle="--", linewidth=1.5)
    plt.text(x=-0.4, y=target - (0.02 * (ylim[1] - ylim[0]) if ylim else 0.25 * target),
             s=target_label, color="red", fontsize=10, fontweight="bold", va="bottom")
    plt.title(f"{metric} Across {param} Values")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/{metric.lower()}_vs_{param.lower()}.png", dpi=300)
    plt.close()


# ============================================================
# REGRESSION FITTING
# ============================================================

import numpy as np
def regression_models():
    """Return dictionary of candidate regression functions with overflow protection."""
    return {
        "inverse": lambda x, a, b, c: a + b / np.clip(x + c, 1e-6, np.inf),
        "exp_decay": lambda x, a, b, c: a + b * np.exp(np.clip(-c * x, -700, 700)),
        "power_law": lambda x, a, b, c: a + b * np.power(np.clip(x, 1e-6, np.inf), -c),
        "logarithmic": lambda x, a, b, c: a + b * np.log(np.clip(x, 1e-6, np.inf)),
        "quadratic": lambda x, a, b, c: a + b * x + c * np.square(x),
        "rational": lambda x, a, b, c: (a + b * x) / np.clip(1 + c * x, 1e-6, np.inf),
        "sigmoid": lambda x, a, b, c: a + b / (1 + np.exp(np.clip(-c * x, -700, 700))),
        "exp_rise": lambda x, a, b, c: a - b * np.exp(np.clip(-c * x, -700, 700)),
    }



def robust_fit(func, x, y, p0=None):
    """Robust least-squares fit with overflow protection."""
    if p0 is None:
        p0 = np.ones(3)

    def safe_func(x, *p):
        try:
            val = func(x, *p)
            # Clip extreme values to prevent overflow
            return np.clip(val, -1e6, 1e6)
        except FloatingPointError:
            return np.zeros_like(x)

    def residuals(p):
        res = y - safe_func(x, *p)
        return np.clip(res, -1e6, 1e6)

    # Add bounds to prevent diverging parameters
    result = least_squares(residuals, p0, loss='soft_l1', bounds=(-1e6, 1e6))
    return result.x


def fit_best_regression(x, y):
    """Try multiple regression models and select best by R²."""
    models = regression_models()
    results = []

    for name, func in models.items():
        try:
            popt = robust_fit(func, x, y)
            r2 = r2_score(y, func(x, *popt))
            results.append((name, r2, popt))
        except Exception:
            results.append((name, np.nan, None))

    df_res = pd.DataFrame(results, columns=["Model", "R²", "Params"]).sort_values("R²", ascending=False)
    return df_res.iloc[0]  # best model info


def plot_regression(df, best_models):
    """Fit and visualize regression between BPD and FID."""

    filtered = df[df["BPD"] < 20]
    x, y = filtered["BPD"].to_numpy(), filtered["FID"].to_numpy()
    best_fit = fit_best_regression(x, y)
    fun = regression_models()[best_fit["Model"]]
    a, b, c = best_fit["Params"]
    print(f"Best model is {best_fit['Model']}")


    plt.figure(figsize=(9, 8))
    sns.scatterplot(data=df, x="BPD", y="FID", hue="MODEL", style="MODEL", s=60, alpha=0.8, edgecolor="black")
    # sns.scatterplot(data=filtered, x="BPD", y="FID", s=60, alpha=0.8, edgecolor="black")

    # Prepare curve for plotting
    r1 = (max(x) - min(x)) * .005 
    r2 = (max(x) - min(x)) * .01
    x_fit = np.linspace(min(x) - r2, max(x) + r1, 200)
    y_fit = fun(x_fit, a, b, c)

    # Overlay best-fit curve
    plt.plot(x_fit, y_fit, color="black", lw=2, label=f"{best_fit['Model']} fit")

    # Highlight top configs
    for (label, row), color in zip(best_models.items(), plt.cm.Set1.colors):
        plt.scatter(row["BPD"], row["FID"], s=200, facecolor="none", edgecolor=color, linewidth=2, label=label, zorder=5)

    plt.axvline(BPD_TARGET, color="red", linestyle="--", label="CM reported BPD")
    
    plt.title(f"BPD–FID Regression ({best_fit['Model']} model)")
    plt.xlabel("BPD")
    plt.ylabel("FID")

    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/regression_{best_fit['Model']}.png", dpi=300)
    plt.close()


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    df = load_data(CSV_PATH)
    df = df[df["BPD"] < 20]
    plot_correlation_heatmap(df)

    print("\n=== Summary Statistics ===")
    print(df.describe())

    best_bpd, best_fid, best_rank, best_norm = compute_best_configs(df)
    best_models = {
        "Best BPD": best_bpd,
        "Best FID": best_fid,
        "Best RankSum": best_rank,
        "Best NormScore": best_norm,
    }

    print("\n=== Best Configurations ===")
    for label, model in best_models.items():
        print(f"\n{label}:\n{model}\n")

    plot_scatter_by_category(df, best_models)

    # Metric plots
    bpd_label = f"{BPD_TARGET} BPD target"
    fid_label = f"{FID_TARGET} FID target"
    for param in ["BETA", "CH", "DIM", "DT"]:
        plot_metric_vs_param(df, "BPD", param, BPD_TARGET, bpd_label)
        plot_metric_vs_param(df, "FID", param, FID_TARGET, fid_label)

    plot_regression(df, best_models)

    df.to_csv("grid_results_processed.csv", index=False)
    print("\n✅ Cleaned results saved to 'grid_results_processed.csv'.")
    print(f"✅ Plots saved in the '{PLOTS_DIR}' folder.")


if __name__ == "__main__":
    main()
