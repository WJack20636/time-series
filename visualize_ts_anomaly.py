from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score


REPO_ROOT = Path(__file__).resolve().parent
DATA_FILE = REPO_ROOT / "data" / "nab" / "realKnownCause" / "ambient_temperature_system_failure.csv"
LABEL_FILE = REPO_ROOT / "data" / "nab" / "labels" / "combined_windows.json"
FIG_DIR = REPO_ROOT / "figures"



def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_FILE)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    labels = json.loads(LABEL_FILE.read_text(encoding="utf-8"))
    windows = labels["realKnownCause/ambient_temperature_system_failure.csv"]

    gt = np.zeros(len(df), dtype=bool)
    for start, end in windows:
        start_ts = pd.to_datetime(start, utc=True)
        end_ts = pd.to_datetime(end, utc=True)
        gt |= (df["timestamp"] >= start_ts) & (df["timestamp"] <= end_ts)

    df["is_anomaly_gt"] = gt
    return df



def robust_zscore_method(df: pd.DataFrame, window: int = 144, threshold: float = 4.0) -> pd.DataFrame:
    out = df.copy()
    out["rolling_median"] = out["value"].rolling(window=window, center=True, min_periods=window // 2).median()
    out["rolling_mad"] = (
        (out["value"] - out["rolling_median"]).abs().rolling(window=window, center=True, min_periods=window // 2).median()
    )
    denom = 1.4826 * out["rolling_mad"].fillna(out["rolling_mad"].median()) + 1e-8
    out["score_robust_z"] = ((out["value"] - out["rolling_median"]) / denom).abs().fillna(0.0)
    out["pred_robust_z"] = out["score_robust_z"] > threshold
    return out



def isolation_forest_method(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["diff_1"] = out["value"].diff()
    out["lag_1"] = out["value"].shift(1)
    out["lag_2"] = out["value"].shift(2)
    out["roll_mean_24"] = out["value"].rolling(24).mean()
    out["roll_std_24"] = out["value"].rolling(24).std()

    feature_cols = ["value", "diff_1", "lag_1", "lag_2", "roll_mean_24", "roll_std_24"]
    X = out[feature_cols].replace([np.inf, -np.inf], np.nan).bfill().ffill().fillna(0.0)

    gt_rate = float(out["is_anomaly_gt"].mean())
    contamination = min(0.05, max(0.005, gt_rate * 1.5))

    model = IsolationForest(
        n_estimators=300,
        max_samples="auto",
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X)

    out["score_iforest"] = -model.decision_function(X)
    threshold = np.quantile(out["score_iforest"], 1.0 - contamination)
    out["pred_iforest"] = out["score_iforest"] >= threshold
    return out



def evaluate(df: pd.DataFrame, pred_col: str) -> dict[str, float]:
    y_true = df["is_anomaly_gt"].astype(int)
    y_pred = df[pred_col].astype(int)
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }



def shade_gt_windows(ax: plt.Axes, df: pd.DataFrame) -> None:
    gt = df["is_anomaly_gt"].to_numpy()
    ts = df["timestamp"].to_numpy()
    starts = np.where((~gt[:-1]) & (gt[1:]))[0] + 1
    ends = np.where((gt[:-1]) & (~gt[1:]))[0] + 1

    if gt[0]:
        starts = np.r_[0, starts]
    if gt[-1]:
        ends = np.r_[ends, len(gt) - 1]

    for i, (s, e) in enumerate(zip(starts, ends, strict=False)):
        ax.axvspan(ts[s], ts[e], color="tomato", alpha=0.2, label="Ground Truth" if i == 0 else None)



def plot_all(df: pd.DataFrame, metrics_robust: dict[str, float], metrics_iforest: dict[str, float]) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Fig1: 原始序列 + 真实异常窗口
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(df["timestamp"], df["value"], lw=1.2, color="#1f77b4", label="Signal")
    shade_gt_windows(ax, df)
    ax.set_title("NAB Example: Raw Time Series with Ground-Truth Anomaly Window")
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_raw_with_ground_truth.png", dpi=180)
    plt.close(fig)

    # Fig2: Robust Z-Score
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    ax1.plot(df["timestamp"], df["value"], lw=1.0, color="#2a9d8f", label="Signal")
    ax1.scatter(
        df.loc[df["pred_robust_z"], "timestamp"],
        df.loc[df["pred_robust_z"], "value"],
        s=10,
        color="crimson",
        label="Predicted Anomaly",
    )
    shade_gt_windows(ax1, df)
    ax1.set_title("Method A: Robust Z-Score Detection")
    ax1.legend(loc="upper right")

    ax2.plot(df["timestamp"], df["score_robust_z"], color="#264653", lw=1.0, label="Robust Z-Score")
    ax2.axhline(4.0, color="crimson", ls="--", lw=1.0, label="Threshold=4.0")
    shade_gt_windows(ax2, df)
    ax2.set_ylabel("Score")
    ax2.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_robust_zscore_detection.png", dpi=180)
    plt.close(fig)

    # Fig3: Isolation Forest
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    ax1.plot(df["timestamp"], df["value"], lw=1.0, color="#457b9d", label="Signal")
    ax1.scatter(
        df.loc[df["pred_iforest"], "timestamp"],
        df.loc[df["pred_iforest"], "value"],
        s=10,
        color="#e63946",
        label="Predicted Anomaly",
    )
    shade_gt_windows(ax1, df)
    ax1.set_title("Method B: Isolation Forest Detection")
    ax1.legend(loc="upper right")

    threshold = np.quantile(df["score_iforest"], 0.95)
    ax2.plot(df["timestamp"], df["score_iforest"], color="#1d3557", lw=1.0, label="Anomaly Score")
    ax2.axhline(threshold, color="#e63946", ls="--", lw=1.0, label="Adaptive Threshold")
    shade_gt_windows(ax2, df)
    ax2.set_ylabel("Score")
    ax2.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_iforest_detection.png", dpi=180)
    plt.close(fig)

    # Fig4: 方法比较
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [1, 2]})

    labels = ["Precision", "Recall", "F1"]
    robust_vals = [metrics_robust["precision"], metrics_robust["recall"], metrics_robust["f1"]]
    iforest_vals = [metrics_iforest["precision"], metrics_iforest["recall"], metrics_iforest["f1"]]
    x = np.arange(len(labels))
    width = 0.35

    ax1.bar(x - width / 2, robust_vals, width=width, label="Robust Z-Score", color="#2a9d8f")
    ax1.bar(x + width / 2, iforest_vals, width=width, label="Isolation Forest", color="#457b9d")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylim(0, 1.0)
    ax1.set_title("Method Comparison on NAB Example")
    ax1.legend(loc="upper right")

    ax2.plot(df["timestamp"], df["value"], lw=1.0, color="gray", alpha=0.8, label="Signal")
    ax2.scatter(
        df.loc[df["pred_robust_z"], "timestamp"],
        df.loc[df["pred_robust_z"], "value"],
        s=8,
        color="#2a9d8f",
        label="Robust Z-Score",
    )
    ax2.scatter(
        df.loc[df["pred_iforest"], "timestamp"],
        df.loc[df["pred_iforest"], "value"],
        s=8,
        color="#e63946",
        label="Isolation Forest",
    )
    shade_gt_windows(ax2, df)
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Value")
    ax2.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_method_comparison.png", dpi=180)
    plt.close(fig)



def main() -> None:
    df = load_data()
    df = robust_zscore_method(df)
    df = isolation_forest_method(df)

    metrics_robust = evaluate(df, "pred_robust_z")
    metrics_iforest = evaluate(df, "pred_iforest")

    plot_all(df, metrics_robust, metrics_iforest)

    print("Done. Figures saved to:", FIG_DIR)
    print("Robust Z-Score metrics:", metrics_robust)
    print("Isolation Forest metrics:", metrics_iforest)


if __name__ == "__main__":
    main()
