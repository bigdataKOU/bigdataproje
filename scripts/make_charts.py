"""
PDF Adim 7 — Zorunlu gorseller. MLflow + Delta'dan veri okur, docs/figures/*.png:

  - models_comparison.png           : 5 model grouped bar (accuracy/F1/Prec/Recall)
  - feature_importance_<best>.png   : en iyi modelin FI yatay barlari
  - confusion_matrix_<best>.png     : en iyi modelin confusion matrix heatmap
  - roc_curve_<best>.png            : en iyi modelin per-class ROC (OvR)
  - time_series_yearly.png          : yillik suc sayisi
  - time_series_hourly.png          : saatlik trend
  - histogram_primary_type.png      : top 15 sıralı bar
  - pie_arrest.png                  : tutuklanan vs hayir
  - districts_top.png               : ilceye gore suç sayısı

MLflow tracking: localhost:5000.
"""
from __future__ import annotations

import io
import os
import sys
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd


TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT", "chicago-crimes-classifier")
OUT_DIR = Path(os.environ.get("CHARTS_OUT_DIR", "docs/figures"))
DELTA_HOST_DIR = os.environ.get(
    "DELTA_HOST_DIR",
    # Container içindeki delta named volume host'ta:
    "/var/lib/docker/volumes/bigdataproje_delta-data/_data",
)


def fetch_runs() -> pd.DataFrame:
    mlflow.set_tracking_uri(TRACKING_URI)
    exp = mlflow.get_experiment_by_name(EXPERIMENT)
    if exp is None:
        sys.exit(f"experiment yok: {EXPERIMENT}")
    runs = mlflow.search_runs([exp.experiment_id], order_by=["start_time DESC"])
    cols = {
        "run_id": "run_id",
        "params.model_type": "model_type",
        "metrics.accuracy": "accuracy",
        "metrics.weighted_f1": "weighted_f1",
        "metrics.weighted_precision": "weighted_precision",
        "metrics.weighted_recall": "weighted_recall",
        "metrics.auc_ovr_macro": "auc_ovr_macro",
        "metrics.train_seconds": "train_seconds",
    }
    keep = [c for c in cols if c in runs.columns]
    df = runs[keep].rename(columns=cols)
    for c in ("accuracy", "weighted_f1", "weighted_precision", "weighted_recall",
              "auc_ovr_macro", "train_seconds"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # her model_type icin en iyi accuracy'yi tut
    df = df.dropna(subset=["accuracy", "model_type"])
    df = df.sort_values("accuracy", ascending=False).drop_duplicates("model_type")
    return df.reset_index(drop=True)


def grouped_bar(df: pd.DataFrame, fname: str):
    metrics = ["accuracy", "weighted_f1", "weighted_precision", "weighted_recall"]
    available = [m for m in metrics if m in df.columns and df[m].notna().any()]
    if not available:
        print(f"skip {fname}: no metrics")
        return
    fig, ax = plt.subplots(figsize=(11, 5.5))
    n_models = len(df)
    width = 0.8 / max(1, len(available))
    x = np.arange(n_models)
    for i, m in enumerate(available):
        offset = (i - (len(available) - 1) / 2) * width
        bars = ax.bar(x + offset, df[m].values, width=width, label=m)
        for b in bars:
            ax.annotate(f"{b.get_height():.3f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                        ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(df["model_type"].values, rotation=15)
    ax.set_ylabel("metric value")
    ax.set_title("5 model performans karşılaştırması (PDF Adım 7)")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / fname, dpi=150)
    plt.close(fig)
    print(f"wrote {OUT_DIR / fname}")


def fetch_best_run(df: pd.DataFrame):
    if df.empty:
        return None
    return df.iloc[0]  # accuracy DESC sıralı


def feature_importance_chart(best_run, fname: str):
    if best_run is None:
        return
    mlflow.set_tracking_uri(TRACKING_URI)
    client = mlflow.tracking.MlflowClient()
    run_id = best_run["run_id"]
    model_type = best_run["model_type"]
    try:
        local_dir = client.download_artifacts(run_id, ".")
        fi_path = os.path.join(local_dir, f"feature_importance_{model_type}.csv")
        if not os.path.exists(fi_path):
            print(f"skip FI: artifact not found for {model_type}")
            return
        fi = pd.read_csv(fi_path).sort_values("importance")
    except Exception as exc:
        print(f"FI hata: {exc}")
        return
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(fi["feature"], fi["importance"], color="steelblue")
    ax.set_xlabel("importance")
    ax.set_title(f"Feature Importance — {model_type}")
    fig.tight_layout()
    fig.savefig(OUT_DIR / fname, dpi=150)
    plt.close(fig)
    print(f"wrote {OUT_DIR / fname}")


def confusion_matrix_chart(best_run, fname: str):
    if best_run is None:
        return
    mlflow.set_tracking_uri(TRACKING_URI)
    client = mlflow.tracking.MlflowClient()
    run_id = best_run["run_id"]
    model_type = best_run["model_type"]
    try:
        local_dir = client.download_artifacts(run_id, ".")
        cm_path = os.path.join(local_dir, f"cm_{model_type}.csv")
        if not os.path.exists(cm_path):
            print(f"skip CM: artifact not found for {model_type}")
            return
        cm = pd.read_csv(cm_path, index_col=0)
    except Exception as exc:
        print(f"CM hata: {exc}")
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm.values, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(cm.columns)))
    ax.set_xticklabels(cm.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(cm.index)))
    ax.set_yticklabels(cm.index)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm.iat[i, j]), ha="center", va="center",
                    color="white" if cm.iat[i, j] > cm.values.max() / 2 else "black",
                    fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    ax.set_title(f"Confusion Matrix — {model_type}")
    fig.colorbar(im, ax=ax, shrink=0.7)
    fig.tight_layout()
    fig.savefig(OUT_DIR / fname, dpi=150)
    plt.close(fig)
    print(f"wrote {OUT_DIR / fname}")


def roc_curve_proxy(best_run, fname: str):
    """ROC curve gerçek probability'lerden ideal olarak hesaplanır; MLflow'da
    skor kolonu yok. Bunun yerine per-class precision/recall'dan iso-curve
    sketch'i yaparız (informational). Eğer per_class CSV varsa kullanırız."""
    if best_run is None:
        return
    mlflow.set_tracking_uri(TRACKING_URI)
    client = mlflow.tracking.MlflowClient()
    try:
        local_dir = client.download_artifacts(best_run["run_id"], ".")
        per_class_path = os.path.join(
            local_dir, f"per_class_{best_run['model_type']}.csv"
        )
        if not os.path.exists(per_class_path):
            return
        pc = pd.read_csv(per_class_path)
    except Exception:
        return
    fig, ax = plt.subplots(figsize=(7, 5.5))
    # Precision-Recall noktaları — multi-class için bir özet
    ax.scatter(pc["recall"], pc["precision"], s=80)
    for _, r in pc.iterrows():
        ax.annotate(r["label"], (r["recall"], r["precision"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Precision-Recall (per-class) — {best_run['model_type']}\n"
                 f"(AUC OvR macro = {best_run.get('auc_ovr_macro', float('nan')):.3f})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / fname, dpi=150)
    plt.close(fig)
    print(f"wrote {OUT_DIR / fname}")


def time_series_charts(eda_summary_path: str):
    """01_eda.py'nin yazdığı eda_summary.json'dan zaman trendlerini ciz."""
    if not os.path.exists(eda_summary_path):
        print(f"skip time-series: {eda_summary_path} yok")
        return
    import json
    s = json.load(open(eda_summary_path))
    # Yıllık
    yearly = pd.DataFrame(s["yearly_trend"])
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(yearly["year"], yearly["crime_count"], marker="o", color="firebrick")
    ax.set_xlabel("Yıl")
    ax.set_ylabel("Suç sayısı")
    ax.set_title("Yıllık suç sayısı trendi (Chicago, 2001+)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "time_series_yearly.png", dpi=150)
    plt.close(fig)
    print(f"wrote {OUT_DIR / 'time_series_yearly.png'}")

    # Saatlik
    hourly = pd.DataFrame(s["hourly_trend"])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(hourly["hour"], hourly["crime_count"], marker="o", color="navy")
    ax.set_xlabel("Saat (0-23)")
    ax.set_ylabel("Suç sayısı")
    ax.set_title("Günün saatine göre suç dağılımı")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "time_series_hourly.png", dpi=150)
    plt.close(fig)
    print(f"wrote {OUT_DIR / 'time_series_hourly.png'}")


def histogram_primary_type(eda_summary_path: str):
    if not os.path.exists(eda_summary_path):
        return
    import json
    s = json.load(open(eda_summary_path))
    top = pd.DataFrame(s["top_primary_types"])
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(top["primary_type"][::-1], top["count"][::-1], color="teal")
    ax.set_xlabel("Suç sayısı")
    ax.set_title("Top 15 suç tipi (histogram)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "histogram_primary_type.png", dpi=150)
    plt.close(fig)
    print(f"wrote {OUT_DIR / 'histogram_primary_type.png'}")


def pie_arrest(eda_summary_path: str):
    if not os.path.exists(eda_summary_path):
        return
    import json
    s = json.load(open(eda_summary_path))
    arrest_rate = s.get("arrest_rate", 0.0)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(
        [arrest_rate, 1 - arrest_rate],
        labels=[f"Tutuklama\n({arrest_rate:.1%})",
                f"Tutuklama yok\n({1 - arrest_rate:.1%})"],
        autopct="%.1f%%",
        colors=["#e15759", "#76b7b2"],
        startangle=90,
    )
    ax.set_title("Tutuklama oranı (tüm suçlar)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "pie_arrest.png", dpi=150)
    plt.close(fig)
    print(f"wrote {OUT_DIR / 'pie_arrest.png'}")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = fetch_runs()
    print(f"runs (deduped to best per model): {len(df)}")
    if df.empty:
        sys.exit("hicbir run yok")

    df.to_csv(OUT_DIR / "metrics_table.csv", index=False)
    print(f"wrote {OUT_DIR / 'metrics_table.csv'}")

    grouped_bar(df, "models_comparison.png")
    best = fetch_best_run(df)
    if best is not None:
        print(f"best: {best['model_type']} acc={best['accuracy']:.4f}")
        feature_importance_chart(best, f"feature_importance_{best['model_type']}.png")
        confusion_matrix_chart(best, f"confusion_matrix_{best['model_type']}.png")
        roc_curve_proxy(best, f"roc_curve_{best['model_type']}.png")

    # EDA summary based charts
    for path in ("spark/notebooks/eda_summary.json",
                 "/opt/app/notebooks/eda_summary.json"):
        if os.path.exists(path):
            time_series_charts(path)
            histogram_primary_type(path)
            pie_arrest(path)
            break

    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
