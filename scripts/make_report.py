"""
HTML teknik rapor üretir — PDF Adım 7 zorunlu görseller dahil, inline base64 PNG'lerle.

Çıktı:
  docs/rapor.html

Kullanım (dashboard container içinde):
  docker compose exec dashboard python /app/make_report.py
veya host'tan:
  make report
"""
from __future__ import annotations

import base64
import io
import json
import os
from datetime import datetime
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT", "chicago-crimes-classifier")
OUT_HTML = Path(os.environ.get("REPORT_OUT", "/app/rapor.html"))
EDA_SUMMARY_CANDIDATES = [
    "/opt/app/notebooks/eda_summary.json",
    "spark/notebooks/eda_summary.json",
    "/app/eda_summary.json",
]


def fig_html(fig) -> str:
    return pio.to_html(fig, include_plotlyjs="cdn", full_html=False,
                       config={"displaylogo": False})


def load_eda_summary() -> dict:
    for p in EDA_SUMMARY_CANDIDATES:
        if os.path.exists(p):
            with open(p) as f:
                return json.load(f)
    return {}


def fetch_runs() -> pd.DataFrame:
    mlflow.set_tracking_uri(TRACKING_URI)
    exp = mlflow.get_experiment_by_name(EXPERIMENT)
    if exp is None:
        return pd.DataFrame()
    runs = mlflow.search_runs([exp.experiment_id])
    cols = {
        "run_id": "run_id",
        "params.model_type": "model_type",
        "metrics.accuracy": "accuracy",
        "metrics.weighted_f1": "weighted_f1",
        "metrics.weighted_precision": "weighted_precision",
        "metrics.weighted_recall": "weighted_recall",
        "metrics.auc_ovr_macro": "auc_ovr_macro",
        "metrics.train_seconds": "train_seconds",
        "params.n_rows": "n_rows",
    }
    keep = [c for c in cols if c in runs.columns]
    df = runs[keep].rename(columns=cols)
    for c in ("accuracy", "weighted_f1", "weighted_precision", "weighted_recall",
              "auc_ovr_macro", "train_seconds"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["accuracy", "model_type"])
    df = df.sort_values("accuracy", ascending=False).drop_duplicates("model_type")
    return df.reset_index(drop=True)


def chart_models_comparison(df: pd.DataFrame) -> str:
    if df.empty:
        return "<p><em>Model verisi yok.</em></p>"
    metrics = ["accuracy", "weighted_f1", "weighted_precision", "weighted_recall"]
    available = [m for m in metrics if m in df.columns and df[m].notna().any()]
    melted = df[["model_type"] + available].melt(
        id_vars="model_type", var_name="metric", value_name="value",
    )
    fig = px.bar(
        melted, x="model_type", y="value", color="metric",
        barmode="group",
        text="value",
        title="PDF Adım 7 — 5 Model Performans Karşılaştırması (grouped bar chart)",
        labels={"value": "metrik değeri", "model_type": "model"},
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_layout(height=480, yaxis_range=[0, 1.05])
    return fig_html(fig)


def chart_auc(df: pd.DataFrame) -> str:
    if "auc_ovr_macro" not in df.columns or df["auc_ovr_macro"].dropna().empty:
        return ""
    sub = df.dropna(subset=["auc_ovr_macro"]).sort_values("auc_ovr_macro")
    fig = px.bar(
        sub, x="auc_ovr_macro", y="model_type", orientation="h",
        text="auc_ovr_macro",
        title="AUC-ROC (OvR macro) — yüksek = iyi",
        color="auc_ovr_macro", color_continuous_scale="Viridis",
    )
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_layout(height=350, xaxis_range=[0, 1.0])
    return fig_html(fig)


def chart_train_time(df: pd.DataFrame) -> str:
    if "train_seconds" not in df.columns:
        return ""
    sub = df.dropna(subset=["train_seconds"]).sort_values("train_seconds")
    fig = px.bar(
        sub, x="train_seconds", y="model_type", orientation="h",
        text="train_seconds",
        title="Eğitim süresi (saniye) — düşük = hızlı",
        color="train_seconds", color_continuous_scale="OrRd",
    )
    fig.update_traces(texttemplate="%{text:.1f}s", textposition="outside")
    fig.update_layout(height=350)
    return fig_html(fig)


def download_artifact(run_id: str, fname: str) -> str | None:
    """MLflow artifact'i lokale indir, dosya yolunu döndür."""
    try:
        os.makedirs("/tmp/mlflow_dl", exist_ok=True)
        client = mlflow.tracking.MlflowClient()
        local_path = client.download_artifacts(run_id, fname, dst_path="/tmp/mlflow_dl")
        return local_path if os.path.exists(local_path) else None
    except Exception as e:
        print(f"download {fname} failed: {e}")
        return None


def chart_feature_importance(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    best = df.iloc[0]
    fname = f"feature_importance_{best['model_type']}.csv"
    p = download_artifact(best["run_id"], fname)
    if not p:
        return f"<p><em>{best['model_type']} için feature importance artifact bulunamadı.</em></p>"
    fi = pd.read_csv(p).sort_values("importance")
    fig = px.bar(
        fi, x="importance", y="feature", orientation="h",
        text="importance",
        title=f"PDF Adım 7 — Feature Importance (horizontal bar chart) — {best['model_type']}",
        color="importance", color_continuous_scale="Blues",
    )
    fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
    fig.update_layout(height=480)
    return fig_html(fig)


def chart_confusion_matrix(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    best = df.iloc[0]
    fname = f"cm_{best['model_type']}.csv"
    p = download_artifact(best["run_id"], fname)
    if not p:
        return ""
    cm = pd.read_csv(p, index_col=0)
    fig = px.imshow(
        cm.values,
        x=list(cm.columns), y=list(cm.index),
        color_continuous_scale="Blues",
        labels=dict(x="tahmin", y="gerçek", color="count"),
        text_auto=True,
        title=f"PDF Adım 7 — Confusion Matrix (en iyi: {best['model_type']})",
    )
    fig.update_layout(height=520)
    return fig_html(fig)


def chart_per_class_pr(df: pd.DataFrame) -> str:
    """Her sınıf için precision/recall scatter + accuracy/F1 — ROC Curve proxy."""
    if df.empty:
        return ""
    best = df.iloc[0]
    fname = f"per_class_{best['model_type']}.csv"
    p = download_artifact(best["run_id"], fname)
    if not p:
        return ""
    pc = pd.read_csv(p)
    fig = px.scatter(
        pc, x="recall", y="precision", text="label",
        size="tp", color="f1",
        color_continuous_scale="Plasma",
        title=f"PDF Adım 7 — Per-class Precision/Recall (ROC proxy) — {best['model_type']}",
        labels={"f1": "F1 skoru"},
    )
    fig.update_traces(textposition="top center")
    fig.update_layout(height=480, xaxis_range=[0, 1.05], yaxis_range=[0, 1.05])
    return fig_html(fig)


def chart_yearly_trend(eda: dict) -> str:
    if not eda or "yearly_trend" not in eda:
        return ""
    df = pd.DataFrame(eda["yearly_trend"])
    fig = px.line(df, x="year", y="crime_count", markers=True,
                  title="PDF Adım 7 — Yıllık suç sayısı trendi (line chart)",
                  labels={"year": "Yıl", "crime_count": "Suç sayısı"})
    fig.update_layout(height=400)
    return fig_html(fig)


def chart_hourly_trend(eda: dict) -> str:
    if not eda or "hourly_trend" not in eda:
        return ""
    df = pd.DataFrame(eda["hourly_trend"])
    fig = px.line(df, x="hour", y="crime_count", markers=True,
                  title="Günün saatine göre suç sayısı",
                  labels={"hour": "Saat (0-23)", "crime_count": "Suç sayısı"})
    fig.update_layout(height=380)
    return fig_html(fig)


def chart_top_types(eda: dict) -> str:
    if not eda or "top_primary_types" not in eda:
        return ""
    df = pd.DataFrame(eda["top_primary_types"])
    fig = px.bar(df.iloc[::-1], x="count", y="primary_type", orientation="h",
                 text="count", title="PDF Adım 7 — Top 15 suç tipi (histogram)",
                 color="count", color_continuous_scale="Teal")
    fig.update_layout(height=480)
    return fig_html(fig)


def chart_arrest_pie(eda: dict) -> str:
    rate = eda.get("arrest_rate", 0.0)
    fig = go.Figure(data=[go.Pie(
        labels=["Tutuklama yapıldı", "Tutuklama yok"],
        values=[rate, 1 - rate],
        hole=0.4,
        marker=dict(colors=["#e15759", "#76b7b2"]),
    )])
    fig.update_layout(title="PDF Adım 7 — Tutuklama oranı (pie chart)",
                      height=400)
    return fig_html(fig)


def chart_weekly_trend(eda: dict) -> str:
    if not eda or "weekly_trend" not in eda:
        return ""
    day_names = ["Pazar", "Pazartesi", "Salı", "Çarşamba", "Perşembe", "Cuma", "Cumartesi"]
    df = pd.DataFrame(eda["weekly_trend"])
    df["day_name"] = df["day_of_week"].apply(
        lambda d: day_names[(int(d) - 1) % 7] if not pd.isna(d) else "?"
    )
    fig = px.bar(df, x="day_name", y="crime_count",
                 title="Haftanın gününe göre suç sayısı",
                 color="crime_count", color_continuous_scale="Viridis",
                 category_orders={"day_name": day_names})
    fig.update_layout(height=380)
    return fig_html(fig)


def models_table(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    cols = ["model_type", "accuracy", "weighted_f1", "weighted_precision",
            "weighted_recall", "auc_ovr_macro", "train_seconds"]
    cols = [c for c in cols if c in df.columns]
    show = df[cols].copy()
    for c in cols[1:]:
        show[c] = show[c].apply(lambda v: f"{v:.4f}" if pd.notna(v) else "—")
    return show.to_html(index=False, classes="metric-table", border=0)


def build_html() -> str:
    eda = load_eda_summary()
    runs = fetch_runs()

    today = datetime.now().strftime("%d %B %Y")
    best_model = runs.iloc[0]["model_type"] if not runs.empty else "—"
    best_acc = runs.iloc[0]["accuracy"] if not runs.empty else float("nan")

    eda_card = {}
    if eda:
        eda_card = {
            "total_rows": f"{eda.get('total_rows', '?'):,}",
            "unique_types": eda.get("unique_primary_type", "?"),
            "unique_districts": eda.get("unique_district", "?"),
            "arrest_rate": f"{eda.get('arrest_rate', 0):.1%}",
            "domestic_rate": f"{eda.get('domestic_rate', 0):.1%}",
            "year_range": (
                f"{eda['yearly_trend'][0]['year']} – {eda['yearly_trend'][-1]['year']}"
                if eda.get("yearly_trend") else "—"
            ),
        }

    html_parts = []
    html_parts.append(f"""<!doctype html>
<html lang="tr">
<head>
<meta charset="utf-8">
<title>Chicago Crimes Pipeline — Teknik Rapor</title>
<style>
  body {{ font-family: -apple-system, "Segoe UI", Roboto, sans-serif;
         max-width: 1100px; margin: 30px auto; padding: 0 20px; color: #222;
         line-height: 1.6; }}
  h1 {{ font-size: 2em; border-bottom: 3px solid #1e88e5; padding-bottom: 0.3em; }}
  h2 {{ color: #1565c0; margin-top: 2em; border-bottom: 1px solid #ddd; padding-bottom: 0.2em; }}
  h3 {{ color: #1976d2; }}
  .header-box {{ background: linear-gradient(135deg, #1e88e5, #0d47a1);
                  color: white; padding: 30px; border-radius: 8px;
                  margin-bottom: 30px; }}
  .header-box h1 {{ color: white; border: none; margin: 0; font-size: 1.7em; }}
  .header-box .sub {{ font-size: 1.1em; margin-top: 10px; }}
  .student-table {{ background: rgba(255,255,255,0.1); padding: 12px;
                     border-radius: 6px; margin-top: 15px; }}
  .student-table table {{ width: 100%; border-collapse: collapse; }}
  .student-table td {{ padding: 6px 12px; color: white; }}
  .student-table tr td:first-child {{ font-weight: bold; }}
  .kpi {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
          gap: 14px; margin: 20px 0; }}
  .kpi .card {{ background: #f4f8fc; border-left: 4px solid #1e88e5;
                padding: 14px; border-radius: 4px; }}
  .kpi .label {{ font-size: 0.85em; color: #555; }}
  .kpi .value {{ font-size: 1.5em; font-weight: bold; color: #0d47a1;
                  margin-top: 4px; }}
  .winner {{ background: linear-gradient(135deg, #43a047, #2e7d32);
             color: white; padding: 20px; border-radius: 6px;
             margin: 15px 0; font-size: 1.1em; }}
  .metric-table {{ border-collapse: collapse; width: 100%; margin: 15px 0;
                    background: white; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
  .metric-table th, .metric-table td {{ padding: 10px 14px; border-bottom: 1px solid #e0e0e0;
                                          text-align: left; }}
  .metric-table th {{ background: #1565c0; color: white; }}
  .metric-table tr:first-child td {{ font-weight: bold; background: #fff8e1; }}
  blockquote {{ border-left: 4px solid #fb8c00; padding: 10px 16px; background: #fff8e1; margin: 15px 0; }}
  code {{ background: #f5f5f5; padding: 2px 6px; border-radius: 3px; font-size: 0.9em; }}
  .footer {{ text-align: center; margin-top: 60px; padding: 30px; background: #f5f5f5; border-radius: 8px; }}
  .footer .thanks {{ font-size: 1.8em; color: #1565c0; font-weight: bold; }}
  ul li {{ margin: 6px 0; }}
</style>
</head>
<body>

<div class="header-box">
  <h1>Chicago Crimes — Uçtan Uca Büyük Veri Pipeline'ı</h1>
  <div class="sub">BLM442 Büyük Veri Analizine Giriş · Dönem Projesi · {today}</div>
  <div class="student-table">
    <table>
      <tr><td>Üniversite</td><td>Kocaeli Üniversitesi</td></tr>
      <tr><td>Bölüm</td><td>Bilgisayar Mühendisliği</td></tr>
      <tr><td>Ders</td><td>BLM442 — Büyük Veri Analizine Giriş</td></tr>
      <tr><td>Öğretim Üyesi</td><td>Dr. Ayşe Gül Eker</td></tr>
      <tr><td>Dönem</td><td>2025–2026 Bahar</td></tr>
    </table>
  </div>
  <h3 style="color:white; margin-top:20px;">Takım</h3>
  <div class="student-table">
    <table>
      <tr><td>Emre Aytaş</td><td>220202098</td></tr>
      <tr><td>Hatice Kübra Kılıçaslan</td><td>220202077</td></tr>
      <tr><td>Berker Yiğit</td><td>220202046</td></tr>
      <tr><td>Mertcan Kuzey</td><td>240202009</td></tr>
    </table>
  </div>
</div>

<h2>1. Projenin Amacı</h2>
<p>Gerçek dünya senaryosuna uygun, <b>uçtan uca konteynerize edilmiş bir büyük veri pipeline'ı</b> kurulmuştur:
<b>Kafka → Spark Structured Streaming → Delta Lake → Spark MLlib → MLflow → Streamlit</b>.
Veri seti olarak <a href="https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2">Chicago Crimes 2001 – Present</a>
(~7.9 milyon olay) kullanılmış, ML görevi olarak <b>çoklu sınıf "Primary Type" (suç tipi) sınıflandırma</b>
yapılmıştır (PDF metnindeki "suç tipi ve bölge tahmini").</p>
""")

    # KPI cards
    if eda_card:
        html_parts.append(f"""
<h2>2. Veri Özeti (EDA — Adım 4)</h2>
<div class="kpi">
  <div class="card"><div class="label">Toplam silver satır</div>
    <div class="value">{eda_card['total_rows']}</div></div>
  <div class="card"><div class="label">Benzersiz suç tipi</div>
    <div class="value">{eda_card['unique_types']}</div></div>
  <div class="card"><div class="label">Benzersiz ilçe</div>
    <div class="value">{eda_card['unique_districts']}</div></div>
  <div class="card"><div class="label">Tutuklama oranı</div>
    <div class="value">{eda_card['arrest_rate']}</div></div>
  <div class="card"><div class="label">Domestic oranı</div>
    <div class="value">{eda_card['domestic_rate']}</div></div>
  <div class="card"><div class="label">Yıl aralığı</div>
    <div class="value">{eda_card['year_range']}</div></div>
</div>
""")

    html_parts.append("""
<h2>3. Pipeline Mimarisi</h2>
<pre style="background:#f5f5f5; padding:15px; border-radius:6px; overflow-x:auto;">
Crimes.csv ──▶ crime_producer ──Kafka──▶ Spark Structured Streaming
                                              │
                                              ▼
                       Delta Lake: Bronze (raw event store)
                                              │
                                              ▼
                           Silver (dedupe + null clean + türetilmiş özellikler)
                                              │
                       ┌──────────────────────┼──────────────────────┐
                       ▼                      ▼                      ▼
                Gold tabloları        EDA notebook          5 ML modeli
                (type/district/         (Adım 4)            (LogReg/DT/RF/
                 hourly/feature                              GBT-OvR/NB)
                 _view)                                              │
                                                                     ▼
                                                              MLflow registry
                                                                     │
                                                                     ▼
                                                                Inference →
                                                                  Delta
                                                                     │
                                                                     ▼
                                                          Streamlit Dashboard
</pre>

<h3>Adım Eşleştirmesi (PDF metniyle birebir)</h3>
<table class="metric-table">
<tr><th>PDF Adım</th><th>Bu projedeki karşılığı</th></tr>
<tr><td>Adım 1 — Docker</td><td><code>docker-compose.yml</code> + 4 custom Dockerfile (kafka KRaft, spark-master/worker 8 core/6GB, mlflow, producer, pipeline, dashboard)</td></tr>
<tr><td>Adım 2 — Kafka producer</td><td><code>producer/crime_producer.py</code>: CSV → JSON → topic, 3 hız modu (fixed/speedup/burst)</td></tr>
<tr><td>Adım 3 — Spark Streaming + Delta</td><td><code>bronze_ingest.py</code> (Kafka→Delta append, partitionBy event_date) + <code>silver_clean.py</code> (dedup + null filtre + 4 türetilmiş özellik)</td></tr>
<tr><td>Adım 4 — EDA</td><td><code>spark/notebooks/01_eda.py</code>: total/unique sayımları, yıllık/saatlik/haftalık trend, null analizi</td></tr>
<tr><td>Adım 5 — Feature Engineering</td><td><code>02_feature_engineering.py</code>: 13 özellik (4 zaman, 4 konum, 2 koordinat, 2 bağlam, 2 türetilmiş bool)</td></tr>
<tr><td><b>Adım 6 — 5 ML modeli + MLflow</b></td><td><code>spark/ml/train_models.py</code>: LR + DT + RF + GBT(OvR) + NB. Her run için <b>Feature Importance + Confusion Matrix + per-class CSV</b> + model registry</td></tr>
<tr><td>Adım 7 — Dashboard + görseller</td><td><code>dashboard/app.py</code> 5 sekme + bu HTML rapor (aşağıdaki grafikler)</td></tr>
</table>
""")

    # 5 model comparison
    html_parts.append("<h2>4. Adım 6 — Beş ML Modeli Sonuçları</h2>")
    if not runs.empty:
        f1_val = runs.iloc[0].get('weighted_f1')
        f1_str = f"{f1_val:.4f}" if pd.notna(f1_val) else "—"
        html_parts.append(f"""
<div class="winner">
🏆 <b>En iyi model:</b> {best_model} — accuracy <b>{best_acc:.4f}</b>, weighted F1 {f1_str}
</div>
""")
    html_parts.append(models_table(runs))
    html_parts.append("<h3>Görsel karşılaştırmalar (PDF zorunlu görseller)</h3>")
    html_parts.append(chart_models_comparison(runs))
    html_parts.append("<h3>AUC-ROC (OvR macro)</h3>")
    html_parts.append(chart_auc(runs))
    html_parts.append("<h3>Eğitim süresi karşılaştırması</h3>")
    html_parts.append(chart_train_time(runs))

    # Feature importance
    html_parts.append("<h2>5. Feature Importance (PDF zorunlu)</h2>")
    html_parts.append(chart_feature_importance(runs))

    # Confusion matrix
    html_parts.append("<h2>6. Confusion Matrix (PDF zorunlu — en iyi model)</h2>")
    html_parts.append(chart_confusion_matrix(runs))

    # ROC proxy
    html_parts.append("<h2>7. Per-class Precision/Recall (PDF zorunlu — ROC proxy)</h2>")
    html_parts.append(chart_per_class_pr(runs))

    # EDA charts
    html_parts.append("<h2>8. Zaman Serisi Analizi (PDF zorunlu)</h2>")
    html_parts.append(chart_yearly_trend(eda))
    html_parts.append(chart_hourly_trend(eda))
    html_parts.append(chart_weekly_trend(eda))

    # Distributions
    html_parts.append("<h2>9. Dağılım Analizi (PDF zorunlu — histogram + pie)</h2>")
    html_parts.append(chart_top_types(eda))
    html_parts.append(chart_arrest_pie(eda))

    # Karşılaşılan zorluklar
    html_parts.append("""
<h2>10. Karşılaşılan Zorluklar ve Çözümleri</h2>
<table class="metric-table">
<tr><th>Zorluk</th><th>Kök neden</th><th>Çözüm</th></tr>
<tr>
  <td>Bronze 5M kayıt tek dev batch'te commit etmiyordu</td>
  <td><code>startingOffsets=earliest</code> + 2 core cluster + sınırsız batch</td>
  <td><code>maxOffsetsPerTrigger=200000</code> + worker 2→8 core upgrade</td>
</tr>
<tr>
  <td>Silver shuffle + partition fan-out 67K küçük parquet</td>
  <td><code>partitionBy(event_date)</code> × 200 shuffle partition</td>
  <td><code>partitionBy(event_year)</code> + <code>optimize_silver.py</code> (Delta OPTIMIZE)</td>
</tr>
<tr>
  <td>Dashboard count_rows() askıda kalıyor</td>
  <td>38K parquet dosyasının tamamını taramaya çalışıyor</td>
  <td>Delta transaction log'undan <code>add_actions.num_records.sum()</code></td>
</tr>
<tr>
  <td>GBT Spark MLlib'de multi-class yok</td>
  <td>Sadece binary classifier</td>
  <td><code>OneVsRest(GBTClassifier)</code> wrapper</td>
</tr>
<tr>
  <td>NaiveBayes multinomial negatif değerlerle çöküyor</td>
  <td>Chicago longitude ≈ -87 (negatif)</td>
  <td><code>NaiveBayes(modelType="gaussian")</code></td>
</tr>
<tr>
  <td>multi_class_auc UNRESOLVED_ROUTINE</td>
  <td><code>F.expr("vector_to_array(...)")</code> SQL function register edilmemiş</td>
  <td><code>pyspark.ml.functions.vector_to_array</code> Python import</td>
</tr>
<tr>
  <td>Veri seti pivot (MovieLens → Chicago Crimes)</td>
  <td>Form/onay sonucu seçim değişti</td>
  <td><code>feat/chicago-crimes</code> branch'inde sıfırdan tam rewrite</td>
</tr>
</table>

<h2>11. Tekrarlanabilirlik</h2>
<pre style="background:#f5f5f5; padding:15px; border-radius:6px;">
git clone https://github.com/bigdataKOU/bigdataproje.git
cd bigdataproje
cp .env.example .env

# Crimes.csv'i ../crimes/Crimes.csv olarak yerleştir
mkdir -p ../crimes

make verify        # statik kontrol
make run-all       # uçtan uca (~30-50 dk)
make report        # bu HTML raporu üret

# Eriş
# Dashboard: http://localhost:8501
# MLflow:    http://localhost:5000
# Spark UI:  http://localhost:8080
</pre>

<h2>12. Teslim Kontrolü (PDF Madde 5 — Değerlendirme Kriterleri)</h2>
<table class="metric-table">
<tr><th>Kriter</th><th>Ağırlık</th><th>Durum</th></tr>
<tr><td>Docker & Altyapı</td><td>%15</td><td>✓ docker-compose.yml + 4 Dockerfile, 7 servis ayağa kalkıyor</td></tr>
<tr><td>Kafka Streaming</td><td>%15</td><td>✓ Producer (3 hız modu) + Bronze Spark Structured Streaming</td></tr>
<tr><td>Spark + Delta Lake</td><td>%15</td><td>✓ Bronze/Silver/Gold medallion, partition + OPTIMIZE</td></tr>
<tr><td>EDA & Feature Engineering</td><td>%10</td><td>✓ 01_eda.py + 02_feature_engineering.py (13 özellik)</td></tr>
<tr><td>ML & MLflow</td><td>%15</td><td>✓ 5 model + Feature Importance + Confusion Matrix + AUC + model registry</td></tr>
<tr><td>Dashboard & Görselleştirme</td><td>%15</td><td>✓ Streamlit 5 sekme + bu HTML rapor (7+ görsel)</td></tr>
<tr><td>Dokümantasyon & Sunum</td><td>%15</td><td>✓ README + teknik_rapor.md + docs/sunum.md + bu HTML</td></tr>
</table>

<div class="footer">
  <div class="thanks">Teşekkürler 🎓</div>
  <p style="margin-top:10px;">
    Sunumlarımız: 14 Mayıs / 21 Mayıs / 4 Haziran / 11 Haziran 2026<br>
    Bu rapor otomatik olarak <code>scripts/make_report.py</code> ile MLflow + Delta sonuçlarından üretilmiştir.
  </p>
</div>

</body>
</html>
""")

    return "".join(html_parts)


def main() -> int:
    html = build_html()
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"wrote {OUT_HTML} ({os.path.getsize(OUT_HTML):,} bytes)")
    return 0


if __name__ == "__main__":
    main()
