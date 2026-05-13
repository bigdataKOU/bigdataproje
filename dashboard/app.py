"""Streamlit dashboard: Chicago Crimes pipeline + classifier demo."""
import os
import logging
from pathlib import Path

import mlflow
import pandas as pd
import plotly.express as px
import streamlit as st
from deltalake import DeltaTable

# Logging ayarları
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Sabitler ve Yollar
DELTA_PATH = Path(os.environ.get("DELTA_PATH", "/opt/delta"))
PATHS = {
    "bronze": DELTA_PATH / "bronze" / "crimes",
    "silver": DELTA_PATH / "silver" / "crimes",
    "gold_type": DELTA_PATH / "gold" / "type_stats",
    "gold_district": DELTA_PATH / "gold" / "district_stats",
    "gold_hourly": DELTA_PATH / "gold" / "hourly_stats",
    "predictions": DELTA_PATH / "gold" / "predictions",
}
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")

st.set_page_config(page_title="Chicago Crimes Pipeline", layout="wide")

# Yardımcı Fonksiyonlar (Hata Yönetimli)
@st.cache_data(ttl=30)
def read_delta(path: str):
    try:
        if not Path(path).exists():
            return None
        return DeltaTable(path).to_pandas()
    except Exception as e:
        logger.error(f"Delta tablosu okunurken hata ({path}): {e}")
        return None

@st.cache_data(ttl=60)
def read_delta_count(path: str):
    try:
        if not Path(path).exists():
            return None
        return DeltaTable(path).to_pyarrow_dataset().count_rows()
    except Exception as e:
        logger.error(f"Satır sayısı hesaplanırken hata ({path}): {e}")
        return None

def safe_plotly_chart(fig, caption=None):
    """Grafiklerin boş veriyle çökmesini engeller."""
    try:
        st.plotly_chart(fig, use_container_width=True)
        if caption:
            st.caption(caption)
    except Exception as e:
        st.error(f"Grafik çizilemedi: {e}")

# Başlık
st.title("Chicago Crimes — Big Data Pipeline")
st.caption("Kafka → Spark Structured Streaming → Delta Lake → Random Forest → MLflow")

# --- KPI METRICS ---
col1, col2, col3, col4 = st.columns(4)
try:
    bronze_n = read_delta_count(str(PATHS["bronze"]))
    silver_n = read_delta_count(str(PATHS["silver"]))
    types_n = read_delta_count(str(PATHS["gold_type"]))
    districts_n = read_delta_count(str(PATHS["gold_district"]))

    col1.metric("Bronze events", f"{bronze_n:,}" if bronze_n is not None else "—")
    col2.metric("Silver events", f"{silver_n:,}" if silver_n is not None else "—")
    col3.metric("Suç tipi (gold)", f"{types_n:,}" if types_n is not None else "—")
    col4.metric("İlçe (district)", f"{districts_n:,}" if districts_n is not None else "—")
except Exception as e:
    st.warning(f"Metrikler yüklenirken bir sorun oluştu: {e}")

# --- MLFLOW SUMMARY ---
@st.cache_data(ttl=60)
def best_run_summary():
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = mlflow.tracking.MlflowClient()
        exp_name = os.environ.get("MLFLOW_EXPERIMENT", "chicago-crimes-classifier")
        exp = client.get_experiment_by_name(exp_name)
        
        if not exp: return None
        
        runs = client.search_runs(exp.experiment_id, max_results=50)
        if not runs: return None

        valid_runs = [r for r in runs if "accuracy" in r.data.metrics]
        if not valid_runs: return None
        
        best_run = max(valid_runs, key=lambda r: r.data.metrics["accuracy"])
        
        return {
            "accuracy": best_run.data.metrics.get("accuracy"),
            "weighted_f1": best_run.data.metrics.get("weighted_f1"),
            "name": best_run.data.tags.get("mlflow.runName", best_run.info.run_id[:8]),
            "numTrees": best_run.data.params.get("numTrees", "N/A"),
            "maxDepth": best_run.data.params.get("maxDepth", "N/A"),
            "train_seconds": best_run.data.metrics.get("train_seconds"),
        }
    except Exception as e:
        logger.error(f"MLflow best run hatası: {e}")
        return None

best = best_run_summary()
if best:
    st.success(
        f"🏆 En iyi model: **{best['name']}** — "
        f"Accuracy: **{best['accuracy']:.4f}** | "
        f"F1: {best.get('weighted_f1', 0):.4f} | "
        f"Trees: {best['numTrees']} | Depth: {best['maxDepth']}"
    )

# --- TABS ---
tab_overview, tab_district, tab_hourly, tab_preds, tab_mlflow = st.tabs(
    ["📊 Genel", "🗺️ İlçe", "🕒 Saat", "🔮 Tahminler", "🧪 MLflow"]
)

with tab_overview:
    st.subheader("Suç tipi başına toplam (top 20)")
    types = read_delta(str(PATHS["gold_type"]))
    if types is not None and not types.empty:
        try:
            top = types.nlargest(20, "crime_count")
            fig = px.bar(
                top.sort_values("crime_count"),
                x="crime_count", y="primary_type",
                orientation="h", color="arrest_rate",
                color_continuous_scale="RdYlGn_r"
            )
            safe_plotly_chart(fig, "Renk = tutuklama oranı. Çubuk uzunluğu = suç sayısı.")
        except Exception as e:
            st.error(f"Görselleştirme hatası: {e}")
    else:
        st.info("Veri bulunamadı. Lütfen 'gold_features.py' çalıştırın.")

with tab_district:
    districts = read_delta(str(PATHS["gold_district"]))
    if districts is not None and not districts.empty:
        try:
            st.subheader("İlçe Haritası")
            geo_df = districts.dropna(subset=["avg_latitude", "avg_longitude"])
            if not geo_df.empty:
                fig_map = px.scatter_mapbox(
                    geo_df, lat="avg_latitude", lon="avg_longitude",
                    size="crime_count", color="arrest_rate",
                    zoom=9, mapbox_style="open-street-map", height=500
                )
                safe_plotly_chart(fig_map)
            else:
                st.warning("Harita için koordinat verisi bulunamadı.")
        except Exception as e:
            st.error(f"Harita yüklenirken hata: {e}")
    else:
        st.info("İlçe verisi henüz işlenmemiş.")

with tab_hourly:
    hourly = read_delta(str(PATHS["gold_hourly"]))
    if hourly is not None and not hourly.empty:
        try:
            st.subheader("Saatlik Yoğunluk")
            pivot = hourly.pivot_table(index="primary_type", columns="hour_of_day", values="crime_count", fill_value=0)
            fig_heat = px.imshow(pivot, aspect="auto", color_continuous_scale="Viridis")
            safe_plotly_chart(fig_heat)
        except Exception as e:
            st.error(f"Heatmap oluşturulamadı: {e}")
    else:
        st.info("Saatlik istatistikler bulunamadı.")

with tab_preds:
    preds = read_delta(str(PATHS["predictions"]))
    if preds is not None and not preds.empty:
        try:
            st.subheader("Model Tahmin Analizi")
            # Örnekleme hatasını önlemek için kontrol
            sample_size = min(1000, len(preds))
            sample = preds.sample(sample_size)
            
            acc = (sample["actual_primary_type"] == sample["predicted_label"]).mean()
            st.metric("Test Doğruluğu (Örneklem)", f"{acc:.2%}")
            
            cm = pd.crosstab(sample["actual_primary_type"], sample["predicted_label"])
            safe_plotly_chart(px.imshow(cm, text_auto=True), "Karışıklık Matrisi")
        except KeyError as e:
            st.error(f"Beklenen kolonlar bulunamadı: {e}")
        except Exception as e:
            st.error(f"Tahmin sekmesinde hata: {e}")
    else:
        st.info("Henüz tahmin verisi yok.")

with tab_mlflow:
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = mlflow.tracking.MlflowClient()
        exp = client.get_experiment_by_name(os.environ.get("MLFLOW_EXPERIMENT", "chicago-crimes-classifier"))
        
        if exp:
            runs = client.search_runs(exp.experiment_id)
            if runs:
                # Veriyi DataFrame'e dönüştürürken hata yönetimi
                run_data = []
                for r in runs:
                    try:
                        run_data.append({
                            "Run ID": r.info.run_id[:8],
                            "Accuracy": r.data.metrics.get("accuracy"),
                            "Trees": r.data.params.get("numTrees"),
                            "Depth": r.data.params.get("maxDepth"),
                        })
                    except: continue
                st.dataframe(pd.DataFrame(run_data).dropna(subset=["Accuracy"]))
            else:
                st.info("Kayıtlı run bulunamadı.")
        else:
            st.warning("MLflow deneyi bulunamadı.")
    except Exception as e:
        st.error(f"MLflow sunucusuna bağlanılamadı: {e}")

st.divider()
st.caption("Chicago Crimes Dashboard | Error Handling Enabled | Spark 3.5 & Delta Lake")