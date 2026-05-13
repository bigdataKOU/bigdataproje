"""Streamlit dashboard: Chicago Crimes pipeline + classifier demo."""
import os
from pathlib import Path

import mlflow
import pandas as pd
import plotly.express as px
import streamlit as st
from deltalake import DeltaTable

DELTA_PATH = Path(os.environ.get("DELTA_PATH", "/opt/delta"))
BRONZE_PATH = DELTA_PATH / "bronze" / "crimes"
SILVER_PATH = DELTA_PATH / "silver" / "crimes"
GOLD_TYPE_PATH = DELTA_PATH / "gold" / "type_stats"
GOLD_DISTRICT_PATH = DELTA_PATH / "gold" / "district_stats"
GOLD_HOURLY_PATH = DELTA_PATH / "gold" / "hourly_stats"
PREDICTIONS_PATH = DELTA_PATH / "gold" / "predictions"
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")

st.set_page_config(page_title="Chicago Crimes Pipeline", layout="wide")
st.title("Chicago Crimes — Big Data Pipeline")
st.caption("Kafka → Spark Structured Streaming → Delta Lake → Random Forest → MLflow")


@st.cache_data(ttl=30)
def read_delta(path: str):
    try:
        return DeltaTable(path).to_pandas()
    except Exception:
        return None


@st.cache_data(ttl=60)
def read_delta_count(path: str):
    """Hızlı count — Delta transaction log'undan add actions' num_records sum.
    pyarrow_dataset().count_rows() tüm parquet dosyalarını açar; bronze'da
    binlerce küçük dosya varsa dakikalar alabilir."""
    try:
        dt = DeltaTable(path)
        try:
            adds = dt.get_add_actions(flatten=True).to_pandas()
            if "num_records" in adds.columns and adds["num_records"].notna().any():
                return int(adds["num_records"].fillna(0).sum())
        except Exception:
            pass
        # fallback: tam tarama (yavaş ama doğru)
        return DeltaTable(path).to_pyarrow_dataset().count_rows()
    except Exception:
        return None


col1, col2, col3, col4 = st.columns(4)
bronze_n = read_delta_count(str(BRONZE_PATH))
silver_n = read_delta_count(str(SILVER_PATH))
types_n = read_delta_count(str(GOLD_TYPE_PATH))
districts_n = read_delta_count(str(GOLD_DISTRICT_PATH))
col1.metric("Bronze events", f"{bronze_n:,}" if bronze_n is not None else "—")
col2.metric("Silver events", f"{silver_n:,}" if silver_n is not None else "—")
col3.metric("Suç tipi (gold)", f"{types_n:,}" if types_n is not None else "—")
col4.metric("İlçe (district)", f"{districts_n:,}" if districts_n is not None else "—")


@st.cache_data(ttl=60)
def best_run_summary():
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = mlflow.tracking.MlflowClient()
        exp = client.get_experiment_by_name(
            os.environ.get("MLFLOW_EXPERIMENT", "chicago-crimes-classifier")
        )
        if exp is None:
            return None
        runs = client.search_runs(exp.experiment_id, max_results=50)
        best = None
        for r in runs:
            acc = r.data.metrics.get("accuracy")
            if acc is None:
                continue
            if best is None or acc > best[0]:
                best = (acc, r)
        if best is None:
            return None
        r = best[1]
        return {
            "accuracy": best[0],
            "weighted_f1": r.data.metrics.get("weighted_f1"),
            "name": r.data.tags.get("mlflow.runName", r.info.run_id[:8]),
            "numTrees": r.data.params.get("numTrees"),
            "maxDepth": r.data.params.get("maxDepth"),
            "train_seconds": r.data.metrics.get("train_seconds"),
        }
    except Exception:
        return None


best = best_run_summary()
if best:
    f1_str = f"{best['weighted_f1']:.4f}" if best.get('weighted_f1') is not None else "—"
    ts_str = f"{best['train_seconds']:.0f}s" if best.get('train_seconds') is not None else "—"
    st.success(
        f"🏆 En iyi model: **{best['name']}** — accuracy = **{best['accuracy']:.4f}**, "
        f"weighted F1 = {f1_str}, "
        f"numTrees={best['numTrees']}, maxDepth={best['maxDepth']}, "
        f"eğitim={ts_str}"
    )


tab_overview, tab_district, tab_hourly, tab_preds, tab_mlflow = st.tabs(
    ["📊 Genel", "🗺️ İlçe", "🕒 Saat", "🔮 Tahminler", "🧪 MLflow"]
)

with tab_overview:
    st.subheader("Suç tipi başına toplam (top 20)")
    types = read_delta(str(GOLD_TYPE_PATH))
    if types is not None and not types.empty:
        top = types.nlargest(20, "crime_count")
        fig = px.bar(
            top.sort_values("crime_count"),
            x="crime_count", y="primary_type",
            orientation="h",
            color="arrest_rate",
            color_continuous_scale="RdYlGn_r",
            hover_data=["domestic_rate", "frequency_bucket"],
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Renk = tutuklama oranı (yüksek=daha çok yakalama). "
            "Çubuk uzunluğu = toplam suç sayısı."
        )
    else:
        st.info("Henüz gold/type_stats yok — `gold_features.py` çalıştırılmalı.")

    st.subheader("Tutuklama oranı dağılımı (frekansa göre)")
    if types is not None and not types.empty:
        fig2 = px.scatter(
            types,
            x="crime_count", y="arrest_rate",
            size="crime_count", color="frequency_bucket",
            hover_name="primary_type",
            log_x=True,
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.caption(
            "X ekseni log ölçek. Yüksek frekanslı suç tiplerinde tutuklama oranı "
            "genellikle daha düşük (örn. theft) — düşük frekanslı bazıları daha yüksek arrest_rate'e sahip olabilir."
        )


with tab_district:
    st.subheader("İlçe bazında suç sayısı + en sık suç tipi")
    districts = read_delta(str(GOLD_DISTRICT_PATH))
    if districts is not None and not districts.empty:
        fig = px.bar(
            districts.sort_values("crime_count"),
            x="crime_count", y=districts["district"].astype(str),
            orientation="h",
            color="top_primary_type",
            hover_data=["unique_types", "arrest_rate", "size_bucket"],
            title="İlçe başına toplam (top suç tipi renkle)",
        )
        fig.update_layout(height=600, yaxis_title="district")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("İlçe haritası (avg lat/lon)")
        st.plotly_chart(
            px.scatter_mapbox(
                districts.dropna(subset=["avg_latitude", "avg_longitude"]),
                lat="avg_latitude", lon="avg_longitude",
                size="crime_count", color="arrest_rate",
                hover_name="district",
                hover_data=["top_primary_type", "unique_types"],
                zoom=9, height=500,
                mapbox_style="open-street-map",
                color_continuous_scale="RdYlGn_r",
            ),
            use_container_width=True,
        )
        st.caption(
            "Daire büyüklüğü = suç sayısı, renk = tutuklama oranı (kırmızı=düşük). "
            "Konum = ilçe ortalama lat/lon."
        )
    else:
        st.info("Henüz gold/district_stats yok.")


with tab_hourly:
    st.subheader("Saat × Suç tipi heatmap")
    hourly = read_delta(str(GOLD_HOURLY_PATH))
    if hourly is not None and not hourly.empty:
        top_types = (
            hourly.groupby("primary_type")["crime_count"].sum()
            .nlargest(10).index.tolist()
        )
        sub = hourly[hourly["primary_type"].isin(top_types)]
        pivot = sub.pivot_table(
            index="primary_type", columns="hour_of_day",
            values="crime_count", fill_value=0,
        )
        st.plotly_chart(
            px.imshow(
                pivot.reindex(top_types),
                labels=dict(x="hour_of_day", y="primary_type", color="count"),
                aspect="auto",
                color_continuous_scale="Viridis",
            ),
            use_container_width=True,
        )
        st.caption(
            "Top 10 suç tipi × günün saati. Theft genellikle öğleden sonra zirve "
            "yapar, battery ise gece yarısı civarı."
        )

        st.subheader("Saatlik toplam suç sayısı")
        hourly_total = (
            hourly.groupby("hour_of_day")["crime_count"].sum().reset_index()
        )
        st.plotly_chart(
            px.line(hourly_total, x="hour_of_day", y="crime_count", markers=True),
            use_container_width=True,
        )
    else:
        st.info("Henüz gold/hourly_stats yok.")


with tab_preds:
    st.subheader("Random Forest tahminleri")
    preds = read_delta(str(PREDICTIONS_PATH))
    if preds is None or preds.empty:
        st.info("Henüz tahmin yok — `inference.py` çalıştırılmalı.")
    else:
        sample = preds.head(1000)
        match = (sample["actual_label"] == sample["predicted_label"]).mean()
        st.metric("Örnek (1K satır) doğruluk", f"{match:.1%}")

        col_a, col_b = st.columns(2)
        with col_a:
            actual_top = sample["actual_label"].value_counts().head(10)
            st.markdown("**Gerçek dağılım (top 10)**")
            st.bar_chart(actual_top)
        with col_b:
            pred_top = sample["predicted_label"].value_counts().head(10)
            st.markdown("**Tahmin dağılımı (top 10)**")
            st.bar_chart(pred_top)

        st.subheader("Karışıklık matrisi (örnek)")
        cm = pd.crosstab(
            sample["actual_label"],
            sample["predicted_label"],
        )
        st.plotly_chart(
            px.imshow(
                cm.values,
                x=list(cm.columns), y=list(cm.index),
                color_continuous_scale="Blues",
                labels=dict(x="tahmin", y="gerçek"),
                aspect="auto",
            ),
            use_container_width=True,
        )


with tab_mlflow:
    st.subheader("MLflow run karşılaştırması")
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = mlflow.tracking.MlflowClient()
        exp = client.get_experiment_by_name(
            os.environ.get("MLFLOW_EXPERIMENT", "chicago-crimes-classifier")
        )
        if exp is None:
            st.info("Henüz deney yok.")
        else:
            runs = client.search_runs(
                exp.experiment_id,
                order_by=["attributes.start_time DESC"],
                max_results=50,
            )
            rows = []
            for r in runs:
                rows.append({
                    "run_id": r.info.run_id[:8],
                    "model": r.data.params.get("model_type", "?"),
                    "run_name": r.data.tags.get("mlflow.runName", ""),
                    "numTrees": int(r.data.params.get("numTrees", 0) or 0) or None,
                    "maxDepth": int(r.data.params.get("maxDepth", 0) or 0) or None,
                    "accuracy": r.data.metrics.get("accuracy"),
                    "weighted_f1": r.data.metrics.get("weighted_f1"),
                    "weighted_precision": r.data.metrics.get("weighted_precision"),
                    "weighted_recall": r.data.metrics.get("weighted_recall"),
                    "auc_ovr_macro": r.data.metrics.get("auc_ovr_macro"),
                    "train_seconds": r.data.metrics.get("train_seconds"),
                    "n_rows": int(r.data.params.get("n_rows", 0) or 0) or None,
                })
            df = pd.DataFrame(rows).dropna(subset=["accuracy"])
            df = df.sort_values("accuracy", ascending=False) \
                   .drop_duplicates("model").reset_index(drop=True)
            if df.empty:
                st.info("Henüz tamamlanmış run yok.")
            else:
                # None'ları "—" olarak göster, metrikleri yuvarla
                df_display = df.copy()
                for c in ["accuracy", "weighted_f1", "weighted_precision",
                          "weighted_recall", "auc_ovr_macro"]:
                    df_display[c] = df_display[c].apply(
                        lambda v: f"{v:.4f}" if pd.notna(v) else "—"
                    )
                df_display["train_seconds"] = df_display["train_seconds"].apply(
                    lambda v: f"{v:.1f}s" if pd.notna(v) else "—"
                )
                for c in ["numTrees", "maxDepth", "n_rows"]:
                    df_display[c] = df_display[c].apply(
                        lambda v: f"{int(v):,}" if pd.notna(v) else "—"
                    )
                st.dataframe(df_display, use_container_width=True, hide_index=True)

                # maxDepth/numTrees tüm modellerde yok — boş mode() varsa atla
                _md_mode = df["maxDepth"].dropna().mode()
                tree_sweep = (
                    df[df["maxDepth"] == _md_mode.iloc[0]].sort_values("numTrees")
                    if not _md_mode.empty else pd.DataFrame()
                )
                if len(tree_sweep) >= 2:
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.markdown("**Accuracy ile numTrees**")
                        st.plotly_chart(
                            px.line(tree_sweep, x="numTrees", y="accuracy", markers=True),
                            use_container_width=True,
                        )
                    with col_b:
                        st.markdown("**Eğitim süresi (s) ile numTrees**")
                        st.plotly_chart(
                            px.line(tree_sweep, x="numTrees", y="train_seconds", markers=True),
                            use_container_width=True,
                        )

                _nt_mode = df["numTrees"].dropna().mode()
                depth_sweep = (
                    df[df["numTrees"] == _nt_mode.iloc[0]].sort_values("maxDepth")
                    if not _nt_mode.empty else pd.DataFrame()
                )
                if len(depth_sweep) >= 2:
                    col_c, col_d = st.columns(2)
                    with col_c:
                        st.markdown("**Accuracy ile maxDepth**")
                        st.plotly_chart(
                            px.line(depth_sweep, x="maxDepth", y="accuracy", markers=True),
                            use_container_width=True,
                        )
                    with col_d:
                        st.markdown("**Weighted F1 ile maxDepth**")
                        st.plotly_chart(
                            px.line(depth_sweep, x="maxDepth", y="weighted_f1", markers=True),
                            use_container_width=True,
                        )

                st.markdown("**Pareto: doğruluk vs eğitim süresi**")
                pareto = df.dropna(subset=["accuracy", "train_seconds"]).copy()
                if not pareto.empty:
                    # NaN -> '—' (plotly customdata index gosteriyor yoksa)
                    pareto["numTrees_str"] = pareto["numTrees"].apply(
                        lambda v: f"{int(v)}" if pd.notna(v) else "—"
                    )
                    pareto["maxDepth_str"] = pareto["maxDepth"].apply(
                        lambda v: f"{int(v)}" if pd.notna(v) else "—"
                    )
                    pareto["accuracy_str"] = pareto["accuracy"].apply(
                        lambda v: f"{v:.4f}"
                    )
                    pareto["train_seconds_str"] = pareto["train_seconds"].apply(
                        lambda v: f"{v:.1f}s"
                    )
                    fig = px.scatter(
                        pareto, x="train_seconds", y="accuracy",
                        color="run_name", text="run_name",
                    )
                    fig.update_traces(
                        hovertemplate=(
                            "<b>%{customdata[0]}</b><br>"
                            "accuracy = %{customdata[1]}<br>"
                            "train süresi = %{customdata[2]}<br>"
                            "numTrees = %{customdata[3]}<br>"
                            "maxDepth = %{customdata[4]}<extra></extra>"
                        ),
                        customdata=pareto[[
                            "run_name", "accuracy_str", "train_seconds_str",
                            "numTrees_str", "maxDepth_str",
                        ]].values,
                        textposition="top center",
                        marker=dict(size=14),
                    )
                    fig.update_layout(
                        xaxis_title="Eğitim süresi (saniye)",
                        yaxis_title="Accuracy",
                    )
                    st.plotly_chart(fig, use_container_width=True)
    except Exception as exc:
        st.error(f"MLflow erişilemiyor: {exc}")

st.divider()
st.caption(
    "bigdataKOU/bigdataproje · Chicago Crimes 2001-Present · "
    "Spark 3.5 · Delta 3.2 · MLflow 2.16 · RandomForest"
)
