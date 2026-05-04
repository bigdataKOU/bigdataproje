"""Streamlit dashboard: pipeline saglik durumu + ALS oneri demosu."""
import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from deltalake import DeltaTable
from deltalake.exceptions import TableNotFoundError

import mlflow

DELTA_PATH = Path(os.environ.get("DELTA_PATH", "/opt/delta"))
RECS_PATH = DELTA_PATH / "gold" / "user_recommendations"
GOLD_MOVIE_PATH = DELTA_PATH / "gold" / "movie_stats"
GOLD_USER_PATH = DELTA_PATH / "gold" / "user_stats"
BRONZE_PATH = DELTA_PATH / "bronze" / "ratings"
SILVER_PATH = DELTA_PATH / "silver" / "ratings"
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")

st.set_page_config(page_title="MovieLens Pipeline", layout="wide")
st.title("MovieLens Big Data Pipeline")
st.caption("Kafka → Spark Streaming → Delta Lake → ALS → MLflow")


@st.cache_data(ttl=30)
def read_delta(path: str) -> pd.DataFrame | None:
    try:
        return DeltaTable(path).to_pandas()
    except (TableNotFoundError, FileNotFoundError, Exception):
        return None


@st.cache_data(ttl=60)
def read_delta_count(path: str) -> int | None:
    try:
        return DeltaTable(path).to_pyarrow_dataset().count_rows()
    except Exception:
        return None


col1, col2, col3, col4 = st.columns(4)
bronze_n = read_delta_count(str(BRONZE_PATH))
silver_n = read_delta_count(str(SILVER_PATH))
movies_n = read_delta_count(str(GOLD_MOVIE_PATH))
users_n = read_delta_count(str(GOLD_USER_PATH))
col1.metric("Bronze rows", f"{bronze_n:,}" if bronze_n is not None else "—")
col2.metric("Silver rows", f"{silver_n:,}" if silver_n is not None else "—")
col3.metric("Movies (gold)", f"{movies_n:,}" if movies_n is not None else "—")
col4.metric("Users (gold)", f"{users_n:,}" if users_n is not None else "—")

tab_overview, tab_recs, tab_mlflow = st.tabs(
    ["📊 Genel", "🎬 Öneriler", "🧪 MLflow"]
)

with tab_overview:
    st.subheader("Top 20 film (rating sayısı)")
    movies = read_delta(str(GOLD_MOVIE_PATH))
    if movies is not None and not movies.empty:
        top = movies.nlargest(20, "rating_count")
        fig = px.bar(
            top,
            x="rating_count", y="title",
            orientation="h",
            color="avg_rating",
            color_continuous_scale="Viridis",
            hover_data=["genres", "popularity_bucket"],
        )
        fig.update_layout(height=600, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Henüz gold tablosu yok — `gold_features.py` çalıştırılmalı.")

    st.subheader("Aktivite kovaları (kullanıcılar)")
    users = read_delta(str(GOLD_USER_PATH))
    if users is not None and not users.empty:
        bucket_counts = users["activity_bucket"].value_counts().reset_index()
        bucket_counts.columns = ["bucket", "users"]
        st.plotly_chart(px.pie(bucket_counts, names="bucket", values="users",
                               hole=0.4),
                        use_container_width=True)


with tab_recs:
    st.subheader("Kullanıcıya özel top-N öneri")
    recs = read_delta(str(RECS_PATH))
    if recs is None or recs.empty:
        st.info("Henüz öneri tablosu yok — `inference.py` çalıştırılmalı.")
    else:
        users_avail = sorted(recs["userId"].unique())[:200]
        chosen = st.selectbox("userId seç", users_avail)
        sub = (recs[recs["userId"] == chosen]
                  .sort_values("rank")
                  .head(20))
        st.dataframe(
            sub[["rank", "title", "genres", "predicted_rating",
                 "avg_rating", "rating_count"]],
            use_container_width=True,
            hide_index=True,
        )


with tab_mlflow:
    st.subheader("MLflow run karşılaştırması")
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = mlflow.tracking.MlflowClient()
        exp = client.get_experiment_by_name(
            os.environ.get("MLFLOW_EXPERIMENT", "movielens-als")
        )
        if exp is None:
            st.info("Henüz deney yok.")
        else:
            runs = client.search_runs(
                exp.experiment_id,
                order_by=["attributes.start_time DESC"],
                max_results=20,
            )
            rows = []
            for r in runs:
                rows.append({
                    "run_id": r.info.run_id[:8],
                    "rmse": r.data.metrics.get("rmse"),
                    "mae": r.data.metrics.get("mae"),
                    "rank": r.data.params.get("rank"),
                    "regParam": r.data.params.get("regParam"),
                    "maxIter": r.data.params.get("maxIter"),
                    "n_ratings": r.data.params.get("n_ratings"),
                    "train_seconds": r.data.metrics.get("train_seconds"),
                })
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)
            if not df.empty and df["rmse"].notna().any():
                st.plotly_chart(
                    px.scatter(df.dropna(subset=["rmse"]),
                               x="rank", y="rmse",
                               size="n_ratings", color="regParam",
                               hover_data=["run_id"]),
                    use_container_width=True,
                )
    except Exception as exc:
        st.error(f"MLflow erişilemiyor: {exc}")

st.divider()
st.caption("bigdataKOU/bigdataproje · Spark 3.5 · Delta 3.2 · MLflow 2.16")
