"""Streamlit dashboard: pipeline saglik durumu + ALS oneri demosu."""
import os
from pathlib import Path

import mlflow
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from deltalake import DeltaTable

# Konfigürasyon
DELTA_PATH = Path(os.environ.get("DELTA_PATH", "/opt/delta"))
RECS_PATH = DELTA_PATH / "gold" / "user_recommendations"
GOLD_MOVIE_PATH = DELTA_PATH / "gold" / "movie_stats"
GOLD_USER_PATH = DELTA_PATH / "gold" / "user_stats"
BRONZE_PATH = DELTA_PATH / "bronze" / "ratings"
SILVER_PATH = DELTA_PATH / "silver" / "ratings"
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")

# Sayfa konfigürasyonu
st.set_page_config(
    page_title="MovieLens Analiz Paneli",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "MovieLens 25M Streaming Veri Mühendisliği Projesi"}
)

# Özel CSS stilleri
st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .main-title {
        text-align: center;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Başlık
st.markdown("# 🎬 MovieLens Analiz Paneli")
st.markdown("""
Kafka → Spark Streaming → Delta Lake → ALS → MLflow
""", help="Tam akış projesi: Gerçek zamanlı veri işleme ve film önerileri")



# Veri okuma fonksiyonları (cache ile)
@st.cache_data(ttl=30)
def read_delta(path: str):
    """Delta Lake tablosunu pandas DataFrame olarak oku."""
    try:
        return DeltaTable(path).to_pandas()
    except Exception as e:
        st.error(f"Veri okunamadı: {str(e)[:100]}")
        return None


@st.cache_data(ttl=60)
def read_delta_count(path: str):
    """Delta Lake tablosundaki satır sayısını oku."""
    try:
        return DeltaTable(path).to_pyarrow_dataset().count_rows()
    except Exception:
        return None


# Pipeline durumu - KPI kartları
st.markdown("### 📊 Pipeline Durumu")
col1, col2, col3, col4 = st.columns(4)

bronze_n = read_delta_count(str(BRONZE_PATH))
silver_n = read_delta_count(str(SILVER_PATH))
movies_n = read_delta_count(str(GOLD_MOVIE_PATH))
users_n = read_delta_count(str(GOLD_USER_PATH))

with col1:
    st.metric(
        "🔵 Bronze (Ham)",
        f"{bronze_n:,}" if bronze_n is not None else "—",
        help="Kafka'dan alınan ham rating verisi"
    )
with col2:
    st.metric(
        "🟢 Silver (Temiz)",
        f"{silver_n:,}" if silver_n is not None else "—",
        help="Temizlenmiş ve dörüştürülmüş veriler"
    )
with col3:
    st.metric(
        "⭐ Filmler (Gold)",
        f"{movies_n:,}" if movies_n is not None else "—",
        help="Film istatistikleri ve metrikleri"
    )
with col4:
    st.metric(
        "👥 Kullanıcılar (Gold)",
        f"{users_n:,}" if users_n is not None else "—",
        help="Kullanıcı aktivite metrikleri"
    )



# Sekme tabanlı içerik
tab_overview, tab_recs, tab_mlflow = st.tabs(
    ["� Genel Analiz", "� Film Önerileri", "🧪 Model Performansı"]
)

# TAB 1: Genel Analiz
with tab_overview:
    st.markdown("### En İyi Filmler (Rating Sayısı)")
    
    movies = read_delta(str(GOLD_MOVIE_PATH))
    if movies is not None and not movies.empty:
        # Top 20 filmler
        top = movies.nlargest(20, "rating_count")
        
        # Geliştirilmiş bar chart
        fig = px.bar(
            top,
            x="rating_count",
            y="title",
            orientation="h",
            color="avg_rating",
            color_continuous_scale="Viridis",
            hover_data={
                "rating_count": ":,",
                "avg_rating": ":.2f",
                "genres": True,
                "title": False
            },
            labels={
                "rating_count": "Rating Sayısı",
                "avg_rating": "Ortalama Puan"
            }
        )
        fig.update_layout(
            height=500,
            yaxis={"categoryorder": "total ascending"},
            showlegend=True,
            hovermode="closest"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("💡 Henüz gold tablosu yok. Spark jobs'u çalıştırmak için `docker-compose up` kullanın.")

    # Kullanıcı aktivitesi
    st.markdown("### Kullanıcı Aktivite Dağılımı")
    
    users = read_delta(str(GOLD_USER_PATH))
    if users is not None and not users.empty:
        bucket_counts = users["activity_bucket"].value_counts().reset_index()
        bucket_counts.columns = ["Aktivite Seviyesi", "Kullanıcı Sayısı"]
        
        # Geliştirilmiş pie chart
        fig = px.pie(
            bucket_counts,
            names="Aktivite Seviyesi",
            values="Kullanıcı Sayısı",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Set3,
            hover_data={"Kullanıcı Sayısı": ":,"}
        )
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("💡 Henüz kullanıcı aktivite verisi yok.")


# TAB 2: Film Önerileri
with tab_recs:
    st.markdown("### Kullanıcıya Özel Film Önerileri")
    
    recs = read_delta(str(RECS_PATH))
    if recs is None or recs.empty:
        st.warning("💡 Henüz öneri tablosu yok. Model eğitimi için `train_als.py` ve inference için `inference.py` çalıştırmalısınız.")
    else:
        users_avail = sorted(recs["userId"].unique())[:200]
        
        col_user, col_info = st.columns([3, 2])
        
        with col_user:
            chosen = st.selectbox(
                "👤 Kullanıcı Seç",
                users_avail,
                help="Önerileri görmek istediğiniz kullanıcıyı seçin"
            )
        
        # Seçilen kullanıcının önerileri
        sub = (recs[recs["userId"] == chosen]
              .sort_values("rank")
              .head(20)
              .copy())
        
        if not sub.empty:
            with col_info:
                st.metric(
                    "Toplam Öneri",
                    len(sub),
                    help=f"Kullanıcı {chosen} için önerilen film sayısı"
                )
            
            # Geliştirilmiş tablo gösterimi
            display_cols = ["rank", "title", "genres", "predicted_rating", "avg_rating", "rating_count"]
            sub_display = sub[display_cols].copy()
            sub_display.columns = ["Sıra", "Film Adı", "Türler", "Tahmin Puanı", "Ortalama Puan", "Toplam Rating"]
            
            st.dataframe(
                sub_display,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Tahmin Puanı": st.column_config.NumberColumn(format="%.2f ⭐"),
                    "Ortalama Puan": st.column_config.NumberColumn(format="%.2f"),
                    "Toplam Rating": st.column_config.NumberColumn(format="%,d"),
                }
            )
        else:
            st.warning(f"Kullanıcı {chosen} için öneri bulunamadı.")


# TAB 3: MLflow Model Performansı
with tab_mlflow:
    st.markdown("### ALS Model Performansı")
    
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = mlflow.tracking.MlflowClient()
        exp = client.get_experiment_by_name(
            os.environ.get("MLFLOW_EXPERIMENT", "movielens-als")
        )
        
        if exp is None:
            st.info("💡 Henüz model eğitimi gerçekleştirilmemiş. `train_als.py` komutunu çalıştırın.")
        else:
            runs = client.search_runs(
                exp.experiment_id,
                order_by=["attributes.start_time DESC"],
                max_results=20,
            )
            
            if not runs:
                st.warning("Henüz tamamlanan eğitim çalıştırması yok.")
            else:
                rows = []
                for r in runs:
                    rows.append({
                        "Run ID": r.info.run_id[:8],
                        "RMSE": r.data.metrics.get("rmse"),
                        "MAE": r.data.metrics.get("mae"),
                        "Rank": r.data.params.get("rank"),
                        "Reg Param": r.data.params.get("regParam"),
                        "Max Iterasyon": r.data.params.get("maxIter"),
                        "Rating Sayısı": r.data.params.get("n_ratings"),
                        "Eğitim Süresi (s)": r.data.metrics.get("train_seconds"),
                    })
                
                df = pd.DataFrame(rows)
                
                # Performans metriklerini göster
                col1, col2, col3 = st.columns(3)
                
                if df["RMSE"].notna().any():
                    best_rmse = df["RMSE"].min()
                    with col1:
                        st.metric("🎯 En İyi RMSE", f"{best_rmse:.4f}")
                
                if df["MAE"].notna().any():
                    best_mae = df["MAE"].min()
                    with col2:
                        st.metric("📊 En İyi MAE", f"{best_mae:.4f}")
                
                with col3:
                    st.metric("🏃 Toplam Çalıştırmalar", len(df))
                
                # Model performansı tablosu
                st.markdown("#### Son 20 Eğitim Çalıştırması")
                st.dataframe(
                    df.sort_values("Run ID", ascending=False),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "RMSE": st.column_config.NumberColumn(format="%.4f"),
                        "MAE": st.column_config.NumberColumn(format="%.4f"),
                        "Rank": st.column_config.NumberColumn(),
                        "Reg Param": st.column_config.NumberColumn(format="%.2e"),
                        "Rating Sayısı": st.column_config.NumberColumn(format="%,d"),
                        "Eğitim Süresi (s)": st.column_config.NumberColumn(format="%.1f"),
                    }
                )
                
                # RMSE vs Rank scatter plot
                if not df.empty and df["RMSE"].notna().any():
                    st.markdown("#### Model Parametreleri ve Performans")
                    
                    fig = px.scatter(
                        df.dropna(subset=["RMSE"]),
                        x="Rank",
                        y="RMSE",
                        size="Rating Sayısı",
                        color="Reg Param",
                        hover_data=["Run ID", "MAE", "Eğitim Süresi (s)"],
                        color_continuous_scale="Viridis",
                        labels={
                            "Rank": "Rank (Latent Faktörler)",
                            "RMSE": "RMSE (Ortalama Hata)",
                            "Rating Sayısı": "Rating Sayısı"
                        }
                    )
                    fig.update_layout(height=450)
                    st.plotly_chart(fig, use_container_width=True)
                    
    except Exception as exc:
        st.error(f"❌ MLflow erişilemiyor: {str(exc)[:200]}")

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #888;">
    <small>
    🔗 MovieLens 25M Streaming Veri Mühendisliği Projesi<br>
    Kocaeli Üniversitesi • Spark 3.5 • Delta 3.2 • MLflow 2.16 • Streamlit 1.39
    </small>
</div>
""", unsafe_allow_html=True)

