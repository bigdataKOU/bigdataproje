# PROGRESS — Adım adım yaptıklarımız

Bu dosya sırasıyla hangi dosyada ne yaptığımızı tutuyor; sunum ve takım arkadaşlarına anlatım için.

## 1. Repo iskeleti
- `.gitignore` — data, delta-store, mlruns, checkpoints, __pycache__ ignore.
- `.env.example` — KAFKA_BROKER, MLFLOW_TRACKING_URI, ALS hyperparam vs. örnekleri.
- `data/`, `delta-store/`, `mlruns/`, `notebooks/`, `docs/` boş klasörleri.
- Eski `readme.md` silindi (yerine kapsamlı `README.md`).

## 2. Docker Compose mimarisi
- `docker-compose.yml` — 7 servis: kafka (KRaft, no zookeeper), spark-master, spark-worker, mlflow, producer, pipeline, dashboard.
- Volume bind mount'ları: `./spark`, `./delta-store`, `./checkpoints`, `./mlruns` ve dataset (`../ml-25m`).
- YAML anchor: `x-spark-env`, `x-spark-volumes` ile DRY.

## 3. MLflow tracking server
- `mlflow/Dockerfile` — python:3.11-slim + mlflow 2.16.2 + boto3 + psycopg2.
- Backend: SQLite (`/opt/mlflow-store/mlflow.db`), artifact root: `/opt/mlruns`.
- Port 5000.

## 4. Kafka producer
- `producer/Dockerfile` — confluent-kafka image.
- `producer/ratings_producer.py`:
  - ratings.csv'i satır satır okur, JSON olarak Kafka'ya basar.
  - `PRODUCER_SPEEDUP` faktörüyle timestamp aralarını sıkıştırır (gerçek zamanlı simülasyon).
  - `PRODUCER_MAX_RECORDS` ile durdurma sınırı.
  - SIGINT/SIGTERM ile clean shutdown.

## 5. Spark image
- `spark/Dockerfile` — apache/spark:3.5.8-python3 + delta-spark 3.2 + spark-sql-kafka 3.5.1 + mlflow-spark 2.16.
- JAR'lar build sırasında ivy ile pre-resolve ediliyor → spark-submit hızlı.
- `spark/run.sh` — spark-submit wrapper, packages + delta config'leri.

## 6. Bronze layer
- `spark/jobs/_session.py` — ortak SparkSession factory + path sabitleri.
- `spark/jobs/bronze_ingest.py`:
  - Kafka 'ratings' topic'ten structured streaming read.
  - JSON parse (RATING_SCHEMA: userId, movieId, rating, timestamp, ingestedAt).
  - Kafka metadata (topic, partition, offset, timestamp) korunur.
  - Delta'ya append, `event_date` partition.
  - Checkpoint: `/opt/checkpoints/bronze`.

## 7. Silver layer
- `spark/jobs/silver_clean.py`:
  - Bronze'dan streaming read.
  - `foreachBatch` içinde:
    - rating ∈ [0.5, 5.0] filter, null kontrol.
    - `(userId, movieId, timestamp)` ile dedup.
    - movies.csv broadcast join (title, genres, genre_list).
    - Delta MERGE upsert (idempotent).
  - Checkpoint: `/opt/checkpoints/silver`.

## 8. Gold layer
- `spark/jobs/gold_features.py`:
  - Silver'dan batch read.
  - `gold/movie_stats`: count, avg, std, popularity_bucket (high/medium/low).
  - `gold/user_stats`: count, active_days, unique_movies, activity_bucket (power/active/casual).

## 9. ML training
- `spark/ml/train_als.py`:
  - Silver'dan (userId, movieId, rating) çek, opsiyonel sample.
  - 80/20 train/test split.
  - ALS(rank, regParam, maxIter) — coldStart="drop", nonnegative=True.
  - MLflow:
    - Params: rank, regParam, maxIter, train_ratio, sample_fraction, n_ratings.
    - Metrics: rmse, mae, train_seconds.
    - Model: `mlflow.spark.log_model` + registry'ye `movielens-als-recommender` adıyla.
    - Artifact: 20 örnek user için top-K öneri CSV'si.

## 10. Inference
- `spark/ml/inference.py`:
  - Registry'den en son model versiyonunu yükle (stage opsiyonel).
  - `recommendForAllUsers(TOP_K)` → flatten + rank.
  - `gold/movie_stats` ile join (title, genres, popüllüğü).
  - `gold/user_recommendations` Delta tablosuna yaz.

## 11. Streamlit dashboard
- `dashboard/Dockerfile` — streamlit 1.39 + deltalake (Spark gerektirmez!) + plotly.
- `dashboard/app.py`:
  - 4 metrik kartı: bronze/silver/movie/user satır sayıları.
  - Tab "Genel": top 20 film bar chart (rating count'a göre), aktivite kovaları pie.
  - Tab "Öneriler": userId selectbox → top-20 film tablosu.
  - Tab "MLflow": run karşılaştırma tablosu + RMSE vs rank scatter.

## 12. Yardımcılar
- `Makefile` — `make build/up/bronze/silver/gold/train/inference/dashboard/clean`.
- `notebooks/01_eda.ipynb` — keşifsel veri analizi (rating dist, yıl trend, top filmler).
- `docs/architecture.md` — teknik tasarım gerekçeleri.
- `README.md` — mimari ASCII, takım, kurulum, çalıştırma adımları.

## 13. Test edilen
- `docker compose config --quiet` ✅ (compose syntax)
- Tüm Python dosyaları `py_compile` ✅
- Kafka KRaft ayağa kalktı, healthy ✅
- Spark master + worker registered (2 core, 4GB) ✅
- MLflow tracking server gunicorn dinliyor (5000) ✅
- Producer Kafka'ya 1730 mesaj bastı, topic offset doğrulandı ✅

## 14. Yapılmadı (disk dolduğu için durdurduk)
- Pipeline image build (~4 GB Maven cache); Dockerhub'a push düşünülebilir.
- Bronze ingest job canlı çalıştırma.
- Silver/Gold job'larının end-to-end testi.
- ALS train + MLflow run kaydı.
- Inference + dashboard demo.
- Bunların hepsi yazılmış ve çalışmaya hazır kod.

## Sıradaki adımlar (disk açıldıktan sonra)
1. `make verify` (sentaks kontrolü, disk yormaz).
2. `make run-all` — tek komut uçtan-uca pipeline.
3. Sonuçlar: dashboard:8501, mlflow:5000, sparkUI:8080.

## 15. İkinci tur iyileştirmeleri (commit 2)
- `spark/Dockerfile` — Maven JAR pre-resolve adımı kaldırıldı (4 GB → 1.2 GB).
- `docker-compose.yml` — `ivy-cache` named volume eklendi; JAR'lar runtime'da cache'leniyor.
- `producer/ratings_producer.py` — `PRODUCER_MODE`: `fixed` (varsayılan, 1000 msg/s sabit) / `speedup` (timestamp tabanlı) / `burst` (sleep yok). Eski speedup-only modu çok yavaştı.
- `spark/jobs/silver_clean.py` — `cleaned.persist()` + `try/finally unpersist`; merge ile count aynı plan üzerinden çalışsın.
- `spark/ml/inference.py` — eksik `import mlflow.spark` eklendi.
- `dashboard/app.py` — gereksiz `TableNotFoundError` import temizliği, `pd.DataFrame | None` type-hint kaldırıldı (3.10 öncesi uyumluluk).
- `scripts/run_all.sh` — bronze/silver arka planda, producer foreground, gold→train→inference sırasıyla.
- `scripts/verify.sh` — compose config + py_compile + bash -n + dockerfile parse + dataset kontrolü + requirements pin kontrolü.
- `Makefile` — `make verify`, `make run-all` hedefleri.
