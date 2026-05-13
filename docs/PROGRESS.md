# PROGRESS — Adım adım yapılanlar

Bu dosya sırayla hangi dosyada ne yapıldığını tutuyor; sunum ve takım arkadaşlarına anlatım için.

## 1. Repo iskeleti
- `.gitignore` — data, delta-store, mlruns, checkpoints, __pycache__, log dosyaları ignore.
- `.env.example` — `KAFKA_TOPIC_CRIMES`, `MLFLOW_EXPERIMENT`, `CRIMES_DIR`, producer modları örnekleri.
- `data/`, `notebooks/`, `docs/` klasörleri.

## 2. Docker Compose mimarisi
- `docker-compose.yml` — 7 servis: `kafka` (KRaft, no zookeeper), `spark-master`, `spark-worker` (8 core/6GB), `mlflow`, `producer`, `pipeline`, `dashboard`.
- Volume bind mount'ları: `./spark`, named volume `delta-data`/`checkpoints-data`/`mlruns-data`/`ivy-cache`, dataset `${CRIMES_DIR:-../crimes}:/opt/data/crimes:ro`.
- YAML anchor: `x-spark-volumes` ile DRY.

## 3. MLflow tracking server
- `mlflow/Dockerfile` — python:3.11-slim + mlflow 2.16.2 + boto3 + psycopg2.
- Backend: SQLite, artifact root: `/opt/mlruns`.

## 4. Kafka producer (Adım 2)
- `producer/crime_producer.py`:
  - `Crimes.csv`'i satır satır okur, JSON olarak Kafka'ya basar.
  - `Date` alanını epoch ms'ye parse, boolean/int/float dönüşümleri.
  - `id` veya `Primary Type` null ise satır atlanır.
  - 3 mod: `fixed`/`speedup`/`burst`. SIGINT/SIGTERM temiz kapanış.

## 5. Spark image
- `spark/Dockerfile` — apache/spark:3.5.8-python3 + delta-spark 3.2 + spark-sql-kafka 3.5.1 + mlflow-spark 2.16.
- `spark/run.sh` — spark-submit wrapper; env'den `SPARK_DRIVER_MEMORY`/`SPARK_EXECUTOR_MEMORY` override.

## 6. Bronze layer (Adım 3)
- `spark/jobs/_session.py` — ortak SparkSession factory + path helper.
- `spark/jobs/bronze_ingest.py`:
  - Kafka `crimes` topic'ten structured streaming read.
  - JSON parse `CRIME_SCHEMA` (17 alan).
  - Delta'ya append, `event_date` partition.
  - `maxOffsetsPerTrigger=200000` (kritik fix: sınırsız batch'te commit takılıyor).

## 7. Silver layer (Adım 3)
- `spark/jobs/silver_clean.py`:
  - Bronze'dan streaming/batch read (`SILVER_BATCH_ONCE`).
  - Null filtre + `dropDuplicates(["id"])`.
  - Türetilmiş: `hour_of_day`, `day_of_week`, `month`, `event_year`.
  - `partitionBy(event_year)` (28 dizin — `event_date` fan-out yapardı).
  - Streaming sinkinde `foreachBatch` + Delta MERGE upsert.

## 8. Optimize silver
- `spark/jobs/optimize_silver.py`:
  - `DeltaTable.optimize().executeCompaction()`.
  - ML read fazını hızlandırır (binlerce küçük parquet → ~1GB hedef boyut).

## 9. Gold layer (Adım 3)
- `spark/jobs/gold_features.py`:
  - `gold/type_stats`: primary_type başına count, arrest_rate, domestic_rate, frequency_bucket.
  - `gold/district_stats`: district başına count, en sık primary_type (window function), avg lat/lon, size_bucket.
  - `gold/hourly_stats`: hour × primary_type heatmap için.

## 10. EDA (Adım 4)
- `spark/notebooks/01_eda.py`:
  - Temel istatistikler, eksik değer analizi, top-15 dağılım.
  - Yıllık/saatlik/haftalık trendler.
  - Sayısal `describe`.
  - `gold/eda_overview` Delta + JSON özet (chart üretimi için).

## 11. Feature Engineering (Adım 5)
- `spark/notebooks/02_feature_engineering.py`:
  - 13 özellik (PDF kuralı en az 5):
    - Zaman: `hour_of_day`, `day_of_week`, `month`, `event_year`
    - Konum: `district`, `ward`, `community_area`, `beat`
    - Koordinat: `latitude`, `longitude`
    - Bağlam: `arrest_int`, `domestic_int`
    - Türetilmiş bool: `is_weekend`, `is_night`
  - `gold/feature_view` Delta — ML-hazır snapshot.

## 12. ML Training — 5 model (Adım 6)
- `spark/ml/train_models.py`:
  - Silver'dan örnek (`SAMPLE_FRACTION`).
  - Top-N `primary_type` + OTHER (sınıf dengesi).
  - StringIndexer → VectorAssembler → her klasifier ayrı pipeline.
  - **5 model sırayla:**
    1. `LogisticRegression(multinomial, maxIter=20)`
    2. `DecisionTreeClassifier(maxDepth=10, maxBins=64)`
    3. `RandomForestClassifier(numTrees=50, maxDepth=10)`
    4. `OneVsRest(GBTClassifier(maxIter=20, maxDepth=5))`
    5. `NaiveBayes(multinomial)`
  - **Her run için MLflow logla:**
    - Params + Metrics (accuracy, weighted_f1/p/r, auc_ovr_macro, train_seconds).
    - **Feature Importance** (RF/DT: `featureImportances`, LR: `|coef|.mean()`, OvR: alt-modeller ortalaması) — CSV artifact + per-feature MLflow metric.
    - **Confusion Matrix** CSV artifact.
    - **Per-class precision/recall/F1** CSV artifact.
    - Model + registry (`chicago-crime-<name>`).
  - Tek modelin çökmesi diğerlerini durdurmaz.

## 13. Inference
- `spark/ml/inference.py`:
  - MLflow'da en yüksek `accuracy`'ye sahip modeli bul.
  - `runs:/{run_id}/{model_type}_model` URI ile yükle.
  - Silver'dan örnek üzerinde tahmin, `gold/predictions` Delta'ya yaz.

## 14. Streamlit dashboard (Adım 7)
- `dashboard/app.py`:
  - 4 metrik kartı + en iyi model banner'ı.
  - **5 sekme:** Genel, İlçe (Mapbox harita), Saat (heatmap), Tahminler (confusion matrix), MLflow (5 model karşılaştırma + Pareto).

## 15. Charts (Adım 7 zorunlu görseller)
- `scripts/make_charts.py`:
  - MLflow runs + EDA summary JSON'ı oku.
  - 9 PNG: 5 model grouped bar, feature importance hbar, confusion matrix heatmap, ROC curve proxy, yıllık/saatlik line, top-15 histogram, pie chart, district bar.

## 16. Yardımcılar
- `Makefile` — `make build/up/bronze/silver/gold/eda/train/inference/charts/dashboard/clean`.
- `scripts/run_all.sh` — 11 aşamalı tek-komut.
- `scripts/verify.sh` — compose config + py_compile + bash -n + dockerfile + dataset + requirements.

## 17. Branch geçmişi
- `main` → `feat/pipeline-bootstrap` → `feat/hyperparam-sweep` (MovieLens son hali) → `feat/chicago-crimes` (mevcut, tam rewrite).
- Veri seti pivot (MovieLens → Chicago Crimes): proje sonunda form bilgilerine uyum sağlamak için tam yeniden yazım.

## 18. Test edilen
- `make verify` ✅
- Kafka healthy, Spark master+worker registered (8 core), MLflow up
- Producer 2M kayıt → Kafka (~15K/s burst mode)
- (Pipeline çalışırken bu bölüm sonuçlarla güncellenir)
