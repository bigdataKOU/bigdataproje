# Mimari Detayları

## Veri akışı

1. **Üretici (`producer/ratings_producer.py`)**
   - `ml-25m/ratings.csv` dosyasını satır satır okur.
   - Her rating'i JSON olarak `ratings` Kafka topic'ine basar.
   - `timestamp` farklarını `PRODUCER_SPEEDUP` faktörüyle hızlandırır (gerçek zaman simülasyonu).

2. **Bronze (`spark/jobs/bronze_ingest.py`)**
   - Spark Structured Streaming ile Kafka'dan `subscribe="ratings"`.
   - JSON'u `RATING_SCHEMA` ile parse eder, Kafka metadata'sını da saklar.
   - Delta'ya `event_date` partitioning ile append.
   - Checkpoint: `/opt/checkpoints/bronze`.

3. **Silver (`spark/jobs/silver_clean.py`)**
   - Bronze Delta tablosundan **streaming read**.
   - `foreachBatch` ile her micro-batch için:
     - `rating ∈ [0.5, 5.0]` filtresi, null kontrolleri.
     - `(userId, movieId, timestamp)` key'iyle deduplication.
     - `movies.csv` ile broadcast join (title + genres).
     - Delta `MERGE` ile upsert (idempotent).

4. **Gold (`spark/jobs/gold_features.py`)**
   - Silver'dan batch read.
   - İki agregat tablo:
     - `gold/movie_stats`: count, avg, std, popularity_bucket.
     - `gold/user_stats`: count, active_days, activity_bucket.
   - Overwrite mode (idempotent yeniden çalıştırılabilir).

5. **Eğitim (`spark/ml/train_als.py`)**
   - Silver tablosundan `(userId, movieId, rating)` çek.
   - 80/20 train-test split.
   - ALS hiperparametreleri env'den (`ALS_RANK`, `ALS_REG`, `ALS_ITER`).
   - MLflow:
     - Params loglanır.
     - RMSE + MAE metrikleri.
     - `mlflow.spark.log_model` ile model artifact + registry'ye versiyon kaydı.

6. **Inference (`spark/ml/inference.py`)**
   - MLflow registry'den en son model versiyonunu yükler.
   - `recommendForAllUsers(TOP_K)` → kullanıcı başına top-N film.
   - `gold/movie_stats` ile join'leyip Delta'ya yazar.

7. **Dashboard (`dashboard/app.py`)**
   - `deltalake` Python client ile Delta tablolarını okur (Spark gerekmez).
   - Üç tab: pipeline metrikleri, user-level öneri demosu, MLflow run karşılaştırması.

## Tasarım gerekçeleri

- **KRaft Kafka**: Zookeeper deprecated, tek-node test için ideal.
- **`foreachBatch` MERGE**: Streaming sink'leri append-only; gerçek upsert için bu deyim gerekli.
- **Broadcast join (movies)**: ~62K satır, küçük; her worker'a kopyalamak shuffle'dan ucuz.
- **`coldStartStrategy="drop"`**: Test kümesinde train'de görmediğimiz user/movie varsa NaN üretmek yerine satırı atar — RMSE temiz çıkar.
- **`event_date` partitioning**: Bronze'da gün bazlı sorgu/silme verimli.
- **MLflow SQLite**: Lokal demo için yeterli, production'da Postgres'e taşınır.
