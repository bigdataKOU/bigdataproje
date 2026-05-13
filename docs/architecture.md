# Mimari Detayları

## Veri akışı

1. **Üretici (`producer/crime_producer.py`)**
   - `crimes/Crimes.csv` dosyasını satır satır okur.
   - Her olayı JSON olarak `crimes` Kafka topic'ine basar.
   - `Date` alanını epoch ms'ye parse eder, `Arrest`/`Domestic` boolean dönüşümü.
   - `id` veya `Primary Type` eksikse satırı atlar.
   - 3 mod: `fixed` (sabit hız), `speedup` (timestamp tabanlı), `burst` (max hız).

2. **Bronze (`spark/jobs/bronze_ingest.py`)**
   - Spark Structured Streaming ile Kafka'dan `subscribe="crimes"`.
   - JSON'u `CRIME_SCHEMA` (17 alan) ile parse eder, Kafka metadata'sını da saklar.
   - `event_time` = `event_time_ms/1000`, `event_date` = date.
   - Delta'ya `event_date` partitioning ile append.
   - `maxOffsetsPerTrigger=200000` (cluster mode'da büyük batch'lerin commit takılmasını engeller).

3. **Silver (`spark/jobs/silver_clean.py`)**
   - Bronze Delta tablosundan streaming veya tek-batch read (`SILVER_BATCH_ONCE`).
   - Temizlik:
     - `id`, `primary_type`, `district`, `latitude`, `longitude` null kontrolleri.
     - `id` üzerinden deduplication.
   - Türetilmiş özellikler: `hour_of_day`, `day_of_week`, `month`, `event_year`.
   - `partitionBy(event_year)` ile yaz — 28 yıllık veri için makul granularity, fan-out yok.

4. **Gold (`spark/jobs/gold_features.py`)**
   - Silver'dan batch read.
   - Üç agregat tablo:
     - `gold/type_stats`: primary_type başına count, arrest_rate, domestic_rate, frequency_bucket
     - `gold/district_stats`: district başına count, top primary_type, avg lat/lon (window function)
     - `gold/hourly_stats`: hour × primary_type heatmap için
   - Overwrite mode (idempotent yeniden çalıştırılabilir).

5. **EDA (`spark/notebooks/01_eda.py`) — PDF Adım 4**
   - Temel istatistikler (total/unique/distinct sayımları)
   - Eksik değer analizi (silver sonrası genelde 0)
   - Yıllık/saatlik/haftalık trendler
   - Sayısal değişken `describe`
   - `gold/eda_overview` Delta + JSON özet (chart üretimi için)

6. **Feature Engineering (`spark/notebooks/02_feature_engineering.py`) — PDF Adım 5**
   - 13 özellik (PDF kuralı en az 5):
     - 4 zaman: `hour_of_day`, `day_of_week`, `month`, `event_year`
     - 4 konum: `district`, `ward`, `community_area`, `beat`
     - 2 koordinat: `latitude`, `longitude`
     - 2 bağlam: `arrest_int`, `domestic_int`
     - 2 türetilmiş bool: `is_weekend`, `is_night`
   - `gold/feature_view` Delta — ML-hazır snapshot.

7. **5 Model Eğitimi (`spark/ml/train_models.py`) — PDF Adım 6**
   - Silver'dan örnek (`SAMPLE_FRACTION`).
   - Top-N primary_type + OTHER label (sınıf dengesi).
   - StringIndexer → VectorAssembler → her klasifier ayrı pipeline.
   - **5 model:**
     1. `LogisticRegression(multinomial)`
     2. `DecisionTreeClassifier`
     3. `RandomForestClassifier`
     4. `OneVsRest(GBTClassifier)` (GBT native multi-class değil)
     5. `NaiveBayes(multinomial)`
   - 80/20 train-test split.
   - **Her run için MLflow:**
     - Params loglanır.
     - Metrics: accuracy, weighted F1/precision/recall, AUC-OvR-macro, train_seconds.
     - **Feature Importance** (modelden çıkarılır; LR için |coefficient| ortalaması, RF/DT için `featureImportances`, OneVsRest için alt-modellerin ortalaması) — CSV artifact + per-feature MLflow metric.
     - **Confusion Matrix** CSV artifact.
     - **Per-class precision/recall** CSV artifact.
     - Model + registry kaydı (`mlflow.spark.log_model`).
   - Tek modelin çökmesi diğerlerini durdurmaz (`try/except` her model için).

8. **Inference (`spark/ml/inference.py`)**
   - MLflow'da en yüksek `accuracy`ye sahip modeli bul.
   - `runs:/{run_id}/{model_type}_model` URI ile yükle.
   - Silver'dan örnek satırlar üzerinde tahmin.
   - `predicted_label` (StringIndexer labels'tan geri çöz).
   - `gold/predictions` Delta tablosuna yaz.

9. **Dashboard (`dashboard/app.py`)**
   - `deltalake` Python client ile Delta tablolarını okur (Spark gerekmez).
   - 5 sekme: Genel (suç tipi), İlçe (mapbox), Saat (heatmap), Tahminler (CM görseli), MLflow (5 model karşılaştırma).
   - Üst banner'da en iyi modelin accuracy + F1 + parametreleri.

## Tasarım gerekçeleri

- **KRaft Kafka**: Zookeeper deprecated, tek-node deneme için ideal.
- **`maxOffsetsPerTrigger`**: Sınırsız batch, 2-8 core cluster'da büyük backlog'la commit etmiyor; bound batch her ~10s commit eder.
- **`partitionBy(event_year)` silver'da**: 28 yıllık veri × `partitionBy(event_date)` 10K+ dizine sebep olur (fan-out hell). `event_year` 28 dizin → makul.
- **Delta `OPTIMIZE`**: Silver yazılırken shuffle + partitioning binlerce küçük parquet üretir. ML read fazını hızlandırmak için compact et.
- **Top-N + OTHER label**: Chicago'da 30+ primary_type var; tail uzun. Sınıf dengesizliği eğitimi bozar — top-5 + OTHER ile 6 sınıf yeterince zengin + dengeli.
- **OneVsRest GBT için**: Spark MLlib GBTClassifier sadece binary. Multi-class için OneVsRest wrapper (N binary problem).
- **`local[*]` ML jobs için**: Cluster Spark'ın MLflow artifact yazımı DFS staging sorunları yaratabilir. ML jobs `local[*]` ile tek JVM'de tutarlı.
- **MLflow SQLite**: Lokal demo için yeterli, production'da Postgres'e taşınır.

## Dataset notu

Chicago Police Department CLEAR sistemi verisi:
- 2001'den günümüze 7.9M+ olay (son 7 gün veride yok)
- Block-level adres (gerçek konum maskelenmiş)
- Polis tarafından ön sınıflandırma — değişebilir (resmi disclaimer var)
