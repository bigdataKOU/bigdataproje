# Teknik Rapor — Chicago Crimes ML Pipeline

**BLM442 Büyük Veri Dönem Projesi · 13 Mayıs 2026**

| Üye | Numara |
|---|---|
| Emre Aytaş | 220202098 |
| Hatice Kübra Kılıçaslan | 220202077 |
| Berker Yiğit | 220202046 |
| Mertcan Kuzey | 240202009 |

---

## 1. Problem ve Veri Seti

**Veri seti:** [Chicago Crimes 2001 – Present](https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2), Chicago Police Department CLEAR sisteminden. Yaklaşık **7.9 milyon olay**, 22 kolon (ID, Date, Block, IUCR, Primary Type, Description, Location Description, Arrest, Domestic, Beat, District, Ward, Community Area, FBI Code, X/Y Coordinate, Year, Updated On, Latitude, Longitude, Location).

**Görev:** Bir olay için konum + zaman + bağlam özelliklerinden **`Primary Type` (suç tipi) sınıflandırması**. Bu çoklu sınıf bir sınıflandırma problemidir; PDF metnindeki "suç tipi ve bölge tahmini" tanımına uygundur.

**Tasarım kararı:** Chicago verisetinde 35+ farklı `Primary Type` var ve dağılım çok dengesiz (THEFT ~%17, ARSON < %0.1). Sınıflandırıcının anlamlı çalışabilmesi için **Top-5 sınıfı tut + kalanını OTHER'a topla** → 6 sınıflı dengeli problem.

---

## 2. Mimari ve Teknoloji Yığını

```
Crimes.csv ──▶ crime_producer ──Kafka──▶ Spark Structured Streaming
                                              │
                                              ▼
                       Delta Lake: Bronze (raw event store)
                                              │
                                              ▼
                           Silver (dedupe + null clean + türetilmiş ozellikler)
                                              │
                       ┌──────────────────────┼──────────────────────┐
                       ▼                      ▼                      ▼
                Gold tablolari         EDA notebook          5 ML modeli
                (type/district/         (Adim 4)            (LogReg/DT/RF/
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
```

**Servisler (`docker-compose.yml`):** `kafka` (KRaft 3.9), `spark-master` + `spark-worker` (Apache Spark 3.5.8, 8 core / 6GB), `mlflow` (tracking + registry, SQLite backend), `producer` (custom Python), `pipeline` (PySpark + Delta + MLflow), `dashboard` (Streamlit). Tümü Docker named-volume kullanır (host `make clean` sorunsuz).

---

## 3. Pipeline Adımları (PDF metniyle birebir eşleştirme)

| PDF Adım | Bu projede karşılığı |
|---|---|
| **Adım 1 — Docker ortamı** | `docker-compose.yml` + 4 custom Dockerfile (producer/spark/mlflow/dashboard) |
| **Adım 2 — Kafka producer** | `producer/crime_producer.py`: CSV → JSON → topic, 3 hız modu (fixed/speedup/burst), log her 5K mesaj |
| **Adım 3 — Spark Structured Streaming** | `bronze_ingest.py` (Kafka→Delta append, partitionBy event_date, maxOffsetsPerTrigger=200K) + `silver_clean.py` (dedupe + null filtre + 4 türetilmiş özellik) |
| **Adım 4 — EDA** | `spark/notebooks/01_eda.py`: total/unique sayımları, eksik değer analizi, yıllık/saatlik/haftalık trend, sayısal `describe`, `gold/eda_overview` Delta, `eda_summary.json` |
| **Adım 5 — Feature Engineering** | `spark/notebooks/02_feature_engineering.py`: 13 özellik (4 zaman, 4 konum, 2 koordinat, 2 bağlam, 2 türetilmiş bool `is_weekend`/`is_night`). `gold/feature_view` Delta. Her özelliğin iş mantığı docstring'de |
| **Adım 6 — 5 ML modeli + MLflow** | `spark/ml/train_models.py`: LogReg + DecisionTree + RandomForest + GBT(OneVsRest) + NaiveBayes. Her run için MLflow params/metrics/model + **Feature Importance** + **Confusion Matrix** + per-class CSV |
| **Adım 7 — Dashboard + görseller** | `dashboard/app.py` 5 sekme + `scripts/make_charts.py` 9 PNG (5-model bar, FI hbar, CM heatmap, ROC sketch, yıllık/saatlik line, top-15 hist, pie, district bar) |

---

## 4. Karşılaşılan Zorluklar ve Çözümleri

### Z1 — Bronze tek dev micro-batch'te commit etmiyordu
**Sebep:** `startingOffsets=earliest` + sınırsız batch + 2 core worker → 5M kayıt tek batch'te 10+ dk Delta log'a yazılamadı.
**Çözüm:** `maxOffsetsPerTrigger=200000` (her ~5s commit) + worker upgrade 2→8 core.

### Z2 — Silver shuffle + partition fan-out 67K küçük parquet üretti
**Sebep:** `dropDuplicates` shuffle + `partitionBy(event_date)` 10K event_date × 200 shuffle partition.
**Çözüm:**
- `partitionBy(event_year)` ile sadece 28 dizin (event_date yerine).
- `optimize_silver.py` ile Delta `OPTIMIZE` compaction (67K → 8.5K dosya). ALS/RF read fazı bu sayede dakikalar yerine saniyeler sürüyor.

### Z3 — Sweep script'i tek run sonra exit ediyordu
**Sebep:** `docker compose exec -T` stdin'i tüketti, while-read loop kalan satırı alamadan EOF.
**Çözüm:** Exec çağrısına `< /dev/null` eklenerek stdin yönlendirildi. (Not: sweep konsepti sonradan kaldırıldı — PDF tek model değil **5 model** istiyor; bu fix kalan kodu temizledikten sonra geçmişte hala önemli bir öğrenim.)

### Z4 — GBT multi-class değil
**Sebep:** Spark MLlib `GBTClassifier` sadece binary.
**Çözüm:** `OneVsRest(GBTClassifier)` wrapper — N sınıf için N binary problem. Feature importance toplama `models` attribute'ı üzerinden alt modellerin ortalaması.

### Z5 — Veri seti pivot (MovieLens → Chicago Crimes)
**Bağlam:** Proje aşaması sonunda hocadan/grup içinden geri bildirim: MovieLens yerine Chicago Crimes seçildi (form ile kayıtlı). Tüm kodun (`producer`, bronze/silver/gold schema, ML görevi, dashboard, docs) MovieLens izlerinden tamamen ayıklanması gerekti. **ALS (collaborative filtering) → Random Forest + 4 diğer classifier (multi-class)** değişimi en büyük yapısal farktı.
**Çözüm:** `feat/chicago-crimes` branch'inde sıfırdan rewrite (bkz. commit geçmişi).

---

## 5. Sonuçlar

Aşağıdaki rakamlar `run_all.sh` ile 2M kayıt ingest (sample 0.2) konfigurasyonunda alınmıştır.

> Bu bölüm, sweep tamamlanıp `make_charts.py` çıktıları üretildikten sonra dolacaktır.
> Otomatik karşılaştırma: `docs/figures/metrics_table.csv` ve `docs/figures/models_comparison.png`.

**Beklenen sıralama (literatürden):**

1. RandomForest / GBT — ağaç ensemble'ları kategorik+numerik karışık özellikleri en iyi öğrenir.
2. DecisionTree — tek ağaç, overfit riski var ama RF'in altyapısını gösterir.
3. LogisticRegression — lineer baseline. Konum + saat lineer kararlarla orta seviye.
4. NaiveBayes — özellik bağımsızlığı varsayımı (çok güçlü); konum/zaman korelasyonu varken doğruluk düşer.

**Feature Importance hipotezi** (RF'den çıkacak): `hour_of_day`, `latitude`, `longitude`, `district`, `community_area` ilk 5'te bekleniyor. `year` orta, `arrest_int` ek bağlam (bazı suç tiplerinde tutuklama daha olası).

---

## 6. Sınırlamalar ve Gelecek İş

- **Class imbalance:** OTHER bucket (5+ sınıfın birleşimi) hala büyük. SMOTE veya class weight kullanılabilir.
- **Zaman-bağımlı split:** Random split 80/20 yerine yıl-bazlı split (2001-2020 train, 2021+ test) modelin gerçek zaman serisi performansını ölçer.
- **Geo-features:** Lat/lon raw yerine cluster (KMeans) ya da H3 hex bucket olarak verilebilir.
- **GBT-OvR çok yavaş:** 5+ sınıfta her sınıf için ayrı GBT eğitiliyor. Alternatif: XGBoost (Spark-Rapids veya Hadoop dışı).

---

## 7. Tekrarlanabilirlik

```bash
git clone https://github.com/bigdataKOU/bigdataproje.git
cd bigdataproje
git switch feat/chicago-crimes
cp .env.example .env

# Crimes.csv'i ../crimes/Crimes.csv olarak yerleştir (Chicago Open Data linkinden)
mkdir -p ../crimes
# (CSV'i taşı)

make verify        # statik
make run-all       # uçtan uca (~30-50 dk)

# Eriş
# Dashboard: http://localhost:8501
# MLflow:    http://localhost:5000
# Spark UI:  http://localhost:8080
```

Tüm commit geçmişi GitHub'da: `feat/chicago-crimes` branch.
