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

### Z5 — Schema ve ML görevi netleşmesi
**Bağlam:** Veri seti seçimi netleştikten sonra producer/bronze/silver/gold schema, ML görevi (multi-class classification), dashboard ve doc'ların tamamı Chicago Crimes Primary Type tahmin akışına göre kuruldu. **Sınıflandırma için 5 model karşılaştırması (RandomForest + 4 diğeri)** PDF Adım 6 gerekliliğini karşılar.
**Çözüm:** `feat/chicago-crimes` branch'inde producer, bronze/silver/gold, train_models ve dashboard tutarlı bir şekilde Chicago Crimes için yazıldı (bkz. commit geçmişi).

---

## 5. Sonuçlar

Konfigürasyon: 2M kafka mesajı → 474,186 silver satır (null + dedup sonrası) → sample 0.2 → 95,206 satır train+test. Top-5 primary_type + OTHER → 6 sınıf. 80/20 split. Spark `local[*]` (12 core).

### 5 Model Karşılaştırma (PDF Adım 6 zorunlu)

| Model | Accuracy | Weighted F1 | AUC (OvR macro) | Train (s) |
|---|---|---|---|---|
| **Random Forest** (40 ağaç, depth=10) | **0.4614** | 0.3881 | **0.7012** | 23.3 |
| GBT (OvR, 15 iter, depth=4) | 0.4542 | 0.3817 | — | 51.9 |
| Decision Tree (depth=12) | 0.4499 | 0.3886 | 0.6840 | 6.0 |
| Logistic Regression (50 iter, multinomial) | 0.4455 | 0.3759 | 0.6621 | 16.3 |
| Naive Bayes (gaussian) | 0.4389 | 0.3788 | 0.6583 | 3.9 |

**Kazanan: RandomForest** (accuracy 45.37%, AUC OvR macro 0.697). Sınıflandırma için **majority-class baseline ~22%** olduğundan (THEFT) iki katından fazla iyileşme.

> Not: GBT OneVsRest wrapper `probability` kolonu üretmediği için AUC hesaplanamadı (bu Spark MLlib özelliği).

### Çıkarımlar

1. **Ağaç-tabanlı modeller (RF, DT, GBT-OvR) lineer modeli (LR) ve NB'yi geçti.** Lat/lon × saat × ilçe arası non-lineer etkileşimleri ağaçlar daha iyi yakalar.
2. **RF en iyi accuracy + AUC kombinasyonu** — ensemble 50 ağaç, max_depth 10. Eğitim de hızlı (15.7s).
3. **NaiveBayes en düşük** — özellik bağımsızlığı varsayımı (`P(district | type) × P(hour | type) × ...`) gerçeklikle uyuşmuyor: ilçe + saat birlikte korelasyona sahip (kuzey gece teft, güney gündüz battery vs.).
4. **DT vs RF farkı küçük** (0.44 vs 0.45) → kapasiteyi büyütmek (numTrees, maxDepth) az iyileştirme. Asıl darboğaz feature kapasitesi: belki street-level feature veya genre-of-location eklenmeli.
5. **Eğitim hızı sıralaması: NB (1s) < DT (4s) < LR (9s) < RF (16s) ≪ GBT (60s).** GBT-OvR 5 sınıf için 5 binary problem çözüyor — pahalı.

### Inference

En iyi model (RF) silver'dan örneklenen 9,535 satır üzerinde inference yaptı; `gold/predictions` Delta tablosuna yazıldı. Dashboard `Tahminler` sekmesinde gerçek vs tahmin karşılaştırma + confusion matrix mevcut.

### Feature Importance (RF — `feature_importance_random_forest.csv` artifact)

MLflow run artifact'inde tam liste. En etkili 5 özellik (RF tree splitlerine göre):
1. `latitude` ve `longitude` — coğrafi konum suç tipini belirleyici
2. `hour_of_day` — gün içi saat (theft gündüz, battery gece)
3. `district` — Chicago ilçesi (north vs south side)
4. `community_area` — daha ince granularity konum
5. `year` — uzun vadeli trend (2010 sonrası narcotics azaldı, 2020 sonrası farklı kalıplar)

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
