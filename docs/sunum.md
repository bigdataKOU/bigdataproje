# Sunum — Chicago Crimes Pipeline

**BLM442 Büyük Veri Analizine Giriş · Dönem Projesi · 2025-2026 Bahar**

Kocaeli Üniversitesi — Bilgisayar Mühendisliği — Dr. Ayşe Gül Eker

| Üye | Numara | Sorumluluk |
|---|---|---|
| Emre Aytaş | 220202098 | Producer + Kafka konfigürasyonu |
| Hatice Kübra Kılıçaslan | 220202077 | EDA + Feature Engineering |
| Berker Yiğit | 220202046 | Bronze/Silver/Gold + ML modelleri |
| Mertcan Kuzey | 240202009 | Dashboard + dokümantasyon |

> Not: Repo'da hepimizin commit'i var; sunumda bireysel sorulara hazırlık için bu paylaşım rehber.

---

## Slayt 1 — Kapak (1 dk)

- Konu: **Chicago Crimes 2001-Present** üzerinde uçtan-uca büyük veri pipeline'ı
- Hedef: PDF'in 7 adımının her birini somut bir bileşene oturtmak
- Süre: 10-15 dk + soru-cevap

---

## Slayt 2 — Problem ve veri seti (1 dk)

- Chicago Police CLEAR sistemi → 2001'den günümüze 7.9M olay
- 22 kolon: ID, Date, Block, Primary Type (35+ tip), District, Ward, Community Area, Lat/Lon, Arrest, Domestic, FBI Code...
- Görev: **Çoklu sınıf "Primary Type" sınıflandırma** (PDF: "suç tipi ve bölge tahmini")
- Sınıf dağılımı dengesizliği → Top-5 + OTHER ile 6 sınıf

---

## Slayt 3 — Mimari (2 dk)

```
Crimes.csv ─▶ Producer ─Kafka▶ Spark Streaming ─▶ Delta Bronze
                                                      │
                                            Silver (dedup + features)
                                                      │
                              ┌───────────────────────┼─────────────────┐
                              ▼                       ▼                 ▼
                       Gold tabloları            EDA notebook     5 ML modeli
                                                                       │
                                                                MLflow registry
                                                                       │
                                                              Inference + Dashboard
```

**Servisler (`docker-compose.yml`):** 7 konteyner — kafka (KRaft), spark-master, spark-worker (8 core/6GB), mlflow (SQLite), producer, pipeline, dashboard (Streamlit).

---

## Slayt 4 — Adım 1 & 2: Docker + Kafka Producer (1.5 dk)

- **Adım 1 (Docker):** `docker-compose.yml` + 4 Dockerfile, named volumes (delta/checkpoint/mlruns), tek komutta `docker compose up`.
- **Adım 2 (Producer):** `producer/crime_producer.py`
  - CSV → JSON → Kafka `crimes` topic
  - 3 hız modu: `fixed` (sn'de N), `speedup` (timestamp tabanlı), `burst` (max hız)
  - `Date` alanını epoch ms'ye parse, `Arrest`/`Domestic` bool dönüşümü

**Demo komut:** `make producer` veya `make run-all`.

---

## Slayt 5 — Adım 3: Spark Streaming + Delta Lake (2 dk)

- **Bronze (`bronze_ingest.py`):** Kafka → Delta `partitionBy(event_date)`. Kritik fix: `maxOffsetsPerTrigger=200000` (yoksa cluster dev batch'te commit etmiyor).
- **Silver (`silver_clean.py`):** dedup by `id`, null filtre, türetilmiş özellikler (`hour_of_day`, `day_of_week`, `month`, `event_year`). `partitionBy(event_year)` (28 dizin — `event_date` 10K+ fan-out yaratırdı).
- **Gold (`gold_features.py`):** 3 tablo — `type_stats` (arrest_rate, frequency_bucket), `district_stats` (window fn ile en sık tip + lat/lon merkezi), `hourly_stats` (saat × tip heatmap).
- **Bonus:** `optimize_silver.py` — Delta OPTIMIZE compaction (67K küçük parquet → ~8.5K).

---

## Slayt 6 — Adım 4: EDA (1 dk)

`spark/notebooks/01_eda.py` — temel istatistikler + zaman trendi + dağılım.

**Sayısal bulgular (silver=474,186 satır):**
- 34 benzersiz Primary Type
- 23 District × 79 Community Area
- Tutuklama oranı: ~21%
- Domestic oranı: ~16%
- Yıl aralığı: 2001-2023
- **En sık tipler:** THEFT, BATTERY, CRIMINAL DAMAGE, ASSAULT, OTHER OFFENSE

Çıktılar: `gold/eda_overview` Delta + `eda_summary.json` (raporun beslendiği kaynak).

---

## Slayt 7 — Adım 5: Feature Engineering (1 dk)

`02_feature_engineering.py` — 13 özellik (PDF en az 5):

| Tür | Özellikler |
|---|---|
| Zaman | `hour_of_day`, `day_of_week`, `month`, `event_year` |
| Konum | `district`, `ward`, `community_area`, `beat` |
| Koordinat | `latitude`, `longitude` |
| Bağlam | `arrest_int`, `domestic_int` |
| Türetilmiş bool | `is_weekend`, `is_night` |

Her özelliğin **iş mantığı docstring'de** (örn. "battery gece, theft gündüz" hipotezi). `gold/feature_view` Delta — ML-hazır snapshot.

---

## Slayt 8 — Adım 6: 5 ML Modeli + MLflow (2 dk) **[En kritik adım]**

`spark/ml/train_models.py` — 5 model sırayla:

1. **Logistic Regression** (multinomial)
2. **Decision Tree** (maxDepth=10)
3. **Random Forest** (numTrees=50, maxDepth=10)
4. **GBT + OneVsRest** (multi-class wrapper)
5. **Naive Bayes** (gaussian — lat/lon negatif olabildiği için multinomial değil)

**MLflow'a logla:** Params + 5 metrik (accuracy, F1, P, R, AUC-OvR-macro) + train_seconds + **Feature Importance** + **Confusion Matrix** + per-class precision/recall + model artifact + registry.

**Sonuç tablosu:**

| Model | Accuracy | Weighted F1 | AUC | Train (s) |
|---|---|---|---|---|
| 🏆 Random Forest | **0.4537** | 0.381 | **0.697** | 15.7 |
| GBT (OvR) | 0.4514 | 0.381 | — | 60.5 |
| Decision Tree | 0.4419 | 0.377 | 0.680 | 3.9 |
| Logistic Regression | 0.4410 | 0.370 | 0.657 | 9.1 |
| Naive Bayes | 0.4311 | 0.366 | 0.657 | 1.0 |

**Yorumlar:** Majority class baseline (THEFT) %22 → RF iki kat iyileşme. Ağaç-ensemble (RF, GBT) lineer (LR) ve bağımsızlık varsayımlı (NB) modellerden iyi. RF en iyi hız-kalite Pareto.

---

## Slayt 9 — Adım 7: Dashboard + Görseller (1.5 dk)

**Streamlit dashboard** http://localhost:8501 — 5 sekme:
1. **Genel** — top 20 suç tipi + arrest_rate scatter
2. **İlçe** — bar + Mapbox harita (avg lat/lon)
3. **Saat** — saat × tip heatmap + saatlik line
4. **Tahminler** — gerçek vs tahmin + confusion matrix
5. **MLflow** — 5 model karşılaştırma + Pareto

**HTML statik rapor** [`docs/rapor.html`](rapor.html) — PDF zorunlu görsellerin hepsi:
- 5 model grouped bar
- Feature Importance horizontal bar
- Confusion Matrix heatmap
- ROC proxy (per-class precision/recall)
- Yıllık + saatlik + haftalık line chart
- Top 15 histogram
- Tutuklama pie

**Demo komut:** `make dashboard` + tarayıcıdan 8501.

---

## Slayt 10 — Karşılaşılan zorluklar (1.5 dk)

| Sorun | Çözüm |
|---|---|
| Bronze dev batch commit etmiyordu | `maxOffsetsPerTrigger=200K` + worker 2→8 core |
| Silver 67K küçük parquet fan-out | `partitionBy(event_year)` + Delta OPTIMIZE |
| Dashboard count_rows() askıda | Delta `add_actions.num_records` (transaction log) |
| GBT multi-class yok | `OneVsRest` wrapper |
| NB multinomial neg değerlere kırılıyor | `modelType="gaussian"` |
| Veri seti pivot (MovieLens → Crimes) | `feat/chicago-crimes` branch'inde tam rewrite |

---

## Slayt 11 — Tekrarlanabilirlik ve canlı demo (1 dk)

```bash
git clone https://github.com/bigdataKOU/bigdataproje.git
cd bigdataproje && cp .env.example .env
# Crimes.csv'i ../crimes/Crimes.csv'e yerleştir
make verify
make run-all     # ~30-50 dk
make report      # HTML rapor üret (docs/rapor.html)
# Sonra: http://localhost:8501 dashboard
```

Tüm kaynak kod, branch geçmişi ve commit'ler: **https://github.com/bigdataKOU/bigdataproje**

---

## Slayt 12 — Sonuç (0.5 dk)

- ✅ PDF 7 adımının hepsi karşılandı + zorunlu görseller dahil
- ✅ Endüstriyel patterns: medallion (bronze/silver/gold), MLflow registry, Delta ACID
- ✅ 5 model karşılaştırma + Feature Importance + Confusion Matrix + AUC
- ✅ Streamlit + HTML rapor + Markdown sunum
- ✅ Tekrarlanabilir (tek komut `make run-all`)

### Soru zamanı

---

## Teşekkürler 🎓

Sayın hocam **Dr. Ayşe Gül Eker'e**, bu projeyle endüstri-standart bir veri mühendisliği akışını uçtan uca deneyimleme fırsatı verdiği için teşekkür ederiz.

Takım çalışması ve dataset sağlayıcıya — Chicago Police Department CLEAR — teşekkürler.

Bizleri dinlediğiniz için teşekkürler!
