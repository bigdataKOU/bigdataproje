# Chicago Crimes — Streaming Tabanlı ML Pipeline

Kocaeli Üniversitesi Bilgisayar Mühendisliği — **BLM442 Büyük Veri Dönem Projesi** (2026 Bahar).

Bu repo, **Kafka → Spark Structured Streaming → Delta Lake → ML (Spark MLlib) → MLflow → Streamlit** akışıyla uçtan-uca konteynerize bir veri mühendisliği + veri bilimi projesi içerir. Veri seti olarak [Chicago Crimes 2001–Present](https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2) (~7.9 milyon kayıt) kullanılmıştır. ML görevi: **suç tipi ve bölge tahmini**.

## Takım

| Ad Soyad | Numara |
|---|---|
| Emre Aytaş | 220202098 |
| Hatice Kübra Kılıçaslan | 220202077 |
| Berker Yiğit | 220202046 |
| Mertcan Kuzey | 240202009 |

## Mimari

```
┌──────────────┐    JSON     ┌────────┐    Structured     ┌─────────────────┐
│  Producer    │────────────▶│ Kafka  │────Streaming─────▶│  Spark (PySpark)│
│ Crimes.csv   │  crimes     │ (KRaft)│                   │  ┌──────────┐   │
└──────────────┘  topic      └────────┘                   │  │ Bronze   │   │
                                                          │  │ Silver   │   │   Delta Lake
                                                          │  │ Gold     │───┼──▶ /opt/delta
                                                          │  └──────────┘   │
                                                          │  ┌──────────┐   │
                                                          │  │ 5 ML mod.│───┼──▶ MLflow
                                                          │  │ Inference│   │     (registry)
                                                          │  └──────────┘   │
                                                          └─────────────────┘
                                                                   │
                                                                   ▼
                                                          ┌─────────────────┐
                                                          │ Streamlit UI    │
                                                          │ :8501           │
                                                          └─────────────────┘
```

### Medallion (Bronze / Silver / Gold)

- **Bronze** — Kafka mesajlarının ham haliyle Delta'ya yazılması (`/opt/delta/bronze/crimes`). `partitionBy(event_date)` + `maxOffsetsPerTrigger=200K`.
- **Silver** — Temizlenmiş + dedupe (id) + null filtre + `hour_of_day`/`day_of_week`/`month` türetilmiş tablo (`/opt/delta/silver/crimes`). `partitionBy(event_year)`.
- **Gold**:
  - `gold/type_stats` — suç tipi başına toplam, arrest_rate, domestic_rate, frequency_bucket
  - `gold/district_stats` — ilçe başına toplam, en sık suç tipi, lat/lon merkezi
  - `gold/hourly_stats` — saat × suç tipi heatmap için
  - `gold/feature_view` — ML-hazır feature tablosu (Adım 5 çıktısı)
  - `gold/eda_overview` — EDA özet metrikleri (Adım 4)
  - `gold/predictions` — en iyi modelin Spark MLflow-yüklü inference çıktısı

## Servisler (docker-compose)

| Servis | Image | Port | Görev |
|---|---|---|---|
| `kafka` | apache/kafka:3.9.0 (KRaft) | 9092/9094 | Streaming broker (3 partition, auto-create topics) |
| `spark-master` | apache/spark:3.5.8-python3 | 7077, 8080 | Cluster master |
| `spark-worker` | apache/spark:3.5.8-python3 | — | 8 core / 6GB executor |
| `mlflow` | custom (python:3.11) | 5000 | Tracking + model registry |
| `producer` | custom | — | `Crimes.csv` → Kafka |
| `pipeline` | custom (PySpark + Delta + MLflow) | — | Spark job runner |
| `dashboard` | custom (Streamlit) | 8501 | Görselleştirme |

## ML görevi: Suç tipi tahmini

**Girdi (12 özellik):**
- Konum: `district`, `ward`, `community_area`, `beat`, `latitude`, `longitude`
- Zaman: `hour_of_day`, `day_of_week`, `month`, `year`
- Bağlam: `arrest_int` (boolean→int), `domestic_int`

**Hedef:** `primary_type` — Top-5 suç tipi + "OTHER" (~6 sınıf).

**5 model karşılaştırması** (PDF Adım 6 zorunlu):
1. Logistic Regression (multinomial)
2. Decision Tree Classifier
3. Random Forest Classifier
4. Gradient Boosted Trees + OneVsRest (multi-class wrapper)
5. Naive Bayes (multinomial)

**Her model için MLflow'a logla:**
- Parametreler (sınıf sayısı, sample fraction, model-spesifik hiperparametreler)
- Metrikler: `accuracy`, `weighted_f1`, `weighted_precision`, `weighted_recall`, `auc_ovr_macro`, `train_seconds`
- **Feature Importance** (CSV artifact + per-feature MLflow metric)
- **Confusion Matrix** (CSV artifact)
- **Per-class precision/recall** (CSV artifact)
- Modelin kendisi (`mlflow.spark.log_model` + registry)

## Hızlı başlangıç

### Ön gereksinimler

- Docker 24+ ve Docker Compose v2
- 8 GB+ RAM, ~12 GB boş disk
- Chicago Crimes CSV (`Crimes.csv`) — proje yanında `../crimes/Crimes.csv` veya `CRIMES_DIR` env ile başka yol

### 1. Kurulum

```bash
git clone https://github.com/bigdataKOU/bigdataproje.git
cd bigdataproje
cp .env.example .env

# Veri seti
mkdir -p ../crimes
# Crimes_-_2001_to_Present.csv'i ../crimes/Crimes.csv olarak yerleştir
```

### Windows kullanıcıları

WSL2 + Docker Desktop önerilir. Repoyu WSL filesystem'i içine klonla (Windows path'i değil — 10× yavaş).

### 2. Statik doğrulama

```bash
make verify
```

### 3. Tek komutla uçtan-uca pipeline

```bash
make run-all
```

Bu otomatik yapar:
1. Servisleri başlatır
2. Bronze streaming background, producer 2M kayıt basar
3. Bronze flush, silver batch, gold batch
4. **EDA** notebook (Adım 4) — `gold/eda_overview` + `eda_summary.json`
5. **Feature engineering** notebook (Adım 5) — `gold/feature_view`
6. **5 ML modeli** eğitilir (Adım 6) — MLflow'a parametre + metrik + FI + CM + model kaydı
7. **Inference** en iyi modelle
8. **Charts** (`docs/figures/*.png`) — PDF zorunlu görseller

İlk çalıştırma ~30-50 dk (image build + 2M ingestion + 5 model). Sonraki çalıştırmalar ~10-15 dk.

**Çıktılar:**
- Dashboard: http://localhost:8501
- MLflow UI: http://localhost:5000
- Spark UI: http://localhost:8080

### 4. Manüel adım adım

```bash
make up                  # kafka + spark + mlflow ayağa
make build               # tüm image'lar
make bronze   &          # streaming, terminal A
make producer            # Crimes'ı bas
make silver              # batch (SILVER_BATCH_ONCE=1)
make gold                # gold tablolar
make eda                 # EDA notebook (Adım 4)
make train               # 5 model + MLflow (Adım 6)
make inference           # en iyi modelle tahmin
make charts              # docs/figures/*.png
make dashboard           # http://localhost:8501
```

## Konfigürasyon (`.env`)

| Değişken | Varsayılan | Açıklama |
|---|---|---|
| `CRIMES_DIR` | `../crimes` | Dataset host yolu (Crimes.csv burada) |
| `PRODUCER_MODE` | `fixed` | `fixed` / `speedup` / `burst` |
| `PRODUCER_RATE` | `1000` | mesaj/sn (fixed mode) |
| `PRODUCER_MAX_RECORDS` | `500000` | Üretilecek max kayıt sayısı |
| `MLFLOW_EXPERIMENT` | `chicago-crimes-classifier` | MLflow deney adı |
| `SAMPLE_FRACTION` | `0.2` | Silver'dan örnek oranı (train_models) |
| `TOP_N_TYPES` | `5` | Top-N primary_type, kalanı OTHER |

## Repo yapısı

```
bigdataproje/
├── docker-compose.yml
├── Makefile
├── .env.example
├── scripts/
│   ├── run_all.sh          ← Tek-komut full pipeline
│   ├── verify.sh           ← Static check'ler
│   └── make_charts.py      ← PDF Adım 7 zorunlu görseller
├── producer/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── crime_producer.py   ← Crimes.csv → Kafka JSON
├── spark/
│   ├── Dockerfile
│   ├── run.sh              ← spark-submit wrapper
│   ├── jobs/
│   │   ├── _session.py
│   │   ├── bronze_ingest.py    ← Kafka stream → Delta bronze
│   │   ├── silver_clean.py     ← bronze→silver (dedup, parse)
│   │   ├── gold_features.py    ← gold tablolar (type/district/hourly)
│   │   └── optimize_silver.py  ← Delta OPTIMIZE compaction
│   ├── notebooks/
│   │   ├── 01_eda.py           ← Adım 4: Keşifsel Veri Analizi
│   │   └── 02_feature_engineering.py  ← Adım 5: Özellik Mühendisliği
│   └── ml/
│       ├── train_models.py     ← Adım 6: 5 model + MLflow + FI + CM
│       └── inference.py        ← En iyi modelle batch tahmin
├── mlflow/Dockerfile           ← Tracking server (SQLite backend)
├── dashboard/                  ← Streamlit
│   ├── Dockerfile
│   ├── requirements.txt
│   └── app.py
└── docs/
    ├── PROGRESS.md         ← Adım adım yapılanlar
    ├── architecture.md     ← Teknik tasarım
    ├── teknik_rapor.md     ← 2-3 sayfa rapor (PDF zorunluluğu)
    └── figures/            ← PNG görseller (Adım 7)
```

## Değerlendirme (PDF 5. bölüm)

| Kriter | Ağırlık | Bu repodaki kanıt |
|---|---|---|
| Docker & Altyapı | %15 | `docker-compose.yml` 7 servis, her servis için `Dockerfile` |
| Kafka Streaming | %15 | `producer/crime_producer.py` + `spark/jobs/bronze_ingest.py` |
| Spark + Delta Lake | %15 | `bronze_ingest.py`, `silver_clean.py`, `gold_features.py`, `optimize_silver.py` |
| EDA & Feature Engineering | %10 | `spark/notebooks/01_eda.py`, `02_feature_engineering.py` (8+ özellik) |
| ML Modelleri & MLflow | %15 | `spark/ml/train_models.py` — 5 model + FI + CM + AUC + registry |
| Dashboard & Görselleştirme | %15 | `dashboard/app.py` (5 sekme) + `scripts/make_charts.py` PNG'leri |
| Dokümantasyon & Sunum | %15 | README, `docs/teknik_rapor.md`, kod yorumları |

## Tasarım kararları

- **Kafka KRaft mode** — Zookeeper bağımlılığı yok, tek container.
- **Delta Lake 3.2** — ACID transactions, streaming MERGE, `OPTIMIZE` compaction.
- **MLflow + SQLite + lokal artifact store** — Self-contained, S3/MinIO gerek yok.
- **5 sınıflandırıcı karşılaştırması** — PDF zorunluluğu. Multi-class olarak: LogReg/DT/RF/NaiveBayes native, GBT için OneVsRest wrapper.
- **Top-N + OTHER label** — Chicago'da 30+ suç tipi var; sınıflandırma stabilitesi için top-5 + OTHER.
- **Streamlit** — Demo için 1 dosyalık dashboard, 5 sekme (Genel, İlçe, Saat, Tahminler, MLflow).
- **WSL2 + Docker Desktop** — Linux Spark/Kafka native değil; WSL filesystem 10× hızlı.

## Lisans / Atıf

Chicago Crimes dataset: City of Chicago, Chicago Police Department's CLEAR system. Veriler "as-is" yayınlanmıştır; herhangi bir uyumsuzluğun sorumluluğu kullanıcıya aittir (resmi disclaimer mevcuttur).
