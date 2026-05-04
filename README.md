# MovieLens 25M — Streaming Tabanlı Film Öneri Pipeline'ı

Kocaeli Üniversitesi Bilgisayar Mühendisliği — Büyük Veri Dönem Projesi (2026 Bahar).

Bu repo, **Kafka → Spark Structured Streaming → Delta Lake → ALS → MLflow → Streamlit** akışıyla
uçtan-uca konteynerize bir veri mühendisliği + veri bilimi projesi içerir. Veri seti olarak
[MovieLens 25M](https://grouplens.org/datasets/movielens/25m/) (25 milyon rating) kullanılır.

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
│ ratings.csv  │  ratings    │ (KRaft)│                   │  ┌──────────┐   │
└──────────────┘  topic      └────────┘                   │  │ Bronze   │   │
                                                          │  │ Silver   │   │   Delta Lake
                                                          │  │ Gold     │───┼──▶ /opt/delta
                                                          │  └──────────┘   │
                                                          │  ┌──────────┐   │
                                                          │  │ ALS train│───┼──▶ MLflow
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

- **Bronze** — Kafka mesajlarının ham haliyle Delta'ya yazılması (`/opt/delta/bronze/ratings`).
- **Silver** — Temizlenmiş + dedupe + `movies.csv` ile join'lenmiş tablo (`/opt/delta/silver/ratings`).
- **Gold** — Film bazında popülerlik/ortalama (`gold/movie_stats`), kullanıcı bazında aktivite (`gold/user_stats`), öneri tablosu (`gold/user_recommendations`).

## Servisler (docker-compose)

| Servis | Image | Port | Görev |
|---|---|---|---|
| `kafka` | bitnami/kafka:3.7 (KRaft) | 9092/9094 | Streaming broker |
| `spark-master` | bitnami/spark:3.5.1 | 7077, 8080 | Spark cluster master |
| `spark-worker` | bitnami/spark:3.5.1 | — | Spark executor |
| `mlflow` | custom (python:3.11) | 5000 | Tracking + model registry |
| `producer` | custom | — | `ratings.csv` → Kafka |
| `pipeline` | custom (PySpark + Delta) | — | Spark job runner |
| `dashboard` | custom (Streamlit) | 8501 | Görselleştirme |

## Hızlı başlangıç

### Ön gereksinimler

- Docker 24+ ve Docker Compose v2
- 8 GB+ RAM, **~10 GB boş disk** (image'lar + delta tabloları + Maven cache için)
- MovieLens 25M dataseti `../ml-25m/` (proje kardeşi) ya da `ML25M_DIR` env ile başka yol

### 1. Kurulum

```bash
git clone https://github.com/bigdataKOU/bigdataproje.git
cd bigdataproje
cp .env.example .env

# Veri seti — projenin yanında ../ml-25m olarak duracak şekilde:
cd .. && wget https://files.grouplens.org/datasets/movielens/ml-25m.zip
unzip ml-25m.zip && cd bigdataproje
```

### 2. Hızlı doğrulama (disk yormadan)

Tüm kodun sentaktik olarak sağlam olduğunu, dataset'in yerinde olduğunu kontrol et:

```bash
make verify
```

### 3. Tek komutla uçtan-uca pipeline

```bash
make run-all
```

Bu:
1. Tüm servisleri başlatır (kafka, spark master+worker, mlflow, pipeline, dashboard)
2. Bronze streaming job'unu arka planda başlatır
3. Producer ile 50K rating'i Kafka'ya basar (default: 1000 msg/sec)
4. Silver streaming job'u başlatır
5. 60 sn veri birikmesi için bekler
6. Gold batch + ALS train + Inference çalıştırır
7. Dashboard'u açık bırakır

İlk çalıştırma ~10-15 dakika (image build + Maven JAR cache). Sonraki çalıştırmalar ~2-3 dk.

**Çıktılar:**
- Dashboard: http://localhost:8501
- MLflow UI: http://localhost:5000
- Spark UI: http://localhost:8080

### 4. Manüel adım adım (debug için)

```bash
make up                  # kafka + spark + mlflow ayağa
make build               # tüm image'lar
make bronze   &          # streaming, terminal A
make producer            # ratings'i bas
make silver   &          # streaming, terminal B
make gold                # batch
make train               # ALS + MLflow run
make inference           # öneri tablosu
make dashboard           # http://localhost:8501
```

## Konfigürasyon (`.env`)

| Değişken | Varsayılan | Açıklama |
|---|---|---|
| `ML25M_DIR` | `../ml-25m` | Dataset host yolu |
| `PRODUCER_MODE` | `fixed` | `fixed` / `speedup` / `burst` |
| `PRODUCER_RATE` | `1000` | mesaj/sn (fixed mode) |
| `PRODUCER_SPEEDUP` | `100000` | Timestamp hızlandırma (speedup mode) |
| `PRODUCER_MAX_RECORDS` | `500000` | Üretilecek max rating sayısı |
| `ALS_RANK` | `16` | ALS latent factor sayısı |
| `ALS_REG` | `0.1` | Regularizasyon |
| `ALS_ITER` | `10` | İterasyon sayısı |
| `ALS_SAMPLE_FRACTION` | `1.0` | Train sample oranı (büyük data için 0.1) |

## Repo yapısı

```
bigdataproje/
├── docker-compose.yml
├── Makefile
├── .env.example
├── fixes.txt                 ← Karsilasilan hatalar + cozumler
├── scripts/
│   ├── run_all.sh            ← Tek-komut full pipeline
│   └── verify.sh             ← Static check'ler
├── producer/                 ← Kafka producer
│   ├── Dockerfile
│   ├── requirements.txt
│   └── ratings_producer.py
├── spark/                    ← Spark/Delta jobs
│   ├── Dockerfile
│   ├── run.sh                ← spark-submit wrapper
│   ├── jobs/
│   │   ├── _session.py
│   │   ├── bronze_ingest.py
│   │   ├── silver_clean.py
│   │   └── gold_features.py
│   └── ml/
│       ├── train_als.py
│       └── inference.py
├── mlflow/Dockerfile         ← Tracking server
├── dashboard/                ← Streamlit
│   ├── Dockerfile
│   ├── requirements.txt
│   └── app.py
├── notebooks/01_eda.ipynb    ← Kesifsel veri analizi
└── docs/
    ├── PROGRESS.md           ← Adim adim ne yapildigi
    └── architecture.md
```

## Tasarım kararları

- **Kafka KRaft mode** — Zookeeper'a bağımlılık yok, tek container.
- **Delta Lake 3.2** — ACID + time-travel + streaming MERGE.
- **MLflow + SQLite + lokal artifact store** — Container-içi self-contained, S3/MinIO'ya gerek yok.
- **ALS** — MovieLens için klasik benchmark. Spark ML implicit + explicit destekler.
- **Streamlit** — Demo için 1 dosyalık dashboard, MLflow run karşılaştırması dahil.

## Lisans / Atıf

MovieLens veriseti: F. Maxwell Harper and Joseph A. Konstan. 2015. *The MovieLens Datasets: History and Context.* ACM TiiS.
