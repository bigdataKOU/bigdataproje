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
- 8 GB+ RAM önerilir
- MovieLens 25M dataseti `data/ml-25m/` altında (zip indirilip çıkarılmış)

### 1. Kurulum

```bash
git clone https://github.com/bigdataKOU/bigdataproje.git
cd bigdataproje
cp .env.example .env

# Veri seti
mkdir -p data && cd data
wget https://files.grouplens.org/datasets/movielens/ml-25m.zip
unzip ml-25m.zip
cd ..
```

### 2. Servisleri başlat

```bash
docker compose up -d kafka spark-master spark-worker mlflow
docker compose logs -f mlflow   # 'Listening at: http://0.0.0.0:5000' gorunmeli
```

### 3. Bronze streaming + Producer

```bash
# bronze ingest job'i baslat (terminal A)
docker compose up -d pipeline
docker compose exec pipeline /opt/app/run.sh /opt/app/jobs/bronze_ingest.py

# producer'i baslat (terminal B) - 500K rating'i hizli replay eder
docker compose up producer
```

### 4. Silver + Gold

Bronze biraz veri biriktirdikten sonra (örn. 30 saniye):

```bash
# silver - streaming, surekli calisir
docker compose exec pipeline /opt/app/run.sh /opt/app/jobs/silver_clean.py

# gold - batch, ihtiyac duydukca calistir
docker compose exec pipeline /opt/app/run.sh /opt/app/jobs/gold_features.py
```

### 5. ALS modeli + MLflow

```bash
docker compose exec pipeline /opt/app/run.sh /opt/app/ml/train_als.py
# MLflow UI: http://localhost:5000
```

### 6. Inference + Dashboard

```bash
docker compose exec pipeline /opt/app/run.sh /opt/app/ml/inference.py
docker compose up -d dashboard
# Dashboard: http://localhost:8501
```

## Konfigürasyon (`.env`)

| Değişken | Varsayılan | Açıklama |
|---|---|---|
| `PRODUCER_SPEEDUP` | `100000` | Timestamp hızlandırma faktörü |
| `PRODUCER_MAX_RECORDS` | `500000` | Üretilecek max rating sayısı |
| `ALS_RANK` | `16` | ALS latent factor sayısı |
| `ALS_REG` | `0.1` | Regularizasyon |
| `ALS_ITER` | `10` | İterasyon sayısı |
| `ALS_SAMPLE_FRACTION` | `1.0` | Train sample oranı (büyük data için 0.1) |

## Repo yapısı

```
bigdataproje/
├── docker-compose.yml
├── .env.example
├── data/                     ← ml-25m (gitignore)
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
