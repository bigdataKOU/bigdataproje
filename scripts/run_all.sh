#!/usr/bin/env bash
# Chicago Crimes — uçtan uca pipeline (PDF Adım 1-7).
#
# Akış:
#   1) Servisleri başlat (kafka, spark, mlflow, pipeline, dashboard)
#   2) Bronze ingest streaming (background)
#   3) Producer ile Crimes.csv → Kafka
#   4) Bronze flush bekleme (Delta commit)
#   5) Silver batch (dedup + join, ML-hazır)
#   6) Gold batch (type_stats / district_stats / hourly_stats)
#   7) EDA notebook (gold/eda_overview + summary JSON)
#   8) Feature engineering notebook (gold/feature_view)
#   9) 5 model eğitimi (train_models.py) + MLflow logging
#  10) Inference (en iyi modelle silver üzerinde tahmin)
#  11) Charts üretimi (docs/figures/*.png)
#
# Kullanım:
#   bash scripts/run_all.sh
#   PRODUCER_MAX_RECORDS=2000000 PRODUCER_MODE=burst bash scripts/run_all.sh

set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

: "${PRODUCER_MAX_RECORDS:=2000000}"
: "${PRODUCER_MODE:=burst}"
: "${PRODUCER_RATE:=20000}"
: "${INGEST_WAIT_SECONDS:=300}"
: "${SAMPLE_FRACTION:=0.2}"
: "${TOP_N_TYPES:=5}"
export PRODUCER_MAX_RECORDS PRODUCER_MODE PRODUCER_RATE INGEST_WAIT_SECONDS \
       SAMPLE_FRACTION TOP_N_TYPES

DC="docker compose"
EXEC="${DC} exec -T pipeline"

log()  { printf "\n\033[1;36m== %s ==\033[0m\n" "$*"; }
done_() { printf "\033[1;32m✓ %s\033[0m\n" "$*"; }

# 1) Services
log "1/11 servisleri baslat"
${DC} up -d kafka spark-master spark-worker mlflow pipeline dashboard
until ${DC} ps kafka --format json | grep -q '"Health":"healthy"'; do
    sleep 3
done
${DC} restart pipeline
sleep 12
done_ "services ayakta"

# 2) Bronze
log "2/11 bronze ingest (arka plan)"
${DC} exec -d pipeline bash /opt/app/run.sh /opt/app/jobs/bronze_ingest.py \
    > "${ROOT}/logs-bronze.txt" 2>&1 || true
sleep 5
done_ "bronze running"

# 3) Producer
log "3/11 producer (max=${PRODUCER_MAX_RECORDS} mode=${PRODUCER_MODE})"
${DC} run --rm producer
done_ "producer bitti"

# 4) Bronze flush
log "4/11 bronze ilk commit'i bekleniyor (max ${INGEST_WAIT_SECONDS}s)"
deadline=$(( $(date +%s) + INGEST_WAIT_SECONDS ))
while (( $(date +%s) < deadline )); do
    c=$(${DC} exec -T pipeline ls -1 /opt/delta/bronze/crimes/_delta_log/ \
            2>/dev/null | grep -c "json" || echo 0)
    if (( c >= 3 )); then
        done_ "bronze ${c} commit yazdı"
        break
    fi
    sleep 10
done

# 5) Silver
log "5/11 silver batch"
${DC} exec -T pipeline pkill -f '[b]ronze_ingest.py' 2>/dev/null || true
sleep 4
${EXEC} env SILVER_BATCH_ONCE=1 SPARK_DRIVER_MEMORY=6g \
    bash /opt/app/run.sh /opt/app/jobs/silver_clean.py \
    > "${ROOT}/logs-silver.txt" 2>&1
done_ "silver yazıldı"

# 6) Gold
log "6/11 gold features"
${EXEC} env SPARK_MASTER_URL=local[*] SPARK_DRIVER_MEMORY=4g \
    bash /opt/app/run.sh /opt/app/jobs/gold_features.py \
    > "${ROOT}/logs-gold.txt" 2>&1
done_ "gold yazıldı"

# 7) EDA
log "7/11 EDA notebook"
${EXEC} env SPARK_MASTER_URL=local[*] SPARK_DRIVER_MEMORY=4g \
    bash /opt/app/run.sh /opt/app/notebooks/01_eda.py \
    > "${ROOT}/logs-eda.txt" 2>&1
done_ "EDA tamamlandı"

# 8) Feature engineering
log "8/11 Feature engineering notebook"
${EXEC} env SPARK_MASTER_URL=local[*] SPARK_DRIVER_MEMORY=4g \
    bash /opt/app/run.sh /opt/app/notebooks/02_feature_engineering.py \
    > "${ROOT}/logs-features.txt" 2>&1
done_ "feature view yazıldı"

# 9) Training
log "9/11 5 model eğitimi"
${EXEC} env SPARK_MASTER_URL=local[*] SPARK_DRIVER_MEMORY=6g \
    SAMPLE_FRACTION="${SAMPLE_FRACTION}" TOP_N_TYPES="${TOP_N_TYPES}" \
    bash /opt/app/run.sh /opt/app/ml/train_models.py \
    > "${ROOT}/logs-train.txt" 2>&1
done_ "modeller eğitildi"

# 10) Inference
log "10/11 inference (en iyi modelle)"
${EXEC} env SPARK_MASTER_URL=local[*] SPARK_DRIVER_MEMORY=4g \
    bash /opt/app/run.sh /opt/app/ml/inference.py \
    > "${ROOT}/logs-inference.txt" 2>&1
done_ "tahminler yazıldı"

# 11) Charts
log "11/11 charts üretimi"
if command -v python3 >/dev/null 2>&1; then
    pip install --quiet mlflow matplotlib pandas numpy 2>/dev/null || true
    python3 scripts/make_charts.py > "${ROOT}/logs-charts.txt" 2>&1 || \
        echo "charts hata — manual deneyin"
fi
done_ "charts yazıldı"

echo
echo "  Dashboard : http://localhost:8501"
echo "  MLflow UI : http://localhost:5000"
echo "  Spark UI  : http://localhost:8080"
echo
echo "  Loglar    : ${ROOT}/logs-*.txt"
