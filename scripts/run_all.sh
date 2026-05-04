#!/usr/bin/env bash
# Tek komut: tum pipeline'i ucundan ucuna calistirir.
#
# Akis:
#   1. Servisleri baslat (kafka, spark cluster, mlflow, pipeline, dashboard)
#   2. Bronze streaming job'i pipeline icinde arka planda baslat
#   3. Producer'i calistir (PRODUCER_MAX_RECORDS kadar mesaj basar)
#   4. Silver streaming job'i pipeline icinde arka planda baslat
#   5. ~60 sn bekle (bronze + silver birikme)
#   6. Gold batch job
#   7. ALS train
#   8. Inference
#   9. Ozet (URL'ler ve son durum)
#
# Kullanim:
#   bash scripts/run_all.sh                         # default 50K records, fixed 1000/s
#   PRODUCER_MAX_RECORDS=10000 bash scripts/run_all.sh
#   PRODUCER_MODE=burst PRODUCER_MAX_RECORDS=200000 bash scripts/run_all.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

# ---- konfigurasyon ----
: "${PRODUCER_MAX_RECORDS:=50000}"
: "${PRODUCER_MODE:=fixed}"
: "${PRODUCER_RATE:=1000}"
: "${ALS_SAMPLE_FRACTION:=1.0}"
: "${INGEST_WAIT_SECONDS:=60}"
export PRODUCER_MAX_RECORDS PRODUCER_MODE PRODUCER_RATE ALS_SAMPLE_FRACTION

DC="docker compose"
EXEC="${DC} exec -T pipeline"

log()  { printf "\n\033[1;36m== %s ==\033[0m\n" "$*"; }
done_() { printf "\033[1;32m✓ %s\033[0m\n" "$*"; }

# ---- 1. servisleri baslat ----
log "1/9  servisleri baslatiyorum"
${DC} up -d kafka spark-master spark-worker mlflow pipeline dashboard

log "kafka healthy bekleniyor"
until ${DC} ps kafka --format json | grep -q '"Health":"healthy"'; do
    sleep 3
done
done_ "kafka hazir"

# pipeline icinde kucuk delay (ivy cache ilk kullanim icin warm-up)
sleep 5

# ---- 2. bronze streaming arka planda ----
log "2/9  bronze ingest baslatiliyor (arka plan)"
${DC} exec -d pipeline /opt/app/run.sh /opt/app/jobs/bronze_ingest.py \
    > "${ROOT}/logs-bronze.txt" 2>&1 || true
sleep 3
done_ "bronze running"

# ---- 3. producer (foreground) ----
log "3/9  producer (max=${PRODUCER_MAX_RECORDS} mode=${PRODUCER_MODE})"
${DC} run --rm producer
done_ "producer bitti"

# ---- 4. silver streaming arka planda ----
log "4/9  silver streaming baslatiliyor (arka plan)"
${DC} exec -d pipeline /opt/app/run.sh /opt/app/jobs/silver_clean.py \
    > "${ROOT}/logs-silver.txt" 2>&1 || true
sleep 3
done_ "silver running"

# ---- 5. veri birikmesi icin bekle ----
log "5/9  ${INGEST_WAIT_SECONDS} sn bronze/silver birikmesi"
sleep "${INGEST_WAIT_SECONDS}"

# ---- 6. gold ----
log "6/9  gold features (batch)"
${EXEC} /opt/app/run.sh /opt/app/jobs/gold_features.py
done_ "gold yazildi"

# ---- 7. train ALS ----
log "7/9  ALS egitim + MLflow"
${EXEC} env ALS_SAMPLE_FRACTION="${ALS_SAMPLE_FRACTION}" \
    /opt/app/run.sh /opt/app/ml/train_als.py
done_ "ALS egitildi"

# ---- 8. inference ----
log "8/9  inference (top-N oneriler)"
${EXEC} /opt/app/run.sh /opt/app/ml/inference.py
done_ "oneriler yazildi"

# ---- 9. ozet ----
log "9/9  hazir"
echo
echo "  Dashboard : http://localhost:8501"
echo "  MLflow UI : http://localhost:5000"
echo "  Spark UI  : http://localhost:8080"
echo
echo "  Loglar:"
echo "    bronze    -> tail -f ${ROOT}/logs-bronze.txt"
echo "    silver    -> tail -f ${ROOT}/logs-silver.txt"
echo "    pipeline  -> docker compose logs -f pipeline"
echo
echo "  Streaming job'lari durdurmak icin:"
echo "    docker compose down"
