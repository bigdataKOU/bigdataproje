#!/usr/bin/env bash
# Static check'ler: hicbir container kaldirmadan, disk yormadan
# tum kodun sentaktik olarak saglam oldugunu dogrula.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

red()    { printf "\033[1;31m%s\033[0m\n" "$*"; }
green()  { printf "\033[1;32m%s\033[0m\n" "$*"; }
yellow() { printf "\033[1;33m%s\033[0m\n" "$*"; }

ok=true

# 1. docker-compose.yml syntax
echo
yellow "1) docker compose config"
if docker compose config --quiet 2>&1; then
    green "   ok"
else
    red "   FAIL"; ok=false
fi

# 2. tum python dosyalari py_compile
echo
yellow "2) python py_compile"
PY_FILES=(
    producer/ratings_producer.py
    spark/jobs/_session.py
    spark/jobs/bronze_ingest.py
    spark/jobs/silver_clean.py
    spark/jobs/gold_features.py
    spark/ml/train_als.py
    spark/ml/inference.py
    dashboard/app.py
)
if python3 -m py_compile "${PY_FILES[@]}" 2>&1; then
    green "   ok (${#PY_FILES[@]} dosya)"
else
    red "   FAIL"; ok=false
fi

# 3. shell script'ler
echo
yellow "3) shell scripts (bash -n)"
for s in spark/run.sh scripts/run_all.sh scripts/verify.sh; do
    if bash -n "$s" 2>&1; then
        green "   ok: $s"
    else
        red "   FAIL: $s"; ok=false
    fi
done

# 4. dockerfile yapisi (opsiyonel - hadolint varsa)
echo
yellow "4) dockerfile parse (basit kontrol)"
for d in producer/Dockerfile spark/Dockerfile dashboard/Dockerfile mlflow/Dockerfile; do
    if [[ -f "$d" ]] && grep -qE "^FROM " "$d"; then
        green "   ok: $d"
    else
        red "   FAIL: $d FROM yok ya da dosya yok"; ok=false
    fi
done

# 5. veriseti dosyalari mevcut mu
echo
yellow "5) MovieLens dataset"
ML_DIR="${ML25M_DIR:-../ml-25m}"
for f in ratings.csv movies.csv tags.csv links.csv; do
    if [[ -f "${ML_DIR}/${f}" ]]; then
        size=$(du -h "${ML_DIR}/${f}" | cut -f1)
        green "   ok: ${f} (${size})"
    else
        red "   FAIL: ${ML_DIR}/${f} bulunamadi"; ok=false
    fi
done

# 6. requirements.txt'ler
echo
yellow "6) requirements.txt versiyonlari"
for r in producer/requirements.txt dashboard/requirements.txt; do
    if [[ -f "$r" ]] && grep -qE "==" "$r"; then
        green "   ok: $r ($(wc -l < "$r") paket)"
    else
        red "   FAIL: $r (versiyon pin yok)"; ok=false
    fi
done

echo
if $ok; then
    green "TUM KONTROLLER GECTI ✓"
    exit 0
else
    red "BAZI KONTROLLER BASARISIZ ✗"
    exit 1
fi
