#!/usr/bin/env bash
set -euo pipefail

# Spark job submit helper.
#
# Kullanim:
#   ./run.sh jobs/bronze_ingest.py
#   ./run.sh ml/train_als.py
#
# Tum gerekli jar'lar (Delta + Kafka + MLflow) ve Delta config'leri burada.

JOB="${1:?usage: run.sh <python-script>}"
shift || true

PACKAGES="io.delta:delta-spark_2.12:3.2.0,org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1,org.mlflow:mlflow-spark_2.12:2.16.0"
MASTER="${SPARK_MASTER_URL:-local[*]}"

exec /opt/spark/bin/spark-submit \
    --master "${MASTER}" \
    --packages "${PACKAGES}" \
    --conf spark.jars.ivy=/opt/ivy-cache \
    --conf spark.sql.extensions=io.delta.sql.DeltaSparkSessionExtension \
    --conf spark.sql.catalog.spark_catalog=org.apache.spark.sql.delta.catalog.DeltaCatalog \
    --conf spark.sql.adaptive.enabled=true \
    --conf spark.driver.memory=2g \
    --conf spark.executor.memory=2g \
    --conf spark.driver.host=pipeline \
    --conf spark.driver.bindAddress=0.0.0.0 \
    "${JOB}" "$@"
