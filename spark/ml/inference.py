"""
MLflow registry'den en iyi RF modelini (Production veya en son) yukleyip
silver Delta tablosundaki örnek satırlar üzerinde tahmin yapar. Çıktıyı
gold/predictions Delta tablosuna yazar (dashboard buradan okur).

Calistirma:
  /opt/app/run.sh /opt/app/ml/inference.py

Docker compose'da pipeline varsayilan olarak spark://master kullanir;
bu betik icin `SPARK_MASTER_URL=local[*]` ver (run_all.sh boyle cagirir).
"""
import os
import sys

import mlflow
import mlflow.spark
from pyspark.ml import PipelineModel
from pyspark.sql import functions as F

sys.path.insert(0, "/opt/app/jobs")
from _session import build_spark, predictions_path, silver_path  # noqa: E402


MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME = os.environ.get("MLFLOW_MODEL_NAME", "chicago-crime-type-rf")
MODEL_STAGE = os.environ.get("MLFLOW_MODEL_STAGE", "None")
SAMPLE_FRACTION = float(os.environ.get("INFERENCE_SAMPLE_FRACTION", "0.05"))


def _resolve_model_uri(client) -> str:
    if MODEL_STAGE and MODEL_STAGE != "None":
        mvs = client.get_latest_versions(MODEL_NAME, stages=[MODEL_STAGE])
        if not mvs:
            raise RuntimeError(
                f"'{MODEL_NAME}' için stage='{MODEL_STAGE}' versiyonu yok"
            )
        mv = mvs[0]
    else:
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        if not versions:
            raise RuntimeError(f"registry'de '{MODEL_NAME}' versiyonu yok")
        mv = max(versions, key=lambda v: int(v.version))
    return f"runs:/{mv.run_id}/rf_model"


def main() -> int:
    spark = build_spark("crime-rf-inference")
    spark.sparkContext.setLogLevel("WARN")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()

    uri = _resolve_model_uri(client)
    print(f"[inference] loading model {uri}", flush=True)
    model: PipelineModel = mlflow.spark.load_model(uri)

    df = (
        spark.read.format("delta").load(silver_path())
        .where(F.col("primary_type").isNotNull())
        .withColumn("arrest_int", F.col("arrest").cast("int"))
        .withColumn("domestic_int", F.col("domestic").cast("int"))
        .na.fill(0)
    )
    if SAMPLE_FRACTION < 1.0:
        df = df.sample(False, SAMPLE_FRACTION, seed=42)

    # train_classifier ile aynı label_raw kuralı (TOP-N filtreleme model.transform
    # için zaten gerekmez; ama metric karşılaştırma için aynı tutuyoruz).
    df = df.withColumn("label_raw", F.col("primary_type")).cache()
    n = df.count()
    print(f"[inference] {n} satır üzerinde inference", flush=True)

    preds = model.transform(df)

    indexer_model = next(
        (s for s in model.stages if hasattr(s, "labels")),
        None,
    )
    labels = list(indexer_model.labels) if indexer_model is not None else []

    if labels:
        preds = preds.withColumn(
            "predicted_label",
            F.expr(
                "CASE prediction "
                + " ".join(
                    [f"WHEN {i} THEN '{l}'" for i, l in enumerate(labels)]
                )
                + " ELSE 'UNKNOWN' END"
            ),
        )
    else:
        preds = preds.withColumn("predicted_label",
                                 F.col("prediction").cast("string"))

    output = preds.select(
        "id",
        "event_time",
        "district",
        "latitude",
        "longitude",
        F.col("primary_type").alias("actual_primary_type"),
        "predicted_label",
    )

    (output.write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .save(predictions_path()))
    print(
        f"[inference] {output.count()} tahmin yazildi -> {predictions_path()}",
        flush=True,
    )
    spark.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
