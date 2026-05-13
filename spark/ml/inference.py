"""
MLflow'da kayitli 5 modelden EN IYI olani (accuracy) sec, silver'da ornek
satirlar uzerinde inference yap, gold/predictions Delta tablosuna yaz.

Calistirma:
  /opt/app/run.sh /opt/app/ml/inference.py
"""
import os
import sys

import mlflow
import mlflow.spark
from pyspark.ml import PipelineModel
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.sql import functions as F

sys.path.insert(0, "/opt/app/jobs")
from _session import build_spark, predictions_path, silver_path  # noqa: E402


MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
EXPERIMENT_NAME = os.environ.get("MLFLOW_EXPERIMENT", "chicago-crimes-classifier")
SAMPLE_FRACTION = float(os.environ.get("INFERENCE_SAMPLE_FRACTION", "0.02"))
TOP_N_TYPES = int(os.environ.get("TOP_N_TYPES", "5"))


FEATURE_COLUMNS = [
    "district", "ward", "community_area", "beat",
    "hour_of_day", "day_of_week", "month", "year",
    "latitude", "longitude", "arrest_int", "domestic_int",
]


def find_best_run(client):
    exp = client.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        raise RuntimeError(f"experiment yok: {EXPERIMENT_NAME}")
    runs = client.search_runs([exp.experiment_id], max_results=50)
    best = None
    for r in runs:
        acc = r.data.metrics.get("accuracy")
        if acc is None:
            continue
        if best is None or acc > best[0]:
            best = (acc, r)
    if best is None:
        raise RuntimeError("hicbir run'da accuracy yok")
    return best


def main() -> int:
    spark = build_spark("crime-inference")
    spark.sparkContext.setLogLevel("WARN")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()

    acc, best_run = find_best_run(client)
    model_type = best_run.data.params.get("model_type", "model")
    run_id = best_run.info.run_id
    artifact_path = f"{model_type}_model"
    uri = f"runs:/{run_id}/{artifact_path}"
    print(
        f"[inference] best={model_type} accuracy={acc:.4f} run_id={run_id}",
        flush=True,
    )
    print(f"[inference] loading {uri}", flush=True)
    model = mlflow.spark.load_model(uri)

    df = (
        spark.read.format("delta").load(silver_path())
        .where(F.col("primary_type").isNotNull())
        .withColumn("arrest_int", F.col("arrest").cast("int"))
        .withColumn("domestic_int", F.col("domestic").cast("int"))
        .na.fill(0, FEATURE_COLUMNS)
    )
    if SAMPLE_FRACTION < 1.0:
        df = df.sample(False, SAMPLE_FRACTION, seed=42)

    # train_models'a ozdes sema: top-N + OTHER + StringIndex + VectorAssemble
    top_values_df = (
        df.groupBy("primary_type")
          .agg(F.count("*").alias("c"))
          .orderBy(F.col("c").desc())
          .limit(TOP_N_TYPES)
    )
    top_values = [r["primary_type"] for r in top_values_df.collect()]
    df = df.withColumn(
        "label_raw",
        F.when(F.col("primary_type").isin(top_values), F.col("primary_type"))
         .otherwise(F.lit("OTHER")),
    )
    indexer_model = StringIndexer(
        inputCol="label_raw", outputCol="label", handleInvalid="keep",
    ).fit(df)
    labels = list(indexer_model.labels)
    df = indexer_model.transform(df)
    df = VectorAssembler(
        inputCols=FEATURE_COLUMNS, outputCol="features", handleInvalid="skip",
    ).transform(df).cache()
    n = df.count()
    print(f"[inference] {n} satir, {len(labels)} sinif", flush=True)

    preds = model.transform(df)
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

    output = preds.select(
        F.col("label_raw").alias("actual_label"),
        "predicted_label",
        "district", "latitude", "longitude",
        "hour_of_day", "day_of_week",
    )

    (output.write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .save(predictions_path()))
    n_out = output.count()
    print(f"[inference] {n_out} tahmin yazildi -> {predictions_path()}",
          flush=True)
    spark.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
