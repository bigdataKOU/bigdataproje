"""
Sadece NaiveBayes yeniden egitir (digerleri zaten MLflow'da).
NB ilk denemede multinomial + negatif longitude yuzunden cokmustu;
gaussian modelType ile retry.
"""
import os
import sys
import time

import mlflow
import mlflow.spark
import numpy as np
import pandas as pd
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.sql import functions as F

sys.path.insert(0, "/opt/app/jobs")
sys.path.insert(0, "/opt/app/ml")
from _session import build_spark, silver_path  # noqa: E402
from train_models import (  # noqa: E402
    FEATURE_COLUMNS,
    confusion_matrix,
    feature_importances_from,
    log_per_model,
    multi_class_auc,
    per_class_metrics,
    EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    SAMPLE_FRACTION,
    TOP_N_TYPES,
    TRAIN_RATIO,
)


def main() -> int:
    spark = build_spark("crime-nb-retry")
    spark.sparkContext.setLogLevel("WARN")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = (
        spark.read.format("delta").load(silver_path())
        .where(F.col("primary_type").isNotNull())
        .withColumn("arrest_int", F.col("arrest").cast("int"))
        .withColumn("domestic_int", F.col("domestic").cast("int"))
        .na.fill(0, FEATURE_COLUMNS)
    )
    if SAMPLE_FRACTION < 1.0:
        df = df.sample(False, SAMPLE_FRACTION, seed=42)

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
    ).select(*FEATURE_COLUMNS, "label_raw")

    indexer_model = StringIndexer(
        inputCol="label_raw", outputCol="label", handleInvalid="keep",
    ).fit(df)
    labels = list(indexer_model.labels)
    df = indexer_model.transform(df)
    df = VectorAssembler(
        inputCols=FEATURE_COLUMNS, outputCol="features", handleInvalid="skip",
    ).transform(df).select("features", "label", "label_raw").cache()
    n = df.count()
    train, test = df.randomSplit([TRAIN_RATIO, 1.0 - TRAIN_RATIO], seed=42)

    with mlflow.start_run(run_name="model-naive_bayes") as run:
        nb = NaiveBayes(
            labelCol="label", featuresCol="features", modelType="gaussian",
        )
        t0 = time.time()
        model = nb.fit(train)
        sec = time.time() - t0
        preds = model.transform(test)
        log_per_model(
            "naive_bayes", model, preds, labels, sec,
            params={
                "modelType": "gaussian",
                "top_n_types": TOP_N_TYPES,
                "train_ratio": TRAIN_RATIO,
                "sample_fraction": SAMPLE_FRACTION,
                "n_rows": n,
            },
        )
        print(f"[nb-retry] run_id={run.info.run_id} ({sec:.1f}s)", flush=True)

    spark.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
