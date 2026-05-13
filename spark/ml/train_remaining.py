"""
RF/GBT/NB için OOM-güvenli retry. SAMPLE_FRACTION=0.3 + ufak params.
"""
import os
import sys
import time

import mlflow
import mlflow.spark
from pyspark.ml.classification import (
    GBTClassifier, NaiveBayes, OneVsRest, RandomForestClassifier,
)
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.sql import functions as F

sys.path.insert(0, "/opt/app/jobs")
sys.path.insert(0, "/opt/app/ml")
from _session import build_spark, silver_path  # noqa
from train_models import FEATURE_COLUMNS, log_per_model  # noqa

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
EXPERIMENT_NAME = os.environ.get("MLFLOW_EXPERIMENT", "chicago-crimes-classifier")
SAMPLE_FRACTION = float(os.environ.get("SAMPLE_FRACTION", "0.3"))
TOP_N_TYPES = int(os.environ.get("TOP_N_TYPES", "5"))
TRAIN_RATIO = 0.8


def prep(spark):
    df = (spark.read.format("delta").load(silver_path())
          .where(F.col("primary_type").isNotNull())
          .withColumn("arrest_int", F.col("arrest").cast("int"))
          .withColumn("domestic_int", F.col("domestic").cast("int"))
          .na.fill(0, FEATURE_COLUMNS))
    if SAMPLE_FRACTION < 1.0:
        df = df.sample(False, SAMPLE_FRACTION, seed=42)
    top_values = [r["primary_type"] for r in
                  df.groupBy("primary_type").agg(F.count("*").alias("c"))
                    .orderBy(F.col("c").desc()).limit(TOP_N_TYPES).collect()]
    df = df.withColumn(
        "label_raw",
        F.when(F.col("primary_type").isin(top_values), F.col("primary_type"))
         .otherwise(F.lit("OTHER")),
    ).select(*FEATURE_COLUMNS, "label_raw")
    idx = StringIndexer(inputCol="label_raw", outputCol="label", handleInvalid="keep").fit(df)
    labels = list(idx.labels)
    df = idx.transform(df)
    df = VectorAssembler(inputCols=FEATURE_COLUMNS, outputCol="features",
                         handleInvalid="skip").transform(df)
    df = df.select("features", "label", "label_raw").cache()
    n = df.count()
    train, test = df.randomSplit([TRAIN_RATIO, 1.0 - TRAIN_RATIO], seed=42)
    return train, test, labels, n


def run_one(name, estimator, params, train, test, labels, n):
    with mlflow.start_run(run_name=f"model-{name}") as run:
        try:
            t0 = time.time()
            model = estimator.fit(train)
            sec = time.time() - t0
            preds = model.transform(test)
            log_per_model(name, model, preds, labels, sec,
                          params={**params, "sample_fraction": SAMPLE_FRACTION, "n_rows": n})
            print(f"[{name}] OK ({sec:.1f}s) run_id={run.info.run_id}", flush=True)
        except Exception as e:
            mlflow.log_param("error", str(e)[:500])
            print(f"[{name}] HATA: {e}", flush=True)


def main():
    spark = build_spark("retry-rf-gbt-nb")
    spark.sparkContext.setLogLevel("WARN")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    train, test, labels, n = prep(spark)
    print(f"[data] train+test rows={n}, classes={len(labels)}", flush=True)

    # RF — küçültülmüş: 40 ağaç, depth 10
    run_one("random_forest",
            RandomForestClassifier(labelCol="label", featuresCol="features",
                                   numTrees=40, maxDepth=10, seed=42),
            {"numTrees": 40, "maxDepth": 10}, train, test, labels, n)

    # GBT OvR — küçültülmüş: 15 iter, depth 4
    run_one("gbt_ovr",
            OneVsRest(labelCol="label", featuresCol="features",
                      classifier=GBTClassifier(labelCol="label", featuresCol="features",
                                               maxIter=15, maxDepth=4, seed=42)),
            {"gbt_maxIter": 15, "gbt_maxDepth": 4, "wrapper": "OneVsRest"},
            train, test, labels, n)

    # Naive Bayes gaussian
    run_one("naive_bayes",
            NaiveBayes(labelCol="label", featuresCol="features", modelType="gaussian"),
            {"modelType": "gaussian"}, train, test, labels, n)

    spark.stop()


if __name__ == "__main__":
    sys.exit(main() or 0)
