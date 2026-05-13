"""
Random Forest classifier — Chicago Crimes Primary Type tahmini.

Silver Delta'dan veriyi okur. Hedef değişken (default = primary_type) için
RandomForestClassifier eğitir. Çok sayıda nadir suç tipi olduğu için Top-N
sınıfı tutar, geri kalanını 'OTHER' altında toplar (sınıflandırma stabilize).

Hiperparametreler:
  RF_NUM_TREES (default 50)
  RF_MAX_DEPTH (default 10)
  RF_TOP_N_TYPES (default 10)
  TARGET_COLUMN (default 'primary_type'; 'district' de denenebilir)

Metrikler MLflow'a:
  accuracy, weighted_f1, weighted_precision, weighted_recall, train_seconds

Calistirma (sweep.sh icinden cagiriliyor):
  /opt/app/run.sh /opt/app/ml/train_classifier.py
"""
import os
import sys
import time

import mlflow
import mlflow.spark
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.sql import functions as F

sys.path.insert(0, "/opt/app/jobs")
from _session import build_spark, silver_path  # noqa: E402


MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
EXPERIMENT_NAME = os.environ.get("MLFLOW_EXPERIMENT", "chicago-crimes-classifier")
MODEL_NAME = os.environ.get("MLFLOW_MODEL_NAME", "chicago-crime-type-rf")

RF_NUM_TREES = int(os.environ.get("RF_NUM_TREES", "50"))
RF_MAX_DEPTH = int(os.environ.get("RF_MAX_DEPTH", "10"))
TOP_N_TYPES = int(os.environ.get("RF_TOP_N_TYPES", "10"))
TARGET_COLUMN = os.environ.get("TARGET_COLUMN", "primary_type")
TRAIN_RATIO = float(os.environ.get("TRAIN_RATIO", "0.8"))
SAMPLE_FRACTION = float(os.environ.get("RF_SAMPLE_FRACTION", "1.0"))
RUN_NAME = os.environ.get("RF_RUN_NAME", "")


FEATURE_COLUMNS = [
    "district",
    "ward",
    "community_area",
    "beat",
    "hour_of_day",
    "day_of_week",
    "month",
    "year",
    "latitude",
    "longitude",
    "arrest_int",
    "domestic_int",
]


def main() -> int:
    spark = build_spark("crime-rf-train")
    spark.sparkContext.setLogLevel("WARN")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = (
        spark.read.format("delta").load(silver_path())
        .where(F.col(TARGET_COLUMN).isNotNull())
        .withColumn("arrest_int", F.col("arrest").cast("int"))
        .withColumn("domestic_int", F.col("domestic").cast("int"))
        .na.fill(0, [c for c in FEATURE_COLUMNS if c not in ("arrest_int", "domestic_int")])
    )
    if SAMPLE_FRACTION < 1.0:
        df = df.sample(False, SAMPLE_FRACTION, seed=42)

    # Top-N target değerini tut, kalanı 'OTHER' yap (sınıf dengesi + hız)
    top_values_df = (
        df.groupBy(TARGET_COLUMN)
          .agg(F.count("*").alias("c"))
          .orderBy(F.col("c").desc())
          .limit(TOP_N_TYPES)
    )
    top_values = [r[TARGET_COLUMN] for r in top_values_df.collect()]
    print(f"[rf] top-{TOP_N_TYPES} {TARGET_COLUMN}={top_values}", flush=True)

    df = df.withColumn(
        "label_raw",
        F.when(F.col(TARGET_COLUMN).isin(top_values), F.col(TARGET_COLUMN))
         .otherwise(F.lit("OTHER")),
    ).select(*FEATURE_COLUMNS, "label_raw").cache()
    n_rows = df.count()

    train, test = df.randomSplit([TRAIN_RATIO, 1.0 - TRAIN_RATIO], seed=42)

    run_name = RUN_NAME or f"rf-trees{RF_NUM_TREES}-depth{RF_MAX_DEPTH}"
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({
            "numTrees": RF_NUM_TREES,
            "maxDepth": RF_MAX_DEPTH,
            "top_n_types": TOP_N_TYPES,
            "target_column": TARGET_COLUMN,
            "train_ratio": TRAIN_RATIO,
            "sample_fraction": SAMPLE_FRACTION,
            "n_rows": n_rows,
            "n_features": len(FEATURE_COLUMNS),
        })

        indexer = StringIndexer(
            inputCol="label_raw",
            outputCol="label",
            handleInvalid="keep",
        )
        assembler = VectorAssembler(
            inputCols=FEATURE_COLUMNS,
            outputCol="features",
            handleInvalid="skip",
        )
        rf = RandomForestClassifier(
            featuresCol="features",
            labelCol="label",
            numTrees=RF_NUM_TREES,
            maxDepth=RF_MAX_DEPTH,
            seed=42,
            featureSubsetStrategy="auto",
        )
        pipeline = Pipeline(stages=[indexer, assembler, rf])

        t0 = time.time()
        model = pipeline.fit(train)
        train_seconds = time.time() - t0
        mlflow.log_metric("train_seconds", train_seconds)

        predictions = model.transform(test)

        for metric_name in ("accuracy", "f1", "weightedPrecision", "weightedRecall"):
            evaluator = MulticlassClassificationEvaluator(
                labelCol="label",
                predictionCol="prediction",
                metricName=metric_name,
            )
            value = float(evaluator.evaluate(predictions))
            key = (
                "weighted_f1" if metric_name == "f1"
                else "weighted_precision" if metric_name == "weightedPrecision"
                else "weighted_recall" if metric_name == "weightedRecall"
                else metric_name
            )
            mlflow.log_metric(key, value)
            print(f"[rf] {key}={value:.4f}", flush=True)

        # Sınıf bazında kabaca dağılımı artifact olarak yaz
        class_counts = (
            predictions.groupBy("label_raw")
            .agg(F.count("*").alias("test_rows"))
            .toPandas()
        )
        path = "/tmp/class_distribution.csv"
        class_counts.to_csv(path, index=False)
        mlflow.log_artifact(path)

        mlflow.spark.log_model(
            model,
            artifact_path="rf_model",
            registered_model_name=MODEL_NAME,
        )

        print(
            f"[rf] mlflow run_id={run.info.run_id} train_seconds={train_seconds:.1f}s",
            flush=True,
        )

    spark.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
