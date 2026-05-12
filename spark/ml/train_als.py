"""
ALS collaborative filtering training + MLflow tracking.

Silver Delta tablosundan rating verisini okur, train/test split yapar,
ALS (rank, regParam, maxIter ile) egitir, RMSE/MAE olcer, modeli ve
metrikleri MLflow'a loglar. Onerilen modeli registry'e kaydeder.

Calistirma:
  /opt/app/run.sh /opt/app/ml/train_als.py
"""
import os
import sys
import time

import mlflow
import mlflow.spark
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import functions as F

sys.path.insert(0, "/opt/app/jobs")
from _session import build_spark, silver_path  # noqa: E402


MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
EXPERIMENT_NAME = os.environ.get("MLFLOW_EXPERIMENT", "movielens-als")
MODEL_NAME = os.environ.get("MLFLOW_MODEL_NAME", "movielens-als-recommender")

ALS_RANK = int(os.environ.get("ALS_RANK", "16"))
ALS_REG = float(os.environ.get("ALS_REG", "0.1"))
ALS_ITER = int(os.environ.get("ALS_ITER", "10"))
TRAIN_RATIO = float(os.environ.get("TRAIN_RATIO", "0.8"))
SAMPLE_FRACTION = float(os.environ.get("ALS_SAMPLE_FRACTION", "1.0"))


def main() -> int:
    spark = build_spark("als-train")
    spark.sparkContext.setLogLevel("WARN")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    ratings = (
        spark.read.format("delta").load(silver_path())
        .select("userId", "movieId", "rating")
        .where(F.col("rating").isNotNull())
    )
    if SAMPLE_FRACTION < 1.0:
        ratings = ratings.sample(False, SAMPLE_FRACTION, seed=42)
    ratings = ratings.cache()
    n_ratings = ratings.count()

    train, test = ratings.randomSplit([TRAIN_RATIO, 1.0 - TRAIN_RATIO], seed=42)

    with mlflow.start_run(run_name=f"als-rank{ALS_RANK}-reg{ALS_REG}") as run:
        mlflow.log_params({
            "rank": ALS_RANK,
            "regParam": ALS_REG,
            "maxIter": ALS_ITER,
            "train_ratio": TRAIN_RATIO,
            "sample_fraction": SAMPLE_FRACTION,
            "n_ratings": n_ratings,
        })

        als = ALS(
            userCol="userId",
            itemCol="movieId",
            ratingCol="rating",
            rank=ALS_RANK,
            regParam=ALS_REG,
            maxIter=ALS_ITER,
            coldStartStrategy="drop",
            nonnegative=True,
            seed=42,
        )

        t0 = time.time()
        model = als.fit(train)
        train_seconds = time.time() - t0
        mlflow.log_metric("train_seconds", train_seconds)

        predictions = model.transform(test).where(F.col("prediction").isNotNull())

        rmse = RegressionEvaluator(
            metricName="rmse", labelCol="rating", predictionCol="prediction"
        ).evaluate(predictions)
        mae = RegressionEvaluator(
            metricName="mae", labelCol="rating", predictionCol="prediction"
        ).evaluate(predictions)

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        print(f"[als] rmse={rmse:.4f} mae={mae:.4f} train={train_seconds:.1f}s",
              flush=True)

        mlflow.spark.log_model(
            model,
            artifact_path="als_model",
            registered_model_name=MODEL_NAME,
        )

        top_k = 10
        user_recs = model.recommendForAllUsers(top_k)
        sample_recs = (user_recs.limit(20)
                                .toPandas())
        sample_recs.to_csv("/tmp/sample_user_recs.csv", index=False)
        mlflow.log_artifact("/tmp/sample_user_recs.csv")

        print(f"[als] mlflow run_id={run.info.run_id}", flush=True)

    spark.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
