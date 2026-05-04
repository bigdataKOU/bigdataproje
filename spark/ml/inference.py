"""
MLflow registry'den ALS modeli yukleyip top-N onerileri batch olarak
parquet/csv olarak yazar (dashboard'un okuyacagi yer).

Calistirma:
  /opt/app/run.sh /opt/app/ml/inference.py
"""
import os
import sys

import mlflow
from pyspark.sql import functions as F

sys.path.insert(0, "/opt/app/jobs")
from _session import build_spark, gold_movie_path  # noqa: E402


MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME = os.environ.get("MLFLOW_MODEL_NAME", "movielens-als-recommender")
MODEL_STAGE = os.environ.get("MLFLOW_MODEL_STAGE", "None")
TOP_K = int(os.environ.get("TOP_K", "20"))
OUT_PATH = os.environ.get("RECS_PATH", "/opt/delta/gold/user_recommendations")


def latest_model_uri() -> str:
    if MODEL_STAGE and MODEL_STAGE != "None":
        return f"models:/{MODEL_NAME}/{MODEL_STAGE}"

    client = mlflow.tracking.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    if not versions:
        raise RuntimeError(f"registry'de '{MODEL_NAME}' icin versiyon yok")
    latest = max(versions, key=lambda v: int(v.version))
    return f"models:/{MODEL_NAME}/{latest.version}"


def main() -> int:
    spark = build_spark("als-inference")
    spark.sparkContext.setLogLevel("WARN")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    uri = latest_model_uri()
    print(f"[inference] loading {uri}", flush=True)
    model = mlflow.spark.load_model(uri)

    user_recs = model.recommendForAllUsers(TOP_K)

    flat = (
        user_recs.select(
            "userId",
            F.posexplode("recommendations").alias("rank", "rec"),
        )
        .select(
            "userId",
            (F.col("rank") + F.lit(1)).alias("rank"),
            F.col("rec.movieId").alias("movieId"),
            F.col("rec.rating").alias("predicted_rating"),
        )
    )

    movie_meta = (
        spark.read.format("delta").load(gold_movie_path())
        .select("movieId", "title", "genres", "rating_count", "avg_rating")
    )

    enriched = flat.join(F.broadcast(movie_meta), on="movieId", how="left")

    (enriched.write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .save(OUT_PATH))

    print(f"[inference] wrote recommendations -> {OUT_PATH}", flush=True)
    spark.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
