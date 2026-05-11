"""
MLflow registry'den ALS modeli yukleyip top-N onerileri batch olarak
parquet/csv olarak yazar (dashboard'un okuyacagi yer).

models:/... URI ile mlflow.spark.load_model Spark DFS uzerinden "sparkml"
aramaya calisabiliyor ve bos RDD (ValueError) veriyor; bu yuzden son
run'in artifact_uri + als_model klasorunden dosya yolu ile yuklenir.

Calistirma:
  /opt/app/run.sh /opt/app/ml/inference.py
"""
import os
import sys
from typing import Optional
from urllib.parse import urlparse

import mlflow
from pyspark.ml import PipelineModel
from pyspark.ml.recommendation import ALSModel
from pyspark.sql import functions as F

sys.path.insert(0, "/opt/app/jobs")
from _session import build_spark, gold_movie_path  # noqa: E402


MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME = os.environ.get("MLFLOW_MODEL_NAME", "movielens-als-recommender")
MODEL_STAGE = os.environ.get("MLFLOW_MODEL_STAGE", "None")
TOP_K = int(os.environ.get("TOP_K", "20"))
OUT_PATH = os.environ.get("RECS_PATH", "/opt/delta/gold/user_recommendations")


def _artifact_root_from_uri(artifact_uri: str) -> str:
    raw = (artifact_uri or "").rstrip("/")
    parsed = urlparse(raw)
    if parsed.scheme in ("file", ""):
        return parsed.path or raw.replace("file:", "").lstrip("/")
    return raw


def _local_als_model_candidates(client: mlflow.tracking.MlflowClient) -> list[str]:
    if MODEL_STAGE and MODEL_STAGE != "None":
        mvs = client.get_latest_versions(MODEL_NAME, stages=[MODEL_STAGE])
        if not mvs:
            raise RuntimeError(
                f"'{MODEL_NAME}' icin stage='{MODEL_STAGE}' versiyonu yok"
            )
        mv = mvs[0]
    else:
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        if not versions:
            raise RuntimeError(f"registry'de '{MODEL_NAME}' icin versiyon yok")
        mv = max(versions, key=lambda v: int(v.version))

    run = client.get_run(mv.run_id)
    root = _artifact_root_from_uri(run.info.artifact_uri)
    base = os.path.join(root, "als_model")
    if not os.path.isdir(base):
        raise RuntimeError(f"als_model yok: {base} (run_id={mv.run_id})")
    sparkml = os.path.join(base, "sparkml")
    out: list[str] = []
    if os.path.isdir(sparkml):
        out.append(sparkml)
    out.append(base)
    return out


def _file_uri(path: str) -> str:
    ap = os.path.abspath(path)
    return ap if ap.startswith("file:") else f"file://{ap}"


def load_als_for_inference(client: mlflow.tracking.MlflowClient):
    """Pipeline veya duz ALSModel; recommendForAllUsers donduren asama."""
    last_err: Optional[Exception] = None
    for d in _local_als_model_candidates(client):
        uri = _file_uri(d)
        try:
            m = PipelineModel.load(uri)
            for s in m.stages:
                if isinstance(s, ALSModel):
                    print(f"[inference] PipelineModel.load -> ALSModel ({uri})", flush=True)
                    return s
        except Exception as e:
            last_err = e
        try:
            m = ALSModel.load(uri)
            print(f"[inference] ALSModel.load ({uri})", flush=True)
            return m
        except Exception as e:
            last_err = e
    raise RuntimeError("Spark ALS modeli yuklenemedi") from last_err


def main() -> int:
    spark = build_spark("als-inference")
    spark.sparkContext.setLogLevel("WARN")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    if MODEL_STAGE and MODEL_STAGE != "None":
        print(f"[inference] {MODEL_NAME} stage={MODEL_STAGE}", flush=True)
    else:
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        latest = max(versions, key=lambda v: int(v.version))
        print(
            f"[inference] {MODEL_NAME} v{latest.version} run_id={latest.run_id}",
            flush=True,
        )

    model = load_als_for_inference(client)

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
