"""Delta + Kafka destekli ortak SparkSession factory."""
import os
from pyspark.sql import SparkSession


def build_spark(app_name: str) -> SparkSession:
    builder = (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.shuffle.partitions", "8")
    )

    master = os.environ.get("SPARK_MASTER_URL")
    if master:
        builder = builder.master(master)

    return builder.getOrCreate()


DELTA_PATH = os.environ.get("DELTA_PATH", "/opt/delta")
CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", "/opt/checkpoints")
KAFKA_BROKER = os.environ.get("KAFKA_BROKER", "kafka:9092")
KAFKA_TOPIC_RATINGS = os.environ.get("KAFKA_TOPIC_RATINGS", "ratings")
DATA_PATH = os.environ.get("DATA_PATH", "/opt/data")


def bronze_path() -> str:
    return f"{DELTA_PATH}/bronze/ratings"


def silver_path() -> str:
    return f"{DELTA_PATH}/silver/ratings"


def gold_user_path() -> str:
    return f"{DELTA_PATH}/gold/user_stats"


def gold_movie_path() -> str:
    return f"{DELTA_PATH}/gold/movie_stats"


def movies_csv_path() -> str:
    return f"{DATA_PATH}/ml-25m/movies.csv"
