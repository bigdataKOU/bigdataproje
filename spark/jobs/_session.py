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
KAFKA_TOPIC_CRIMES = os.environ.get("KAFKA_TOPIC_CRIMES", "crimes")
DATA_PATH = os.environ.get("DATA_PATH", "/opt/data")


def bronze_path() -> str:
    return f"{DELTA_PATH}/bronze/crimes"


def silver_path() -> str:
    return f"{DELTA_PATH}/silver/crimes"


def gold_type_path() -> str:
    return f"{DELTA_PATH}/gold/type_stats"


def gold_district_path() -> str:
    return f"{DELTA_PATH}/gold/district_stats"


def gold_hourly_path() -> str:
    return f"{DELTA_PATH}/gold/hourly_stats"


def predictions_path() -> str:
    return f"{DELTA_PATH}/gold/predictions"


def crimes_csv_path() -> str:
    return f"{DATA_PATH}/crimes/Crimes.csv"
