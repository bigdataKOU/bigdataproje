"""Delta + Kafka destekli ortak SparkSession factory (Gelişmiş Yapılandırma)."""
import os
import logging
from pyspark.sql import SparkSession

# Logging yapılandırması
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def build_spark(app_name: str) -> SparkSession:
    """
    Spark oturumunu optimize edilmiş Delta ve Kafka ayarlarıyla oluşturur.
    """
    try:
        builder = (
            SparkSession.builder.appName(app_name)
            # Delta Lake Temel Ayarları
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
            .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
            
            # Performans ve Bellek Optimizasyonu
            .config("spark.sql.adaptive.enabled", "true") # AQE: Dinamik sorgu optimizasyonu
            .config("spark.sql.shuffle.partitions", "8")  # Küçük-orta ölçekli veri için ideal
            .config("spark.driver.memory", "2g")          # Driver çökmesini önlemek için
            .config("spark.executor.memory", "2g")
            
            # Delta Lake Hızlandırma
            .config("spark.databricks.delta.retentionDurationCheck.enabled", "false")
            .config("spark.sql.sources.parallelPartitionDiscovery.threshold", "1000")
        )

        # Cluster/Local Mod Seçimi
        master = os.environ.get("SPARK_MASTER_URL")
        if master:
            builder = builder.master(master)
            logger.info(f"Spark Master: {master} üzerinde başlatılıyor.")
        else:
            builder = builder.master("local[*]")
            logger.info("Spark Local modda başlatılıyor.")

        spark = builder.getOrCreate()
        return spark

    except Exception as e:
        logger.error(f"SparkSession baslatilamadi: {e}")
        raise

# --- Çevresel Değişkenler ve Yol Tanımlamaları ---
DELTA_PATH = os.environ.get("DELTA_PATH", "/opt/delta").rstrip("/")
CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", "/opt/checkpoints").rstrip("/")
KAFKA_BROKER = os.environ.get("KAFKA_BROKER", "kafka:9092")
KAFKA_TOPIC_CRIMES = os.environ.get("KAFKA_TOPIC_CRIMES", "crimes")
DATA_PATH = os.environ.get("DATA_PATH", "/opt/data").rstrip("/")

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

def get_checkpoint_dir(job_name: str) -> str:
    """Her iş için benzersiz bir checkpoint klasörü sağlar."""
    return f"{CHECKPOINT_PATH}/{job_name}"