"""
Bronze layer: Kafka 'crimes' topic'inden ham JSON mesajlari okuyup
Delta tablosuna append eder. Hicbir donusum yok, raw event store.

Calistirma:
  /opt/app/run.sh /opt/app/jobs/bronze_ingest.py
"""
import sys
from pyspark.sql import functions as F
from pyspark.sql.types import (
    BooleanType,
    DoubleType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
)

from _session import (
    KAFKA_BROKER,
    KAFKA_TOPIC_CRIMES,
    bronze_path,
    build_spark,
    CHECKPOINT_PATH,
)


CRIME_SCHEMA = StructType([
    StructField("id", LongType(), False),
    StructField("case_number", StringType(), True),
    StructField("primary_type", StringType(), False),
    StructField("description", StringType(), True),
    StructField("location_description", StringType(), True),
    StructField("arrest", BooleanType(), True),
    StructField("domestic", BooleanType(), True),
    StructField("beat", IntegerType(), True),
    StructField("district", IntegerType(), True),
    StructField("ward", IntegerType(), True),
    StructField("community_area", IntegerType(), True),
    StructField("fbi_code", StringType(), True),
    StructField("year", IntegerType(), True),
    StructField("latitude", DoubleType(), True),
    StructField("longitude", DoubleType(), True),
    StructField("event_time_ms", LongType(), False),
    StructField("ingestedAt", LongType(), True),
])


def main() -> int:
    spark = build_spark("bronze-ingest")
    spark.sparkContext.setLogLevel("WARN")

    raw = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BROKER)
        .option("subscribe", KAFKA_TOPIC_CRIMES)
        .option("startingOffsets", "earliest")
        .option("failOnDataLoss", "false")
        # Micro-batch'i bound et — yoksa dev batch tek seferde commit etmeye
        # calisir ve 2-8 core cluster'da uzun sure hicbir Delta commit olmaz.
        .option("maxOffsetsPerTrigger", "200000")
        .load()
    )

    parsed = (
        raw.select(
            F.col("key").cast("string").alias("kafka_key"),
            F.col("topic").alias("kafka_topic"),
            F.col("partition").alias("kafka_partition"),
            F.col("offset").alias("kafka_offset"),
            F.col("timestamp").alias("kafka_timestamp"),
            F.from_json(F.col("value").cast("string"), CRIME_SCHEMA).alias("payload"),
        )
        .select(
            "kafka_key",
            "kafka_topic",
            "kafka_partition",
            "kafka_offset",
            "kafka_timestamp",
            "payload.*",
        )
        .withColumn("event_time", (F.col("event_time_ms") / 1000.0).cast("timestamp"))
        .withColumn("event_date", F.to_date(F.col("event_time")))
    )

    query = (
        parsed.writeStream.format("delta")
        .outputMode("append")
        .option("checkpointLocation", f"{CHECKPOINT_PATH}/bronze")
        .option("mergeSchema", "true")
        .partitionBy("event_date")
        .trigger(processingTime="10 seconds")
        .start(bronze_path())
    )

    print(f"[bronze] writing to {bronze_path()}", flush=True)
    query.awaitTermination()
    return 0


if __name__ == "__main__":
    sys.exit(main())
