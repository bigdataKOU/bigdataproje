"""
Gold layer: silver tablosundan analitik/ML-hazir agregasyonlar.

Iki gold tablosu uretilir:
  - gold/movie_stats: movieId basina rating count, ortalama, std, son 30 gun
  - gold/user_stats : userId  basina rating count, ortalama, aktif gun sayisi

Bu bir batch job - silver streaming ile birlikte periyodik calistirilir.

Calistirma:
  /opt/app/run.sh /opt/app/jobs/gold_features.py
"""
import sys
from pyspark.sql import functions as F

from _session import (
    build_spark,
    gold_movie_path,
    gold_user_path,
    silver_path,
)


def main() -> int:
    spark = build_spark("gold-features")
    spark.sparkContext.setLogLevel("WARN")

    silver = spark.read.format("delta").load(silver_path())

    movie_stats = (
        silver.groupBy("movieId", "title", "genres")
        .agg(
            F.count("*").alias("rating_count"),
            F.avg("rating").alias("avg_rating"),
            F.stddev_pop("rating").alias("rating_std"),
            F.min("event_time").alias("first_rated_at"),
            F.max("event_time").alias("last_rated_at"),
        )
        .withColumn("popularity_bucket",
                    F.when(F.col("rating_count") >= 1000, "high")
                     .when(F.col("rating_count") >= 100, "medium")
                     .otherwise("low"))
    )

    (movie_stats.write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .save(gold_movie_path()))

    print(f"[gold] movie_stats rows={movie_stats.count()} -> {gold_movie_path()}",
          flush=True)

    user_stats = (
        silver.groupBy("userId")
        .agg(
            F.count("*").alias("rating_count"),
            F.avg("rating").alias("avg_rating"),
            F.countDistinct("event_date").alias("active_days"),
            F.countDistinct("movieId").alias("unique_movies"),
            F.min("event_time").alias("first_rated_at"),
            F.max("event_time").alias("last_rated_at"),
        )
        .withColumn("activity_bucket",
                    F.when(F.col("rating_count") >= 500, "power")
                     .when(F.col("rating_count") >= 50, "active")
                     .otherwise("casual"))
    )

    (user_stats.write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .save(gold_user_path()))

    print(f"[gold] user_stats rows={user_stats.count()} -> {gold_user_path()}",
          flush=True)

    spark.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
