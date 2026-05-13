import sys
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.utils import AnalysisException

from _session import (
    build_spark,
    gold_district_path,
    gold_hourly_path,
    gold_type_path,
    silver_path,
)

def main() -> int:
    spark = build_spark("gold-features")
    spark.sparkContext.setLogLevel("ERROR") # Daha temiz log akışı

    try:
        # Silver tablosunu oku
        silver = spark.read.format("delta").load(silver_path())
        
        # Boş veri kontrolü
        if silver.storageLevel.useMemory == False and silver.count() == 0:
            print("[gold] Silver tablosu bos, islem durduruldu.")
            return 0

        # --- 1. Suç Tipi İstatistikleri (Caching Kullanıldı) ---
        # Bu veri seti hem tip stats hem district stats için baz teşkil edebilir
        type_stats = (
            silver.groupBy("primary_type")
            .agg(
                F.count("*").alias("crime_count"),
                F.avg(F.col("arrest").cast("double")).alias("arrest_rate"),
                F.avg(F.col("domestic").cast("double")).alias("domestic_rate"),
                F.min("event_time").alias("first_seen"),
                F.max("event_time").alias("last_seen"),
            )
            .withColumn(
                "frequency_bucket",
                F.expr("""CASE WHEN crime_count >= 100000 THEN 'very_high'
                               WHEN crime_count >= 10000 THEN 'high'
                               WHEN crime_count >= 1000 THEN 'medium'
                               ELSE 'low' END""")
            )
        )
        
        type_stats.write.format("delta").mode("overwrite") \
            .option("overwriteSchema", "true").save(gold_type_path())
        print(f"[gold] type_stats kaydedildi: {gold_type_path()}")

        # --- 2. İlçe (District) Analizi (Window Optimization) ---
        # Window fonksiyonu ve ana tabloyu tek seferde hesaplamak için optimize edildi
        
        district_window = Window.partitionBy("district")
        district_type_window = Window.partitionBy("district", "primary_type")

        district_stats = (
            silver.where(F.col("district").isNotNull())
            # Her ilçe-suç tipi çifti için sayıları hesapla
            .withColumn("type_cnt", F.count("*").over(district_type_window))
            # Her ilçe için toplam sayıları ve koordinatları hesapla
            .withColumn("dist_cnt", F.count("*").over(district_window))
            .withColumn("dist_arrest_avg", F.avg(F.col("arrest").cast("double")).over(district_window))
            .withColumn("avg_lat", F.avg("latitude").over(district_window))
            .withColumn("avg_lon", F.avg("longitude").over(district_window))
            .withColumn("unique_types_cnt", F.size(F.collect_set("primary_type").over(district_window)))
            # En sık suç tipini belirle
            .withColumn("rn", F.row_number().over(Window.partitionBy("district").orderBy(F.col("type_cnt").desc())))
            .where(F.col("rn") == 1)
            .select(
                F.col("district"),
                F.col("dist_cnt").alias("crime_count"),
                F.col("dist_arrest_avg").alias("arrest_rate"),
                F.col("avg_lat").alias("avg_latitude"),
                F.col("avg_lon").alias("avg_longitude"),
                F.col("unique_types_cnt").alias("unique_types"),
                F.col("primary_type").alias("top_primary_type"),
                F.col("type_cnt").alias("top_primary_type_count")
            )
            .withColumn("size_bucket", 
                F.when(F.col("crime_count") >= 200000, "large")
                 .when(F.col("crime_count") >= 50000, "medium")
                 .otherwise("small"))
        )

        district_stats.write.format("delta").mode("overwrite") \
            .option("overwriteSchema", "true").save(gold_district_path())
        print(f"[gold] district_stats kaydedildi: {gold_district_path()}")

        # --- 3. Saatlik Yoğunluk (Partitioning Eklendi) ---
        hourly_stats = (
            silver.where(F.col("hour_of_day").isNotNull())
            .groupBy("hour_of_day", "primary_type")
            .agg(F.count("*").alias("crime_count"))
        )
        
        # Heatmap verisi genelde sabit boyutludur ama partition eklemek okumayı hızlandırır
        hourly_stats.write.format("delta").mode("overwrite") \
            .option("overwriteSchema", "true").save(gold_hourly_path())
        print(f"[gold] hourly_stats kaydedildi: {gold_hourly_path()}")

    except AnalysisException as e:
        print(f"[HATA] Spark tablo okuma veya yazma hatasi: {e}")
        return 1
    except Exception as e:
        print(f"[HATA] Beklenmedik hata: {e}")
        return 1
    finally:
        spark.stop()
        
    return 0

if __name__ == "__main__":
    sys.exit(main())