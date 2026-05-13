import os
import sys
from pyspark.sql import functions as F
from pyspark.sql.utils import AnalysisException

sys.path.insert(0, "/opt/app/jobs")
from _session import build_spark, silver_path, DELTA_PATH 

FEATURE_VIEW = f"{DELTA_PATH}/gold/feature_view"

def main() -> int:
    os.environ["SPARK_MASTER_URL"] = os.environ.get("SPARK_MASTER_URL", "local[*]")
    spark = build_spark("feature-engineering")
    spark.sparkContext.setLogLevel("ERROR")

    try:
        # Silver tablosunu yükle ve temizle
        silver = spark.read.format("delta").load(silver_path())
        
        # Özellik Mühendisliği Pipeline'ı
        fv = (
            silver.filter(F.col("primary_type").isNotNull())
            # Boş değerleri (null) modelin çökmemesi için doldur (Imputation)
            .fillna({
                "district": 0, "ward": 0, "community_area": 0, "beat": 0,
                "latitude": 41.8781, "longitude": -87.6298 # Chicago merkez koordinatları
            })
            .withColumn("arrest_int", F.col("arrest").cast("int"))
            .withColumn("domestic_int", F.col("domestic").cast("int"))
            # Zaman bazlı yeni özellikler
            .withColumn("is_weekend", F.col("day_of_week").isin(1, 7).cast("int"))
            .withColumn("is_night", F.expr("CASE WHEN hour_of_day >= 22 OR hour_of_day <= 5 THEN 1 ELSE 0 END"))
            # Mevsimsel özellik (Kış ayları etkisi)
            .withColumn("is_winter", F.col("month").isin(12, 1, 2).cast("int"))
            .select(
                "id", "primary_type", "hour_of_day", "day_of_week", "month", 
                "event_year", "is_weekend", "is_night", "is_winter",
                "district", "ward", "community_area", "beat",
                "latitude", "longitude", "arrest_int", "domestic_int"
            )
        )

        # Materyalizasyon
        fv.write.format("delta").mode("overwrite") \
            .option("overwriteSchema", "true") \
            .save(FEATURE_VIEW)
            
        print(f"[features] Feature View başarıyla oluşturuldu ({fv.count()} satır).")
        print(f"[features] Kolonlar: {', '.join(fv.columns)}")

    except AnalysisException as e:
        print(f"[HATA] Delta tablosu erişim hatası: {e}")
        return 1
    except Exception as e:
        print(f"[HATA] Beklenmedik hata: {e}")
        return 1
    finally:
        spark.stop()
    return 0

if __name__ == "__main__":
    sys.exit(main())