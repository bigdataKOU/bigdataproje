"""
Adim 5: Ozellik Muhendisligi (Feature Engineering) — Chicago Crimes.

Bu betik, ML modeline beslenecek anlamli ozellikleri ozetler ve sayisal/
boolean alanlari hazirlayan adimlari "feature view" Delta tablosu olarak
materyalize eder.

Tasarlanan ozellikler (PDF kuralı: en az 5):

  1) hour_of_day   (silver_clean'de turetilir)
       Mantik: Suclar gun icindeki saate gore degisir (theft ogleden sonra,
       battery gece). Sınıflandırıcı icin guclu sinyal.
  2) day_of_week   (silver_clean'de turetilir; 1=Sunday, 7=Saturday)
       Mantik: Hafta sonu vs hafta ici suç dagılımları farklı.
  3) month
       Mantik: Mevsimsel etki — kis aylarında bazi suç tipleri azalır.
  4) event_year
       Mantik: Pandemi ve genel trendler nedeniyle yıl bazlı kalıplar var.
  5) district / ward / community_area / beat
       Mantik: Konum hierarsisi — district kaba, beat ince granularity.
  6) latitude / longitude
       Mantik: Modelin koordinatlardan komsu suç oruntulerini ogrenmesini
       saglar (RF/DT split bunlari kullanir).
  7) arrest_int  (boolean -> int)
       Mantik: Olayin tutuklama ile sonuclanmis olmasi, suç tipi
       sınıflandırmasında belirleyici.
  8) domestic_int (boolean -> int)
       Mantik: Aile içi şiddet kategorisi battery/assault'la yuksek korelasyon.

Cikti: gold/feature_view Delta tablosu — silver'in ML'e hazır halinin
serialized snapshot'i (notebook'tan gozlemleme icin).

Calistirma:
  SPARK_MASTER_URL=local[*] /opt/app/run.sh /opt/app/notebooks/02_feature_engineering.py
"""
import os
import sys

from pyspark.sql import functions as F

sys.path.insert(0, "/opt/app/jobs")
from _session import build_spark, silver_path, DELTA_PATH  # noqa: E402


FEATURE_VIEW = f"{DELTA_PATH}/gold/feature_view"


def main() -> int:
    os.environ["SPARK_MASTER_URL"] = os.environ.get("SPARK_MASTER_URL", "local[*]")
    spark = build_spark("feature-engineering")
    spark.sparkContext.setLogLevel("WARN")

    silver = spark.read.format("delta").load(silver_path())

    fv = (
        silver.where(F.col("primary_type").isNotNull())
        .withColumn("arrest_int", F.col("arrest").cast("int"))
        .withColumn("domestic_int", F.col("domestic").cast("int"))
        .withColumn("is_weekend",
                    F.when(F.col("day_of_week").isin(1, 7), 1).otherwise(0))
        .withColumn("is_night",
                    F.when((F.col("hour_of_day") >= 22)
                           | (F.col("hour_of_day") <= 5), 1).otherwise(0))
        .select(
            "id",
            "primary_type",
            "hour_of_day",
            "day_of_week",
            "month",
            "event_year",
            "is_weekend",
            "is_night",
            "district",
            "ward",
            "community_area",
            "beat",
            "latitude",
            "longitude",
            "arrest_int",
            "domestic_int",
        )
    )

    n = fv.count()
    print(f"[features] feature_view rows={n}", flush=True)
    print("[features] uretilen ozellikler:", flush=True)
    for c in fv.columns:
        print(f"  - {c}", flush=True)

    (fv.write.format("delta").mode("overwrite")
        .option("overwriteSchema", "true")
        .save(FEATURE_VIEW))
    print(f"[features] yazildi -> {FEATURE_VIEW}", flush=True)

    spark.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
