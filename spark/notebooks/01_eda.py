"""
Adim 4: Kesifsel Veri Analizi (EDA) — Chicago Crimes.

Silver Delta tablosundan veriyi okur ve PDF'in istedigi EDA ciktilarini uretir:
  - Temel istatistikler (satir sayisi, benzersiz deger sayilari, olay dagilimi)
  - Eksik deger analizi
  - Zaman serisi: gunluk/saatlik trendler
  - Kategorik ve sayisal degiskenlerin dagilim analizi

Ciktilar:
  - /opt/app/notebooks/eda_summary.json   (sayisal ozet)
  - /opt/delta/gold/eda_overview Delta tablosu (zaman trendi, dashboard'a aktarmak icin)
  - stdout'a metinsel ozet

Calistirma (host):
  make eda

Calistirma (container):
  SPARK_MASTER_URL=local[*] /opt/app/run.sh /opt/app/notebooks/01_eda.py
"""
import json
import os
import sys

from pyspark.sql import functions as F

sys.path.insert(0, "/opt/app/jobs")
from _session import build_spark, silver_path, DELTA_PATH  # noqa: E402


SUMMARY_PATH = "/opt/app/notebooks/eda_summary.json"
EDA_TABLE = f"{DELTA_PATH}/gold/eda_overview"


def main() -> int:
    os.environ["SPARK_MASTER_URL"] = os.environ.get("SPARK_MASTER_URL", "local[*]")
    spark = build_spark("eda")
    spark.sparkContext.setLogLevel("WARN")

    silver = spark.read.format("delta").load(silver_path()).cache()
    summary = {}

    # 1) Temel istatistikler
    total = silver.count()
    summary["total_rows"] = total
    summary["unique_primary_type"] = silver.select("primary_type").distinct().count()
    summary["unique_district"] = silver.select("district").distinct().count()
    summary["unique_ward"] = silver.select("ward").distinct().count()
    summary["unique_community_area"] = silver.select("community_area").distinct().count()
    summary["unique_year"] = silver.select("event_year").distinct().count()
    summary["arrest_rate"] = float(
        silver.where(F.col("arrest")).count() / total if total else 0.0
    )
    summary["domestic_rate"] = float(
        silver.where(F.col("domestic")).count() / total if total else 0.0
    )

    # 2) Eksik deger analizi (silver'da olmamasi gerek ama dogrula)
    null_counts = {}
    for col in silver.columns:
        n = silver.where(F.col(col).isNull()).count()
        if n > 0:
            null_counts[col] = n
    summary["null_counts"] = null_counts

    # 3) Olay dagilimi: top primary_type
    top_types = (
        silver.groupBy("primary_type")
        .count()
        .orderBy(F.col("count").desc())
        .limit(15)
        .collect()
    )
    summary["top_primary_types"] = [
        {"primary_type": r["primary_type"], "count": r["count"]} for r in top_types
    ]

    # 4) Zaman trendleri: yıllık + saatlik + haftalik
    yearly = (
        silver.groupBy("event_year")
        .agg(F.count("*").alias("crime_count"),
             F.avg(F.col("arrest").cast("int")).alias("arrest_rate"))
        .orderBy("event_year")
    )
    yearly_rows = yearly.collect()
    summary["yearly_trend"] = [
        {"year": int(r["event_year"]),
         "crime_count": int(r["crime_count"]),
         "arrest_rate": float(r["arrest_rate"])}
        for r in yearly_rows
    ]

    hourly = (
        silver.groupBy("hour_of_day")
        .agg(F.count("*").alias("crime_count"))
        .orderBy("hour_of_day")
        .collect()
    )
    summary["hourly_trend"] = [
        {"hour": int(r["hour_of_day"]), "crime_count": int(r["crime_count"])}
        for r in hourly
    ]

    weekly = (
        silver.groupBy("day_of_week")
        .agg(F.count("*").alias("crime_count"))
        .orderBy("day_of_week")
        .collect()
    )
    summary["weekly_trend"] = [
        {"day_of_week": int(r["day_of_week"]),
         "crime_count": int(r["crime_count"])}
        for r in weekly
    ]

    # 5) Sayisal degisken dagilimi (latitude/longitude/year)
    desc = silver.select("latitude", "longitude", "year").describe().collect()
    summary["numeric_describe"] = {
        r["summary"]: {
            "latitude": r["latitude"],
            "longitude": r["longitude"],
            "year": r["year"],
        }
        for r in desc
    }

    # 6) Bir sonraki adim icin dashboard'a aktar — gold/eda_overview Delta
    eda_df = spark.createDataFrame(
        [
            {"metric": "total_rows", "value": float(summary["total_rows"])},
            {"metric": "unique_primary_type", "value": float(summary["unique_primary_type"])},
            {"metric": "unique_district", "value": float(summary["unique_district"])},
            {"metric": "arrest_rate", "value": summary["arrest_rate"]},
            {"metric": "domestic_rate", "value": summary["domestic_rate"]},
        ]
    )
    (eda_df.write.format("delta").mode("overwrite")
        .option("overwriteSchema", "true")
        .save(EDA_TABLE))
    print(f"[eda] gold table -> {EDA_TABLE}", flush=True)

    # 7) JSON dump
    os.makedirs(os.path.dirname(SUMMARY_PATH), exist_ok=True)
    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[eda] summary json -> {SUMMARY_PATH}", flush=True)

    # 8) Stdout ozet
    print("\n=== EDA Özeti ===", flush=True)
    print(f"Toplam satir         : {summary['total_rows']:,}", flush=True)
    print(f"Benzersiz suç tipi   : {summary['unique_primary_type']}", flush=True)
    print(f"Benzersiz ilçe       : {summary['unique_district']}", flush=True)
    print(f"Tutuklama orani      : {summary['arrest_rate']:.3%}", flush=True)
    print(f"Yil araligi          : "
          f"{summary['yearly_trend'][0]['year']} - "
          f"{summary['yearly_trend'][-1]['year']}", flush=True)
    print("\nTop 5 suç tipi:", flush=True)
    for t in summary["top_primary_types"][:5]:
        print(f"  {t['primary_type']:30s} {t['count']:>10,}", flush=True)

    spark.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
