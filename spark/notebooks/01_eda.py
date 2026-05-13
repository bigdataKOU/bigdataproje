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
    spark = build_spark("eda-optimized")
    spark.sparkContext.setLogLevel("ERROR")

    # Silver tablosunu oku ve persist et (count + agg işlemleri için tetikleyici olur)
    silver = spark.read.format("delta").load(silver_path()).persist()
    
    if silver.rdd.isEmpty():
        print("[eda] Silver tablosu boş, işlem iptal edildi.")
        return 0

    # 1, 3 ve 4: Çoklu Count/Distinct işlemlerini tek bir agg içine toplayarak Shuffle'ı azaltıyoruz
    # Spark her .count() için tüm tabloyu tekrar taramaz
    metrics = silver.select(
        F.count("*").alias("total"),
        F.countDistinct("primary_type").alias("u_type"),
        F.countDistinct("district").alias("u_district"),
        F.countDistinct("ward").alias("u_ward"),
        F.countDistinct("community_area").alias("u_community"),
        F.avg(F.col("arrest").cast("int")).alias("arrest_rate"),
        F.avg(F.col("domestic").cast("int")).alias("domestic_rate")
    ).collect()[0]

    total = metrics["total"]
    summary = {
        "total_rows": total,
        "unique_primary_type": metrics["u_type"],
        "unique_district": metrics["u_district"],
        "unique_ward": metrics["u_ward"],
        "unique_community_area": metrics["u_community"],
        "arrest_rate": float(metrics["arrest_rate"] or 0),
        "domestic_rate": float(metrics["domestic_rate"] or 0)
    }

    # 2) Eksik değer analizi - Her kolon için ayrı döngü yerine tek seferde hesaplama
    null_exprs = [F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in silver.columns]
    null_data = silver.select(null_exprs).collect()[0].asDict()
    summary["null_counts"] = {k: v for k, v in null_data.items() if v > 0}

    # 3 & 4) Toplu Gruplamalar (Zaman ve Tip Trendleri)
    # Tek bir collect ile list comprehensions kullanarak veriyi Python tarafına çekiyoruz
    summary["top_primary_types"] = [
        r.asDict() for r in silver.groupBy("primary_type").count().orderBy(F.desc("count")).limit(15).collect()
    ]

    # Yıllık ve Saatlik trendleri hesapla
    summary["yearly_trend"] = [
        {"year": int(r["event_year"]), "count": r["cnt"], "arrest_rate": float(r["ar"])} 
        for r in silver.groupBy("event_year").agg(F.count("*").alias("cnt"), F.avg(F.col("arrest").cast("int")).alias("ar")).orderBy("event_year").collect()
    ]

    summary["hourly_trend"] = [
        {"hour": int(r["hour_of_day"]), "count": r["cnt"]} 
        for r in silver.groupBy("hour_of_day").agg(F.count("*").alias("cnt")).orderBy("hour_of_day").collect()
    ]

    # 5) Describe yerine summary - Daha hızlı sonuç verir
    desc_pd = silver.select("latitude", "longitude", "event_year").summary("mean", "stddev", "min", "max").toPandas()
    summary["numeric_describe"] = desc_pd.set_index("summary").to_dict()

    # 6) Delta'ya yazma
    eda_data = [
        ("total_rows", float(summary["total_rows"])),
        ("unique_primary_type", float(summary["unique_primary_type"])),
        ("arrest_rate", summary["arrest_rate"]),
        ("domestic_rate", summary["domestic_rate"])
    ]
    spark.createDataFrame(eda_data, ["metric", "value"]).write.format("delta").mode("overwrite").save(EDA_TABLE)

    # 7) JSON Çıktısı
    try:
        os.makedirs(os.path.dirname(SUMMARY_PATH), exist_ok=True)
        with open(SUMMARY_PATH, "w") as f:
            json.dump(summary, f, indent=2, default=str)
    except Exception as e:
        print(f"[Hata] JSON kaydedilemedi: {e}")

    # 8) Konsol Özeti
    print(f"\n=== EDA TAMAMLANDI | Satır: {total:,} | Suç Tipi: {summary['unique_primary_type']} ===")
    
    silver.unpersist() # Belleği temizle
    spark.stop()
    return 0
