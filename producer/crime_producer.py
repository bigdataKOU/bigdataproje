"""
Chicago Crimes 2001-Present CSV'sini Kafka topic'ine basar.
Gelişmiş hata yönetimi ve veri validasyonu eklenmiş versiyon.
"""

import csv
import json
import os
import signal
import sys
import time
import math
from datetime import datetime, timezone

from confluent_kafka import Producer

# Konfigürasyon
KAFKA_BROKER = os.environ.get("KAFKA_BROKER", "kafka:9092")
TOPIC = os.environ.get("KAFKA_TOPIC_CRIMES", "crimes")
CRIMES_PATH = os.environ.get("CRIMES_PATH", "/opt/data/crimes/Crimes.csv")
MODE = os.environ.get("PRODUCER_MODE", "fixed").lower()
RATE = float(os.environ.get("PRODUCER_RATE", "1000"))
SPEEDUP = float(os.environ.get("PRODUCER_SPEEDUP", "100000"))
MAX_RECORDS = int(os.environ.get("PRODUCER_MAX_RECORDS", "500000"))
FLUSH_EVERY = int(os.environ.get("PRODUCER_FLUSH_EVERY", "5000"))

_running = True

def _stop(*_):
    global _running
    print("\n[producer] Durdurma sinyali alindi, cikiliyor...", flush=True)
    _running = False

signal.signal(signal.SIGINT, _stop)
signal.signal(signal.SIGTERM, _stop)

def delivery_report(err, msg):
    if err is not None:
        sys.stderr.write(f"[delivery] FAIL key={msg.key()} err={err}\n")

def _parse_date_to_epoch_ms(s: str) -> int:
    try:
        dt = datetime.strptime(s, "%m/%d/%Y %I:%M:%S %p")
        return int(dt.replace(tzinfo=timezone.utc).timestamp() * 1000)
    except (ValueError, TypeError):
        return 0

def _to_int(v: str):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None

def _to_float(v: str):
    try:
        f = float(v)
        # JSON'da NaN veya Inf değerleri standart dışıdır, None'a çeviriyoruz
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None

def _to_bool(v: str) -> bool:
    return str(v).strip().lower() in ("true", "1", "yes")

def main() -> int:
    if not os.path.exists(CRIMES_PATH):
        sys.stderr.write(f"HATA: Crimes.csv bulunamadi: {CRIMES_PATH}\n")
        return 2

    # Kafka Producer Yapılandırması
    producer_conf = {
        "bootstrap.servers": KAFKA_BROKER,
        "linger.ms": 50,
        "batch.size": 65536,
        "compression.type": "lz4",
        "acks": "1",
        "queue.buffering.max.messages": 200000,
    }
    
    try:
        producer = Producer(producer_conf)
    except Exception as e:
        sys.stderr.write(f"Kafka Producer olusturulamadi: {e}\n")
        return 1

    sent = 0
    prev_event_ms = None
    started_at = time.time()
    next_emit_at = started_at
    interval = 1.0 / RATE if RATE > 0 else 0.0

    print(
        f"[producer] basladi: broker={KAFKA_BROKER} topic={TOPIC}\n"
        f"[producer] mod={MODE} rate={RATE} speedup={SPEEDUP} limit={MAX_RECORDS}",
        flush=True,
    )

    try:
        # Encoding hatalarına karşı errors='replace' eklendi
        with open(CRIMES_PATH, newline="", encoding="utf-8", errors="replace") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                if not _running or sent >= MAX_RECORDS:
                    break

                event_ms = _parse_date_to_epoch_ms(row.get("Date", ""))
                if event_ms == 0:
                    continue

                # Veri Temizleme ve Dönüştürme
                try:
                    payload = {
                        "id": _to_int(row.get("ID")),
                        "case_number": row.get("Case Number") or None,
                        "primary_type": row.get("Primary Type") or None,
                        "description": row.get("Description") or None,
                        "location_description": row.get("Location Description") or None,
                        "arrest": _to_bool(row.get("Arrest", "")),
                        "domestic": _to_bool(row.get("Domestic", "")),
                        "beat": _to_int(row.get("Beat")),
                        "district": _to_int(row.get("District")),
                        "ward": _to_int(row.get("Ward")),
                        "community_area": _to_int(row.get("Community Area")),
                        "fbi_code": row.get("FBI Code") or None,
                        "year": _to_int(row.get("Year")),
                        "latitude": _to_float(row.get("Latitude")),
                        "longitude": _to_float(row.get("Longitude")),
                        "event_time_ms": event_ms,
                        "ingestedAt": int(time.time() * 1000),
                    }

                    if payload["id"] is None or payload["primary_type"] is None:
                        continue

                    # Hız Kontrolü (Pacing)
                    if MODE == "fixed" and interval > 0:
                        now = time.time()
                        if next_emit_at > now:
                            time.sleep(next_emit_at - now)
                        next_emit_at += interval
                    elif MODE == "speedup" and prev_event_ms is not None and SPEEDUP > 0:
                        wait = max(0.0, (event_ms - prev_event_ms) / 1000.0 / SPEEDUP)
                        if wait > 0:
                            time.sleep(min(wait, 1.0))
                    
                    prev_event_ms = event_ms

                    # Kafka'ya Gönderim
                    producer.produce(
                        TOPIC,
                        key=str(payload["id"]).encode("utf-8"),
                        value=json.dumps(payload).encode("utf-8"),
                        callback=delivery_report,
                    )
                    sent += 1

                except (json.JSONEncodeError, KeyError) as e:
                    sys.stderr.write(f"Satir isleme hatasi: {e}\n")
                    continue

                # Periyodik Log ve Poll
                if sent % FLUSH_EVERY == 0:
                    producer.poll(0) # Callback'leri tetikle
                    elapsed = time.time() - started_at
                    rate = sent / max(1.0, elapsed)
                    print(f"[producer] sent={sent} current_rate={rate:.0f}/s", flush=True)

    except Exception as e:
        sys.stderr.write(f"Dosya okuma hatasi: {e}\n")
    finally:
        print(f"[producer] Kapaniyor, kuyruk temizleniyor (flush)...", flush=True)
        producer.flush(30)

    total_time = time.time() - started_at
    print(f"[producer] Bitti. Toplam: {sent} kayit | Sure: {total_time:.1f}s", flush=True)
    return 0

if __name__ == "__main__":
    sys.exit(main())