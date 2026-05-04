"""
MovieLens ratings.csv'i Kafka topic'ine kronolojik sırada basar.

ratings.csv formati: userId,movieId,rating,timestamp
Producer her satırı JSON olarak kafka'ya yazar. Speedup faktörü ile
gerçek zamandaki saniye farkları sıkıştırılır (1000 = 1000x hızlı).
"""

import csv
import json
import os
import signal
import sys
import time
from confluent_kafka import Producer

KAFKA_BROKER = os.environ.get("KAFKA_BROKER", "kafka:9092")
TOPIC = os.environ.get("KAFKA_TOPIC_RATINGS", "ratings")
RATINGS_PATH = os.environ.get("RATINGS_PATH", "/opt/data/ml-25m/ratings.csv")
SPEEDUP = float(os.environ.get("PRODUCER_SPEEDUP", "100000"))
MAX_RECORDS = int(os.environ.get("PRODUCER_MAX_RECORDS", "500000"))
FLUSH_EVERY = int(os.environ.get("PRODUCER_FLUSH_EVERY", "5000"))

_running = True


def _stop(*_):
    global _running
    _running = False


signal.signal(signal.SIGINT, _stop)
signal.signal(signal.SIGTERM, _stop)


def delivery_report(err, msg):
    if err is not None:
        sys.stderr.write(f"[delivery] FAIL key={msg.key()} err={err}\n")


def main() -> int:
    if not os.path.exists(RATINGS_PATH):
        sys.stderr.write(f"ratings.csv bulunamadi: {RATINGS_PATH}\n")
        return 2

    producer = Producer({
        "bootstrap.servers": KAFKA_BROKER,
        "linger.ms": 50,
        "batch.size": 65536,
        "compression.type": "lz4",
        "acks": "1",
        "queue.buffering.max.messages": 200000,
    })

    sent = 0
    prev_ts: int | None = None
    started_at = time.time()

    print(f"[producer] broker={KAFKA_BROKER} topic={TOPIC} "
          f"path={RATINGS_PATH} speedup={SPEEDUP} max={MAX_RECORDS}", flush=True)

    with open(RATINGS_PATH, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if not _running or sent >= MAX_RECORDS:
                break

            try:
                ts = int(row["timestamp"])
                user_id = int(row["userId"])
                movie_id = int(row["movieId"])
                rating = float(row["rating"])
            except (KeyError, ValueError):
                continue

            payload = {
                "userId": user_id,
                "movieId": movie_id,
                "rating": rating,
                "timestamp": ts,
                "ingestedAt": int(time.time() * 1000),
            }

            if prev_ts is not None and SPEEDUP > 0:
                wait = max(0.0, (ts - prev_ts) / SPEEDUP)
                if wait > 0:
                    time.sleep(min(wait, 1.0))
            prev_ts = ts

            producer.produce(
                TOPIC,
                key=str(user_id).encode(),
                value=json.dumps(payload).encode(),
                callback=delivery_report,
            )
            sent += 1

            if sent % FLUSH_EVERY == 0:
                producer.poll(0)
                rate = sent / max(1.0, time.time() - started_at)
                print(f"[producer] sent={sent} rate={rate:.0f}/s", flush=True)

    producer.flush(30)
    print(f"[producer] done sent={sent} elapsed={time.time() - started_at:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
