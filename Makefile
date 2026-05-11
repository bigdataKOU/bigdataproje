# Pipeline calistirma kolayliklari
.PHONY: help verify run-all up down build kafka mlflow producer bronze silver gold train inference dashboard logs ps clean

help:
	@echo "Hedefler:"
	@echo "  make verify      - Static check'ler (docker'siz, disk yormadan)"
	@echo "  make run-all     - Tek komut uctan-uca pipeline"
	@echo "  make up          - Tum servisleri baslat"
	@echo "  make down        - Tum servisleri durdur"
	@echo "  make build       - Tum image'lari build et"
	@echo "  make kafka       - Sadece kafka"
	@echo "  make mlflow      - Sadece MLflow tracking server"
	@echo "  make producer    - Producer'i tek-sefer calistir"
	@echo "  make bronze      - Bronze streaming job (foreground)"
	@echo "  make silver      - Silver streaming job (foreground)"
	@echo "  make gold        - Gold batch job"
	@echo "  make train       - ALS modelini egit + MLflow log"
	@echo "  make inference   - Top-N onerileri uret"
	@echo "  make dashboard   - Streamlit dashboard"
	@echo "  make logs S=svc  - Servis loglari"
	@echo "  make ps          - Servis durumu"
	@echo "  make clean       - Tum delta + checkpoint + mlflow verisini sil (compose down -v, named volume)"

verify:
	bash scripts/verify.sh

run-all:
	bash scripts/run_all.sh

build:
	docker compose build

up:
	docker compose up -d kafka spark-master spark-worker mlflow

down:
	docker compose down

kafka:
	docker compose up -d kafka

mlflow:
	docker compose up -d mlflow

producer:
	docker compose up producer

bronze:
	docker compose up -d pipeline
	docker compose exec pipeline bash /opt/app/run.sh /opt/app/jobs/bronze_ingest.py

silver:
	docker compose exec pipeline bash /opt/app/run.sh /opt/app/jobs/silver_clean.py

gold:
	docker compose exec pipeline bash /opt/app/run.sh /opt/app/jobs/gold_features.py

train:
	docker compose exec pipeline bash /opt/app/run.sh /opt/app/ml/train_als.py

inference:
	docker compose exec pipeline bash /opt/app/run.sh /opt/app/ml/inference.py

dashboard:
	docker compose up -d dashboard
	@echo "Dashboard: http://localhost:8501"

logs:
	docker compose logs -f $(S)

ps:
	docker compose ps

clean:
	docker compose down -v --remove-orphans 2>/dev/null || true
	@# delta/checkpoint/mlruns/mlflow DB artik Docker named volume'da; down -v yeterli.
	@# Eski surumden kalan host klasorleri (root sahipli) varsa bir kez: sudo rm -rf delta-store checkpoints mlruns mlflow-store
	@-rm -rf delta-store checkpoints mlruns mlflow-store 2>/dev/null || true
