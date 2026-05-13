# Pipeline calistirma kolayliklari
.PHONY: help verify run-all up down build kafka mlflow producer bronze silver gold train inference charts dashboard eda logs ps clean

help:
	@echo "Hedefler:"
	@echo "  make verify      - Static check'ler (docker'siz, disk yormadan)"
	@echo "  make run-all     - Tek komut uctan-uca pipeline (Chicago Crimes)"
	@echo "  make up          - Tum servisleri baslat"
	@echo "  make down        - Tum servisleri durdur"
	@echo "  make build       - Tum image'lari build et"
	@echo "  make kafka       - Sadece kafka"
	@echo "  make mlflow      - Sadece MLflow tracking server"
	@echo "  make producer    - Producer'i tek-sefer calistir"
	@echo "  make bronze      - Bronze streaming job (foreground)"
	@echo "  make silver      - Silver streaming job (foreground)"
	@echo "  make gold        - Gold batch job"
	@echo "  make eda         - EDA notebook calistir (01_eda.py)"
	@echo "  make train       - 5 ML modelini sirayla egit (LogReg/DT/RF/GBT/NB) + MLflow"
	@echo "  make inference   - En iyi modelle silver uzerinde tahmin"
	@echo "  make charts      - docs/figures/ icin PNG grafikleri uret"
	@echo "  make report      - docs/rapor.html HTML teknik raporu uret"
	@echo "  make presentation- docs/sunum.html reveal.js slideshow uret"
	@echo "  make dashboard   - Streamlit dashboard"
	@echo "  make logs S=svc  - Servis loglari"
	@echo "  make ps          - Servis durumu"
	@echo "  make clean       - Tum delta + checkpoint + mlflow verisini sil"

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

eda:
	docker compose exec -e SPARK_MASTER_URL='local[*]' pipeline python3 /opt/app/notebooks/01_eda.py

train:
	docker compose exec -e SPARK_MASTER_URL='local[*]' -e SPARK_DRIVER_MEMORY=6g pipeline bash /opt/app/run.sh /opt/app/ml/train_models.py

inference:
	docker compose exec -e SPARK_MASTER_URL='local[*]' pipeline bash /opt/app/run.sh /opt/app/ml/inference.py

charts:
	python3 scripts/make_charts.py

report:
	@cp scripts/make_report.py dashboard/make_report.py
	@docker cp $$(docker compose ps -q pipeline):/opt/app/notebooks/eda_summary.json dashboard/eda_summary.json 2>/dev/null || true
	docker compose exec -T -e MLFLOW_TRACKING_URI=http://mlflow:5000 -e REPORT_OUT=/app/rapor.html dashboard python /app/make_report.py
	cp dashboard/rapor.html docs/rapor.html
	@echo "HTML rapor: docs/rapor.html"

presentation:
	@cp scripts/make_presentation.py dashboard/make_presentation.py
	@docker cp $$(docker compose ps -q pipeline):/opt/app/notebooks/eda_summary.json dashboard/eda_summary.json 2>/dev/null || true
	docker compose exec -T -e MLFLOW_TRACKING_URI=http://mlflow:5000 -e PRESENTATION_OUT=/app/sunum.html dashboard python /app/make_presentation.py
	cp dashboard/sunum.html docs/sunum.html
	@echo "Slideshow: docs/sunum.html (tarayıcıda aç)"

dashboard:
	docker compose up -d dashboard
	@echo "Dashboard: http://localhost:8501"

logs:
	docker compose logs -f $(S)

ps:
	docker compose ps

clean:
	docker compose down -v --remove-orphans 2>/dev/null || true
	@-rm -rf delta-store checkpoints mlruns mlflow-store 2>/dev/null || true
