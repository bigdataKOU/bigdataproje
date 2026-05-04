FROM apache/spark:3.5.8-python3

USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

ENV SPARK_HOME=/opt/spark
ENV PATH=$SPARK_HOME/bin:$PATH

ENV DELTA_PKG=io.delta:delta-spark_2.12:3.2.0
ENV KAFKA_PKG=org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1
ENV MLFLOW_PKG=org.mlflow:mlflow-spark_2.12:2.16.0

# Pre-resolve packages so spark-submit calls are offline-fast
RUN $SPARK_HOME/bin/spark-shell --packages "${DELTA_PKG},${KAFKA_PKG},${MLFLOW_PKG}" \
        --conf "spark.jars.ivy=/opt/spark/.ivy2" \
        -e ":quit" 2>&1 | tail -5 || true

RUN pip install --no-cache-dir \
        delta-spark==3.2.0 \
        mlflow==2.16.2 \
        pandas==2.2.2 \
        pyarrow==17.0.0 \
        numpy==1.26.4

WORKDIR /opt/app
COPY jobs/ /opt/app/jobs/
COPY ml/   /opt/app/ml/
COPY run.sh /opt/app/run.sh
RUN chmod +x /opt/app/run.sh

CMD ["sleep", "infinity"]
