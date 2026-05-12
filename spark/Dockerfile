FROM apache/spark:3.5.8-python3

USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

ENV SPARK_HOME=/opt/spark
ENV PATH=$SPARK_HOME/bin:$PATH
ENV PYTHONUNBUFFERED=1

# Sadece pip paketleri burada. Ivy/Maven JAR'lari runtime'da
# (--packages ile spark-submit) cekilir; ivy-cache volume sayesinde
# ilk calisma sonrasinda yerel cache'lenir.
RUN pip install --no-cache-dir \
        delta-spark==3.2.0 \
        mlflow==2.16.2 \
        pandas==2.2.2 \
        pyarrow==17.0.0 \
        numpy==1.26.4

# Ivy cache dizini (pipeline servisinde volume olarak mount edilir)
RUN mkdir -p /opt/ivy-cache && chmod 777 /opt/ivy-cache

WORKDIR /opt/app
COPY jobs/ /opt/app/jobs/
COPY ml/   /opt/app/ml/
COPY run.sh /opt/app/run.sh
RUN chmod +x /opt/app/run.sh

CMD ["sleep", "infinity"]
