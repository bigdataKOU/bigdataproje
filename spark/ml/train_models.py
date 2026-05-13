"""
BLM442 zorunlu: en az 5 farkli ML modelini sirayla egit, hepsini MLflow'a logla.

5 model (siniflandirma — Primary Type tahmini):
  1) Logistic Regression  (multinomial)
  2) Decision Tree Classifier
  3) Random Forest Classifier
  4) GBT Classifier (OneVsRest ile multi-class)
  5) Naive Bayes

Her model icin loglar:
  - Parametreler
  - Metrikler: accuracy, weighted_f1, weighted_precision, weighted_recall,
               multi_class_auc_ovr_macro, train_seconds
  - Feature Importance (modelden cikariliyor; LR icin coefficients magnitude)
  - Confusion Matrix CSV artifact
  - Per-class precision/recall artifact
  - Modelin kendisi (mlflow.spark.log_model)

Calistirma:
  /opt/app/run.sh /opt/app/ml/train_models.py
"""
import os
import sys
import time

import mlflow
import mlflow.spark
import numpy as np
import pandas as pd
from pyspark.ml import Pipeline
from pyspark.ml.classification import (
    DecisionTreeClassifier,
    GBTClassifier,
    LogisticRegression,
    NaiveBayes,
    OneVsRest,
    RandomForestClassifier,
)
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
)
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.sql import DataFrame, functions as F

sys.path.insert(0, "/opt/app/jobs")
from _session import build_spark, silver_path  # noqa: E402


MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
EXPERIMENT_NAME = os.environ.get("MLFLOW_EXPERIMENT", "chicago-crimes-classifier")

TOP_N_TYPES = int(os.environ.get("TOP_N_TYPES", "5"))
TRAIN_RATIO = float(os.environ.get("TRAIN_RATIO", "0.8"))
SAMPLE_FRACTION = float(os.environ.get("SAMPLE_FRACTION", "0.2"))


FEATURE_COLUMNS = [
    "district",
    "ward",
    "community_area",
    "beat",
    "hour_of_day",
    "day_of_week",
    "month",
    "year",
    "latitude",
    "longitude",
    "arrest_int",
    "domestic_int",
]


def prep_data(spark):
    df = (
        spark.read.format("delta").load(silver_path())
        .where(F.col("primary_type").isNotNull())
        .withColumn("arrest_int", F.col("arrest").cast("int"))
        .withColumn("domestic_int", F.col("domestic").cast("int"))
        .na.fill(0, FEATURE_COLUMNS)
    )
    if SAMPLE_FRACTION < 1.0:
        df = df.sample(False, SAMPLE_FRACTION, seed=42)

    top_values_df = (
        df.groupBy("primary_type")
          .agg(F.count("*").alias("c"))
          .orderBy(F.col("c").desc())
          .limit(TOP_N_TYPES)
    )
    top_values = [r["primary_type"] for r in top_values_df.collect()]
    print(f"[data] top-{TOP_N_TYPES} types: {top_values}", flush=True)

    df = df.withColumn(
        "label_raw",
        F.when(F.col("primary_type").isin(top_values), F.col("primary_type"))
         .otherwise(F.lit("OTHER")),
    ).select(*FEATURE_COLUMNS, "label_raw")

    indexer = StringIndexer(
        inputCol="label_raw",
        outputCol="label",
        handleInvalid="keep",
    )
    indexer_model = indexer.fit(df)
    labels = list(indexer_model.labels)
    print(f"[data] labels={labels}", flush=True)

    df = indexer_model.transform(df)
    assembler = VectorAssembler(
        inputCols=FEATURE_COLUMNS,
        outputCol="features",
        handleInvalid="skip",
    )
    df = assembler.transform(df)
    df = df.select("features", "label", "label_raw").cache()
    n = df.count()
    print(f"[data] cached rows={n}", flush=True)

    train, test = df.randomSplit([TRAIN_RATIO, 1.0 - TRAIN_RATIO], seed=42)
    return train, test, labels, n


def confusion_matrix(predictions: DataFrame, labels: list[str]):
    """predictions df'inden tablo + Pandas DataFrame (gercek x tahmin)."""
    cm_df = (
        predictions.groupBy("label", "prediction")
        .agg(F.count("*").alias("n"))
        .toPandas()
    )
    pivot = (
        cm_df.pivot(index="label", columns="prediction", values="n")
        .fillna(0).astype(int)
    )
    # label/prediction integer -> label name
    idx_map = {i: labels[i] if i < len(labels) else f"idx_{i}"
               for i in range(len(labels))}
    pivot = pivot.rename(index=idx_map, columns=idx_map)
    return pivot


def per_class_metrics(predictions: DataFrame, labels: list[str]) -> pd.DataFrame:
    """Her sinif icin TP/FP/FN -> precision, recall, f1."""
    rows = []
    for i, name in enumerate(labels):
        tp = predictions.where(
            (F.col("label") == i) & (F.col("prediction") == i)
        ).count()
        fp = predictions.where(
            (F.col("label") != i) & (F.col("prediction") == i)
        ).count()
        fn = predictions.where(
            (F.col("label") == i) & (F.col("prediction") != i)
        ).count()
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
        rows.append({
            "label": name,
            "tp": tp, "fp": fp, "fn": fn,
            "precision": prec, "recall": rec, "f1": f1,
        })
    return pd.DataFrame(rows)


def multi_class_auc(predictions: DataFrame, labels: list[str]) -> float:
    """OvR macro AUC. predictions'da 'probability' kolonu olmali (Vector)."""
    if "probability" not in predictions.columns:
        return float("nan")
    try:
        from pyspark.ml.functions import vector_to_array
        arr_col = vector_to_array("probability")
    except Exception:
        return float("nan")

    aucs = []
    for i in range(len(labels)):
        try:
            bin_df = predictions.select(
                F.expr(f"CASE WHEN label = {i} THEN 1.0 ELSE 0.0 END")
                    .alias("bin_label"),
                arr_col.getItem(i).alias("score"),
            )
            ev = BinaryClassificationEvaluator(
                labelCol="bin_label",
                rawPredictionCol="score",
                metricName="areaUnderROC",
            )
            auc = float(ev.evaluate(bin_df))
            if not np.isnan(auc):
                aucs.append(auc)
        except Exception as exc:
            print(f"[auc] sinif {i} hata: {exc}", flush=True)
    return float(np.mean(aucs)) if aucs else float("nan")


def feature_importances_from(model, base_pipeline_stages) -> dict[str, float]:
    """Train edilmis modelden feature_importance dict cikar."""
    feat_names = FEATURE_COLUMNS
    importances = {}
    # PipelineModel ise son stage'i al
    final_stage = model
    if hasattr(model, "stages") and model.stages:
        # Train sirasinda assembler+indexer disinda model genelde son stage
        for s in reversed(model.stages):
            if hasattr(s, "featureImportances") or hasattr(s, "coefficients") \
                    or hasattr(s, "coefficientMatrix") or hasattr(s, "models"):
                final_stage = s
                break

    if hasattr(final_stage, "featureImportances"):
        fi = final_stage.featureImportances.toArray()
        importances = {feat_names[i]: float(fi[i])
                       for i in range(min(len(feat_names), len(fi)))}
    elif hasattr(final_stage, "coefficientMatrix"):
        cm = final_stage.coefficientMatrix.toArray()  # n_classes x n_features
        importances = {
            feat_names[i]: float(np.abs(cm[:, i]).mean())
            for i in range(min(len(feat_names), cm.shape[1]))
        }
    elif hasattr(final_stage, "coefficients"):
        coef = final_stage.coefficients.toArray()
        importances = {feat_names[i]: float(abs(coef[i]))
                       for i in range(min(len(feat_names), len(coef)))}
    elif hasattr(final_stage, "models") and final_stage.models:
        # OneVsRest: alt modellerin (binary) coef/featImportance ortalamasi
        accum = np.zeros(len(feat_names))
        cnt = 0
        for sub in final_stage.models:
            if hasattr(sub, "featureImportances"):
                fi = sub.featureImportances.toArray()
                accum[: len(fi)] += fi
                cnt += 1
            elif hasattr(sub, "coefficients"):
                c = np.abs(sub.coefficients.toArray())
                accum[: len(c)] += c
                cnt += 1
        if cnt:
            accum /= cnt
            importances = {feat_names[i]: float(accum[i])
                           for i in range(len(feat_names))}
    return importances


def log_per_model(name, model, predictions, labels, train_seconds, params=None):
    """Predictions + model -> metric/artifact logla."""
    mlflow.log_param("model_type", name)
    mlflow.log_param("n_features", len(FEATURE_COLUMNS))
    mlflow.log_param("n_classes", len(labels))
    for k, v in (params or {}).items():
        mlflow.log_param(k, v)

    mlflow.log_metric("train_seconds", train_seconds)

    for metric_name, key in [
        ("accuracy", "accuracy"),
        ("f1", "weighted_f1"),
        ("weightedPrecision", "weighted_precision"),
        ("weightedRecall", "weighted_recall"),
    ]:
        ev = MulticlassClassificationEvaluator(
            labelCol="label", predictionCol="prediction", metricName=metric_name,
        )
        value = float(ev.evaluate(predictions))
        mlflow.log_metric(key, value)
        print(f"[{name}] {key}={value:.4f}", flush=True)

    try:
        auc = multi_class_auc(predictions, labels)
        if not np.isnan(auc):
            mlflow.log_metric("auc_ovr_macro", auc)
            print(f"[{name}] auc_ovr_macro={auc:.4f}", flush=True)
    except Exception as exc:
        print(f"[{name}] auc hesabi atlandi: {exc}", flush=True)

    cm = confusion_matrix(predictions, labels)
    cm_csv = f"/tmp/cm_{name}.csv"
    cm.to_csv(cm_csv)
    mlflow.log_artifact(cm_csv)

    per_class = per_class_metrics(predictions, labels)
    per_class_csv = f"/tmp/per_class_{name}.csv"
    per_class.to_csv(per_class_csv, index=False)
    mlflow.log_artifact(per_class_csv)

    fi = feature_importances_from(model, None)
    if fi:
        fi_df = pd.DataFrame(
            [{"feature": k, "importance": v} for k, v in fi.items()]
        ).sort_values("importance", ascending=False)
        fi_csv = f"/tmp/feature_importance_{name}.csv"
        fi_df.to_csv(fi_csv, index=False)
        mlflow.log_artifact(fi_csv)
        for k, v in fi.items():
            mlflow.log_metric(f"fi_{k}", v)
    else:
        print(f"[{name}] feature_importance hesaplanamadi", flush=True)

    mlflow.spark.log_model(
        model,
        artifact_path=f"{name}_model",
        registered_model_name=f"chicago-crime-{name}",
    )


def main() -> int:
    spark = build_spark("crime-models")
    spark.sparkContext.setLogLevel("WARN")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    train, test, labels, n_rows = prep_data(spark)
    base_params = {
        "top_n_types": TOP_N_TYPES,
        "train_ratio": TRAIN_RATIO,
        "sample_fraction": SAMPLE_FRACTION,
        "n_rows": n_rows,
    }

    # Hiperparametreler env'den override edilebilir
    LR_MAX_ITER = int(os.environ.get("LR_MAX_ITER", "50"))
    DT_MAX_DEPTH = int(os.environ.get("DT_MAX_DEPTH", "15"))
    RF_NUM_TREES = int(os.environ.get("RF_NUM_TREES", "100"))
    RF_MAX_DEPTH = int(os.environ.get("RF_MAX_DEPTH", "15"))
    GBT_MAX_ITER = int(os.environ.get("GBT_MAX_ITER", "30"))
    GBT_MAX_DEPTH = int(os.environ.get("GBT_MAX_DEPTH", "5"))

    models_to_train = [
        (
            "logistic_regression",
            LogisticRegression(
                labelCol="label", featuresCol="features",
                maxIter=LR_MAX_ITER, regParam=0.0, elasticNetParam=0.0,
                family="multinomial",
            ),
            {"maxIter": LR_MAX_ITER, "regParam": 0.0, "family": "multinomial"},
        ),
        (
            "decision_tree",
            DecisionTreeClassifier(
                labelCol="label", featuresCol="features",
                maxDepth=DT_MAX_DEPTH, maxBins=64, seed=42,
            ),
            {"maxDepth": DT_MAX_DEPTH, "maxBins": 64},
        ),
        (
            "random_forest",
            RandomForestClassifier(
                labelCol="label", featuresCol="features",
                numTrees=RF_NUM_TREES, maxDepth=RF_MAX_DEPTH, seed=42,
                featureSubsetStrategy="auto",
            ),
            {"numTrees": RF_NUM_TREES, "maxDepth": RF_MAX_DEPTH},
        ),
        (
            "gbt_ovr",
            OneVsRest(
                labelCol="label", featuresCol="features",
                classifier=GBTClassifier(
                    labelCol="label", featuresCol="features",
                    maxIter=GBT_MAX_ITER, maxDepth=GBT_MAX_DEPTH, seed=42,
                ),
            ),
            {"gbt_maxIter": GBT_MAX_ITER, "gbt_maxDepth": GBT_MAX_DEPTH,
             "wrapper": "OneVsRest"},
        ),
        (
            "naive_bayes",
            # gaussian: lat/lon negatif olabildigi icin multinomial calismaz
            NaiveBayes(
                labelCol="label", featuresCol="features",
                modelType="gaussian",
            ),
            {"modelType": "gaussian"},
        ),
    ]

    for name, est, model_params in models_to_train:
        run_name = f"model-{name}"
        print(f"\n=== {name} egitiliyor ===", flush=True)
        with mlflow.start_run(run_name=run_name) as run:
            try:
                t0 = time.time()
                model = est.fit(train)
                train_seconds = time.time() - t0
                preds = model.transform(test)
                log_per_model(name, model, preds, labels, train_seconds,
                              params={**base_params, **model_params})
                print(f"[{name}] mlflow run_id={run.info.run_id} "
                      f"({train_seconds:.1f}s)", flush=True)
            except Exception as exc:
                mlflow.log_param("error", str(exc)[:500])
                print(f"[{name}] HATA: {exc}", flush=True)

    spark.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
