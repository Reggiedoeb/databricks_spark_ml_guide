# Databricks notebook source
# MAGIC %md
# MAGIC # 06 — Model Persistence & MLflow Integration
# MAGIC
# MAGIC Save, load, track, and serve Spark ML models on Databricks.
# MAGIC
# MAGIC **Topics:**
# MAGIC - Saving / loading models and pipelines natively
# MAGIC - MLflow experiment tracking (params, metrics, artifacts)
# MAGIC - MLflow auto-logging on Databricks
# MAGIC - Registering models in the Model Registry
# MAGIC - Loading registered models for inference

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup — Train a quick model

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml import Pipeline
import random

spark = SparkSession.builder.getOrCreate()

random.seed(42)
rows = []
for i in range(1000):
    salary = random.uniform(30000, 150000)
    years = random.randint(0, 20)
    satisfaction = random.uniform(1, 5)
    hours = random.uniform(30, 60)
    dept = random.choice(["Engineering", "Sales", "Marketing", "HR", "Finance"])
    prob = 0.3 - (salary / 500000) - (satisfaction / 20) + (hours / 200)
    left = 1 if random.random() < max(0.05, min(0.6, prob)) else 0
    rows.append((salary, years, satisfaction, hours, dept, left))

df = spark.createDataFrame(rows, ["salary", "years_exp", "satisfaction", "hours_per_week", "department", "left"])

pipeline = Pipeline(stages=[
    StringIndexer(inputCol="department", outputCol="dept_index"),
    VectorAssembler(
        inputCols=["salary", "years_exp", "satisfaction", "hours_per_week", "dept_index"],
        outputCol="features",
    ),
    RandomForestClassifier(featuresCol="features", labelCol="left", numTrees=100, seed=42),
])

train, test = df.randomSplit([0.8, 0.2], seed=42)
pipeline_model = pipeline.fit(train)

evaluator = BinaryClassificationEvaluator(labelCol="left", metricName="areaUnderROC")
preds = pipeline_model.transform(test)
auc = evaluator.evaluate(preds)
print(f"AUC-ROC: {auc:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Native save / load
# MAGIC
# MAGIC Every Spark ML model and pipeline model has `.save()` / `.load()` methods.
# MAGIC On Databricks, save to DBFS, Unity Catalog Volumes, or cloud storage.

# COMMAND ----------

# Save the fitted pipeline model
model_path = "/tmp/spark_ml_guide/rf_pipeline_model"
pipeline_model.write().overwrite().save(model_path)

# Load it back
from pyspark.ml import PipelineModel
loaded_model = PipelineModel.load(model_path)

# Verify it works
loaded_preds = loaded_model.transform(test)
loaded_auc = evaluator.evaluate(loaded_preds)
print(f"Loaded model AUC-ROC: {loaded_auc:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Save to Unity Catalog Volumes (recommended on Databricks)
# MAGIC
# MAGIC ```python
# MAGIC # Unity Catalog Volume path
# MAGIC volume_path = "/Volumes/my_catalog/my_schema/my_volume/models/rf_pipeline"
# MAGIC pipeline_model.write().overwrite().save(volume_path)
# MAGIC loaded = PipelineModel.load(volume_path)
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## MLflow experiment tracking
# MAGIC
# MAGIC On Databricks, MLflow is **pre-installed** and **auto-configured**.  Every
# MAGIC notebook gets an experiment automatically (based on the notebook path).

# COMMAND ----------

import mlflow

# On Databricks this is already set, but you can point to a specific experiment:
# mlflow.set_experiment("/Users/you@company.com/spark_ml_guide_experiment")

print(f"Tracking URI: {mlflow.get_tracking_uri()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Manual logging

# COMMAND ----------

with mlflow.start_run(run_name="rf_manual_log") as run:
    # Log parameters
    mlflow.log_param("num_trees", 100)
    mlflow.log_param("max_depth", 5)
    mlflow.log_param("train_size", train.count())

    # Train
    model = pipeline.fit(train)
    preds = model.transform(test)

    # Log metrics
    auc = evaluator.evaluate(preds)
    mlflow.log_metric("auc_roc", auc)

    from pyspark.ml.evaluation import MulticlassClassificationEvaluator
    mc_eval = MulticlassClassificationEvaluator(labelCol="left", predictionCol="prediction")
    mlflow.log_metric("accuracy", mc_eval.evaluate(preds, {mc_eval.metricName: "accuracy"}))
    mlflow.log_metric("f1", mc_eval.evaluate(preds, {mc_eval.metricName: "f1"}))

    # Log the Spark ML model
    mlflow.spark.log_model(model, artifact_path="spark_model")

    # Log arbitrary artifacts (e.g., feature importance as text)
    importances = str(model.stages[-1].featureImportances)
    mlflow.log_text(importances, "feature_importances.txt")

    print(f"Run ID: {run.info.run_id}")
    print(f"AUC-ROC: {auc:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Auto-logging (Databricks ML Runtime)
# MAGIC
# MAGIC On Databricks ML Runtime, `spark.ml` models are **auto-logged** by default.
# MAGIC This captures params, metrics, and the model artifact without any manual
# MAGIC `mlflow.log_*` calls.
# MAGIC
# MAGIC To enable/disable explicitly:

# COMMAND ----------

# Enable auto-logging for pyspark.ml
mlflow.pyspark.ml.autolog(
    log_models=True,            # log fitted models as artifacts
    log_input_examples=False,   # log a sample of training data
    log_model_signatures=True,  # log input/output schema
    log_post_training_metrics=True,  # log evaluator metrics after transform
)

# Now any .fit() call is automatically tracked
auto_model = pipeline.fit(train)
auto_preds = auto_model.transform(test)
auto_auc = evaluator.evaluate(auto_preds)
print(f"Auto-logged AUC: {auto_auc:.4f}")

# Check the MLflow UI in Databricks to see the auto-logged run

# COMMAND ----------

# Disable auto-logging when you want full control
mlflow.pyspark.ml.autolog(disable=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Registry
# MAGIC
# MAGIC Register your best model for staging, production, and governance.
# MAGIC
# MAGIC On Databricks with Unity Catalog, use the 3-level namespace:
# MAGIC `catalog.schema.model_name`

# COMMAND ----------

# MAGIC %md
# MAGIC ### Register from a run

# COMMAND ----------

# Register the model from the manual run above
model_name = "spark_ml_guide_attrition_rf"  # or "my_catalog.my_schema.attrition_rf" for UC

# Option 1: Register directly from the run
model_uri = f"runs:/{run.info.run_id}/spark_model"
result = mlflow.register_model(model_uri, model_name)
print(f"Registered version: {result.version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Register with Unity Catalog (recommended)
# MAGIC
# MAGIC ```python
# MAGIC # Set the registry to Unity Catalog
# MAGIC mlflow.set_registry_uri("databricks-uc")
# MAGIC
# MAGIC uc_model_name = "my_catalog.my_schema.attrition_rf"
# MAGIC result = mlflow.register_model(model_uri, uc_model_name)
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ### Set model aliases (replaces stage transitions)
# MAGIC
# MAGIC In Unity Catalog, use **aliases** instead of the legacy stage transitions
# MAGIC (Staging → Production).

# COMMAND ----------

from mlflow import MlflowClient

client = MlflowClient()

# Set an alias on the version we just registered
client.set_registered_model_alias(model_name, "champion", result.version)

# You can also set "challenger" for A/B testing
# client.set_registered_model_alias(model_name, "challenger", new_version)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load a registered model for inference

# COMMAND ----------

# Load by alias
champion_uri = f"models:/{model_name}@champion"
champion_model = mlflow.spark.load_model(champion_uri)

# Run inference
champion_preds = champion_model.transform(test)
display(champion_preds.select("salary", "years_exp", "satisfaction", "left", "prediction", "probability").limit(10))

# COMMAND ----------

# Load by version number
# versioned_uri = f"models:/{model_name}/1"
# versioned_model = mlflow.spark.load_model(versioned_uri)

# COMMAND ----------

# MAGIC %md
# MAGIC ## MLflow cheat sheet for Databricks
# MAGIC
# MAGIC | Task | Code |
# MAGIC |------|------|
# MAGIC | Start a run | `with mlflow.start_run() as run:` |
# MAGIC | Log parameter | `mlflow.log_param("key", value)` |
# MAGIC | Log metric | `mlflow.log_metric("key", value)` |
# MAGIC | Log Spark model | `mlflow.spark.log_model(model, "artifact_path")` |
# MAGIC | Log artifact file | `mlflow.log_artifact("/path/to/file")` |
# MAGIC | Log text | `mlflow.log_text("content", "filename.txt")` |
# MAGIC | Enable auto-log | `mlflow.pyspark.ml.autolog()` |
# MAGIC | Register model | `mlflow.register_model(uri, name)` |
# MAGIC | Set alias | `client.set_registered_model_alias(name, alias, version)` |
# MAGIC | Load by alias | `mlflow.spark.load_model("models:/name@alias")` |
# MAGIC | Load by version | `mlflow.spark.load_model("models:/name/1")` |
# MAGIC | Load from run | `mlflow.spark.load_model("runs:/run_id/artifact_path")` |

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC **Next:** [07 — Databricks Tips & Best Practices](./07_databricks_tips)
