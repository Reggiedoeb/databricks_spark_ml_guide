# Databricks notebook source
# MAGIC %md
# MAGIC # 05 — Hyperparameter Tuning & Cross-Validation
# MAGIC
# MAGIC Systematically search for the best model configuration.
# MAGIC
# MAGIC **Topics:**
# MAGIC - ParamGridBuilder — define the search space
# MAGIC - CrossValidator — k-fold cross-validation
# MAGIC - TrainValidationSplit — single train/validation split (faster)
# MAGIC - Tuning a full Pipeline
# MAGIC - Parallelism on Databricks
# MAGIC - Extracting results

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup — Reuse the classification dataset from Notebook 02

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
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

# Full pipeline: index → assemble → classify
indexer = StringIndexer(inputCol="department", outputCol="dept_index")
assembler = VectorAssembler(
    inputCols=["salary", "years_exp", "satisfaction", "hours_per_week", "dept_index"],
    outputCol="features",
)
rf = RandomForestClassifier(featuresCol="features", labelCol="left", seed=42)

pipeline = Pipeline(stages=[indexer, assembler, rf])

train, test = df.randomSplit([0.8, 0.2], seed=42)
evaluator = BinaryClassificationEvaluator(labelCol="left", metricName="areaUnderROC")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ParamGridBuilder — Define the search space
# MAGIC
# MAGIC Build a grid of parameter combinations to evaluate.  Reference parameters
# MAGIC using the estimator's `.param` attributes.

# COMMAND ----------

from pyspark.ml.tuning import ParamGridBuilder

paramGrid = (
    ParamGridBuilder()
    .addGrid(rf.numTrees, [50, 100, 200])
    .addGrid(rf.maxDepth, [4, 6, 8])
    .addGrid(rf.minInstancesPerNode, [1, 5])
    .build()
)

print(f"Total parameter combinations: {len(paramGrid)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## CrossValidator — K-fold cross-validation
# MAGIC
# MAGIC Splits training data into `numFolds` folds, trains on k-1 folds, evaluates
# MAGIC on the held-out fold, and averages the metric.  Returns the model trained on
# MAGIC **all** training data using the best parameter set.
# MAGIC
# MAGIC **Databricks tip:** Set `parallelism` > 1 to train multiple models in parallel
# MAGIC across your cluster.  A good default is 2–4x the number of worker nodes.

# COMMAND ----------

from pyspark.ml.tuning import CrossValidator

cv = CrossValidator(
    estimator=pipeline,        # the full pipeline
    estimatorParamMaps=paramGrid,
    evaluator=evaluator,
    numFolds=3,                # 3-fold CV
    parallelism=4,             # train 4 models in parallel (adjust to cluster size)
    seed=42,
)

cv_model = cv.fit(train)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Extract cross-validation results

# COMMAND ----------

# Average metric for each parameter combination
avg_metrics = cv_model.avgMetrics
print(f"Best AUC-ROC (CV avg): {max(avg_metrics):.4f}")

# Which param combo was best?
best_idx = avg_metrics.index(max(avg_metrics))
best_params = paramGrid[best_idx]
for param, value in best_params.items():
    print(f"  {param.name}: {value}")

# COMMAND ----------

# The best model is already trained on the full training set
best_pipeline_model = cv_model.bestModel

# Access the RandomForest stage from the pipeline
best_rf = best_pipeline_model.stages[-1]
print(f"Best numTrees: {best_rf.getNumTrees}")
print(f"Best maxDepth: {best_rf.getOrDefault('maxDepth')}")
print(f"Feature importances: {best_rf.featureImportances}")

# COMMAND ----------

# Evaluate on held-out test set
test_preds = cv_model.transform(test)
test_auc = evaluator.evaluate(test_preds)
print(f"Test AUC-ROC: {test_auc:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### View all CV results in a table

# COMMAND ----------

from pyspark.sql import Row

results_rows = []
for i, params in enumerate(paramGrid):
    row = {"avg_auc": avg_metrics[i]}
    for param, value in params.items():
        row[param.name] = value
    results_rows.append(Row(**row))

results_df = spark.createDataFrame(results_rows)
display(results_df.orderBy("avg_auc", ascending=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## TrainValidationSplit — Faster alternative
# MAGIC
# MAGIC Uses a single train/validation split instead of k-fold.  Much faster when
# MAGIC the parameter grid is large or data is plentiful.  `trainRatio=0.8` means
# MAGIC 80% train, 20% validation.

# COMMAND ----------

from pyspark.ml.tuning import TrainValidationSplit

tvs = TrainValidationSplit(
    estimator=pipeline,
    estimatorParamMaps=paramGrid,
    evaluator=evaluator,
    trainRatio=0.8,
    parallelism=4,
    seed=42,
)

tvs_model = tvs.fit(train)

# Best metric
best_metric = max(tvs_model.validationMetrics)
print(f"Best AUC-ROC (validation): {best_metric:.4f}")

# Evaluate on test
tvs_test_preds = tvs_model.transform(test)
tvs_test_auc = evaluator.evaluate(tvs_test_preds)
print(f"Test AUC-ROC: {tvs_test_auc:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Tuning tips for Databricks
# MAGIC
# MAGIC | Tip | Detail |
# MAGIC |-----|--------|
# MAGIC | **parallelism** | Set to 2–4× worker nodes. More parallelism = more memory use. |
# MAGIC | **Cache training data** | `train.cache()` before `.fit()` — avoids recomputing features each fold. |
# MAGIC | **Start coarse, refine** | Use TVS with a wide grid first, then narrow with CV. |
# MAGIC | **Smaller grids** | 18 combos × 3 folds = 54 models. Keep it manageable. |
# MAGIC | **Use ML Runtime** | Databricks ML Runtime includes optimized libraries and MLflow auto-logging. |
# MAGIC | **Monitor in Spark UI** | Check the Jobs tab for stragglers and resource bottlenecks. |
# MAGIC | **Photon** | Photon accelerates the data-loading portion of the pipeline but not the ML training itself. |

# COMMAND ----------

# MAGIC %md
# MAGIC ## Full example — Pipeline with tuning

# COMMAND ----------

from pyspark.ml.classification import GBTClassifier
from pyspark.ml.feature import StandardScaler

# More sophisticated pipeline
full_pipeline = Pipeline(stages=[
    StringIndexer(inputCol="department", outputCol="dept_index"),
    VectorAssembler(
        inputCols=["salary", "years_exp", "satisfaction", "hours_per_week", "dept_index"],
        outputCol="raw_features",
    ),
    StandardScaler(inputCol="raw_features", outputCol="features", withStd=True, withMean=False),
    GBTClassifier(featuresCol="features", labelCol="left", seed=42),
])

# Reference the GBT stage for the param grid
gbt = full_pipeline.getStages()[-1]

param_grid = (
    ParamGridBuilder()
    .addGrid(gbt.maxIter, [50, 100])
    .addGrid(gbt.maxDepth, [3, 5])
    .addGrid(gbt.stepSize, [0.05, 0.1])
    .build()
)

cv_full = CrossValidator(
    estimator=full_pipeline,
    estimatorParamMaps=param_grid,
    evaluator=evaluator,
    numFolds=3,
    parallelism=4,
    seed=42,
)

# Cache before fitting
train.cache()
cv_full_model = cv_full.fit(train)

best_auc = max(cv_full_model.avgMetrics)
test_auc = evaluator.evaluate(cv_full_model.transform(test))
print(f"Best CV AUC: {best_auc:.4f}")
print(f"Test AUC:    {test_auc:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC **Next:** [06 — Model Persistence & MLflow](./06_mlflow_integration)
