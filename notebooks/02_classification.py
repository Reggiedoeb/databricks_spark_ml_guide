# Databricks notebook source
# MAGIC %md
# MAGIC # 02 — Classification
# MAGIC
# MAGIC Train and evaluate classifiers using `pyspark.ml.classification`.
# MAGIC
# MAGIC **Models covered:**
# MAGIC - Logistic Regression
# MAGIC - Decision Tree
# MAGIC - Random Forest
# MAGIC - Gradient-Boosted Trees (GBT)
# MAGIC - Multilayer Perceptron
# MAGIC
# MAGIC **Evaluators:**
# MAGIC - BinaryClassificationEvaluator (AUC-ROC, AUC-PR)
# MAGIC - MulticlassClassificationEvaluator (accuracy, F1, precision, recall)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup — Synthetic binary classification dataset

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml import Pipeline
import random

spark = SparkSession.builder.getOrCreate()

# Generate synthetic employee attrition data
random.seed(42)
rows = []
for i in range(1000):
    salary = random.uniform(30000, 150000)
    years = random.randint(0, 20)
    satisfaction = random.uniform(1, 5)
    hours_per_week = random.uniform(30, 60)
    department = random.choice(["Engineering", "Sales", "Marketing", "HR", "Finance"])
    # Attrition is more likely with low salary, low satisfaction, high hours
    prob = 0.3 - (salary / 500000) - (satisfaction / 20) + (hours_per_week / 200)
    left = 1 if random.random() < max(0.05, min(0.6, prob)) else 0
    rows.append((salary, years, satisfaction, hours_per_week, department, left))

df = spark.createDataFrame(
    rows,
    ["salary", "years_exp", "satisfaction", "hours_per_week", "department", "left"]
)

display(df.groupBy("left").count())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature preparation pipeline

# COMMAND ----------

indexer = StringIndexer(inputCol="department", outputCol="dept_index")
assembler = VectorAssembler(
    inputCols=["salary", "years_exp", "satisfaction", "hours_per_week", "dept_index"],
    outputCol="features",
)
prep_pipeline = Pipeline(stages=[indexer, assembler])
prepped_df = prep_pipeline.fit(df).transform(df)

train, test = prepped_df.randomSplit([0.8, 0.2], seed=42)
print(f"Train: {train.count()}, Test: {test.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Logistic Regression

# COMMAND ----------

from pyspark.ml.classification import LogisticRegression

lr = LogisticRegression(
    featuresCol="features",
    labelCol="left",
    maxIter=100,
    regParam=0.01,       # L2 regularization strength
    elasticNetParam=0.0, # 0 = L2, 1 = L1, between = elastic net
)
lr_model = lr.fit(train)

# Model summary
print("Coefficients:", lr_model.coefficients)
print("Intercept:   ", lr_model.intercept)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Evaluate with BinaryClassificationEvaluator

# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

lr_preds = lr_model.transform(test)

# AUC-ROC (default)
bin_eval = BinaryClassificationEvaluator(labelCol="left", rawPredictionCol="rawPrediction")
auc_roc = bin_eval.evaluate(lr_preds, {bin_eval.metricName: "areaUnderROC"})
auc_pr  = bin_eval.evaluate(lr_preds, {bin_eval.metricName: "areaUnderPR"})

# Accuracy, F1
mc_eval = MulticlassClassificationEvaluator(labelCol="left", predictionCol="prediction")
accuracy = mc_eval.evaluate(lr_preds, {mc_eval.metricName: "accuracy"})
f1       = mc_eval.evaluate(lr_preds, {mc_eval.metricName: "f1"})

print(f"Logistic Regression — AUC-ROC: {auc_roc:.4f}, AUC-PR: {auc_pr:.4f}, Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Decision Tree Classifier

# COMMAND ----------

from pyspark.ml.classification import DecisionTreeClassifier

dt = DecisionTreeClassifier(
    featuresCol="features",
    labelCol="left",
    maxDepth=5,
    minInstancesPerNode=10,
)
dt_model = dt.fit(train)
dt_preds = dt_model.transform(test)

auc = bin_eval.evaluate(dt_preds)
acc = mc_eval.evaluate(dt_preds, {mc_eval.metricName: "accuracy"})
print(f"Decision Tree — AUC-ROC: {auc:.4f}, Accuracy: {acc:.4f}")

# COMMAND ----------

# Feature importance
print("Feature importances:", dt_model.featureImportances)

# COMMAND ----------

# Print the tree for interpretability (useful for shallow trees)
print(dt_model.toDebugString[:2000])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Random Forest Classifier

# COMMAND ----------

from pyspark.ml.classification import RandomForestClassifier

rf = RandomForestClassifier(
    featuresCol="features",
    labelCol="left",
    numTrees=100,
    maxDepth=8,
    seed=42,
    # subsamplingRate=0.8,  # fraction of data per tree
    # featureSubsetStrategy="sqrt",  # features per split
)
rf_model = rf.fit(train)
rf_preds = rf_model.transform(test)

auc = bin_eval.evaluate(rf_preds)
acc = mc_eval.evaluate(rf_preds, {mc_eval.metricName: "accuracy"})
print(f"Random Forest — AUC-ROC: {auc:.4f}, Accuracy: {acc:.4f}")
print("Feature importances:", rf_model.featureImportances)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Gradient-Boosted Tree Classifier

# COMMAND ----------

from pyspark.ml.classification import GBTClassifier

gbt = GBTClassifier(
    featuresCol="features",
    labelCol="left",
    maxIter=100,          # number of boosting rounds
    maxDepth=5,
    stepSize=0.1,         # learning rate
    seed=42,
)
gbt_model = gbt.fit(train)
gbt_preds = gbt_model.transform(test)

# Note: GBT rawPrediction is not a probability — use prediction column for
# MulticlassClassificationEvaluator.  For AUC, BinaryClassificationEvaluator
# works with rawPrediction by default.
auc = bin_eval.evaluate(gbt_preds)
acc = mc_eval.evaluate(gbt_preds, {mc_eval.metricName: "accuracy"})
print(f"GBT — AUC-ROC: {auc:.4f}, Accuracy: {acc:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Multilayer Perceptron Classifier
# MAGIC
# MAGIC A feedforward neural network.  You specify the layer sizes:
# MAGIC - First layer = number of input features
# MAGIC - Last layer = number of classes
# MAGIC - Middle layers = hidden layers

# COMMAND ----------

from pyspark.ml.classification import MultilayerPerceptronClassifier

# Input size = number of features in our vector
input_size = len(train.select("features").first()[0])

mlp = MultilayerPerceptronClassifier(
    featuresCol="features",
    labelCol="left",
    layers=[input_size, 64, 32, 2],  # input → hidden → hidden → output
    maxIter=100,
    blockSize=128,  # batch size
    seed=42,
)
mlp_model = mlp.fit(train)
mlp_preds = mlp_model.transform(test)

acc = mc_eval.evaluate(mlp_preds, {mc_eval.metricName: "accuracy"})
f1  = mc_eval.evaluate(mlp_preds, {mc_eval.metricName: "f1"})
print(f"MLP — Accuracy: {acc:.4f}, F1: {f1:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Side-by-side comparison

# COMMAND ----------

from pyspark.sql import Row

results = spark.createDataFrame([
    Row(model="Logistic Regression",
        AUC_ROC=float(bin_eval.evaluate(lr_preds, {bin_eval.metricName: "areaUnderROC"})),
        Accuracy=float(mc_eval.evaluate(lr_preds, {mc_eval.metricName: "accuracy"}))),
    Row(model="Decision Tree",
        AUC_ROC=float(bin_eval.evaluate(dt_preds)),
        Accuracy=float(mc_eval.evaluate(dt_preds, {mc_eval.metricName: "accuracy"}))),
    Row(model="Random Forest",
        AUC_ROC=float(bin_eval.evaluate(rf_preds)),
        Accuracy=float(mc_eval.evaluate(rf_preds, {mc_eval.metricName: "accuracy"}))),
    Row(model="GBT",
        AUC_ROC=float(bin_eval.evaluate(gbt_preds)),
        Accuracy=float(mc_eval.evaluate(gbt_preds, {mc_eval.metricName: "accuracy"}))),
])
display(results)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluation metric reference
# MAGIC
# MAGIC | Evaluator | metricName | Use case |
# MAGIC |-----------|-----------|----------|
# MAGIC | `BinaryClassificationEvaluator` | `areaUnderROC` (default) | Binary — overall ranking quality |
# MAGIC | | `areaUnderPR` | Binary — better for imbalanced classes |
# MAGIC | `MulticlassClassificationEvaluator` | `accuracy` | Multi/binary — overall correctness |
# MAGIC | | `f1` (default) | Multi/binary — harmonic mean of precision & recall |
# MAGIC | | `weightedPrecision` | Multi — precision weighted by class support |
# MAGIC | | `weightedRecall` | Multi — recall weighted by class support |
# MAGIC | | `logLoss` | Multi — probabilistic accuracy |

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC **Next:** [03 — Regression](./03_regression)
