# Databricks notebook source
# MAGIC %md
# MAGIC # 03 — Regression
# MAGIC
# MAGIC Predict continuous values with `pyspark.ml.regression`.
# MAGIC
# MAGIC **Models covered:**
# MAGIC - Linear Regression
# MAGIC - Decision Tree Regressor
# MAGIC - Random Forest Regressor
# MAGIC - Gradient-Boosted Tree Regressor
# MAGIC - Generalized Linear Regression (GLM)
# MAGIC
# MAGIC **Evaluator:** `RegressionEvaluator` (RMSE, MSE, MAE, R²)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup — Synthetic housing dataset

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
import random, math

spark = SparkSession.builder.getOrCreate()

random.seed(42)
rows = []
for _ in range(2000):
    sqft     = random.uniform(600, 4000)
    bedrooms = random.randint(1, 6)
    age      = random.uniform(0, 50)
    lot_size = random.uniform(2000, 20000)
    garage   = random.randint(0, 3)
    # Price with some noise
    price = (
        50000
        + 120 * sqft
        + 15000 * bedrooms
        - 800 * age
        + 5 * lot_size
        + 20000 * garage
        + random.gauss(0, 25000)
    )
    rows.append((sqft, bedrooms, age, lot_size, garage, round(price, 2)))

df = spark.createDataFrame(rows, ["sqft", "bedrooms", "age", "lot_size", "garage", "price"])

assembler = VectorAssembler(
    inputCols=["sqft", "bedrooms", "age", "lot_size", "garage"],
    outputCol="features",
)
data = assembler.transform(df)
train, test = data.randomSplit([0.8, 0.2], seed=42)
print(f"Train: {train.count()}, Test: {test.count()}")
display(df.describe())

# COMMAND ----------

# MAGIC %md
# MAGIC ## RegressionEvaluator — Helper function

# COMMAND ----------

from pyspark.ml.evaluation import RegressionEvaluator

def evaluate_regression(predictions, label_col="price"):
    """Print RMSE, MAE, and R² for a predictions DataFrame."""
    evaluator = RegressionEvaluator(labelCol=label_col, predictionCol="prediction")
    metrics = {}
    for m in ["rmse", "mse", "mae", "r2"]:
        metrics[m] = evaluator.evaluate(predictions, {evaluator.metricName: m})
    print(f"  RMSE: {metrics['rmse']:,.2f}")
    print(f"  MAE:  {metrics['mae']:,.2f}")
    print(f"  R²:   {metrics['r2']:.4f}")
    return metrics

# COMMAND ----------

# MAGIC %md
# MAGIC ## Linear Regression

# COMMAND ----------

from pyspark.ml.regression import LinearRegression

lr = LinearRegression(
    featuresCol="features",
    labelCol="price",
    maxIter=100,
    regParam=0.01,
    elasticNetParam=0.0,  # 0=L2, 1=L1
)
lr_model = lr.fit(train)

print("Coefficients:", lr_model.coefficients)
print("Intercept:   ", lr_model.intercept)

lr_preds = lr_model.transform(test)
print("\nLinear Regression:")
lr_metrics = evaluate_regression(lr_preds)

# COMMAND ----------

# Training summary (available on the model itself)
summary = lr_model.summary
print(f"Training RMSE: {summary.rootMeanSquaredError:,.2f}")
print(f"Training R²:   {summary.r2:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Decision Tree Regressor

# COMMAND ----------

from pyspark.ml.regression import DecisionTreeRegressor

dt = DecisionTreeRegressor(
    featuresCol="features",
    labelCol="price",
    maxDepth=8,
    minInstancesPerNode=5,
)
dt_model = dt.fit(train)
dt_preds = dt_model.transform(test)

print("Decision Tree Regressor:")
dt_metrics = evaluate_regression(dt_preds)
print("\nFeature importances:", dt_model.featureImportances)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Random Forest Regressor

# COMMAND ----------

from pyspark.ml.regression import RandomForestRegressor

rf = RandomForestRegressor(
    featuresCol="features",
    labelCol="price",
    numTrees=100,
    maxDepth=8,
    seed=42,
)
rf_model = rf.fit(train)
rf_preds = rf_model.transform(test)

print("Random Forest Regressor:")
rf_metrics = evaluate_regression(rf_preds)
print("\nFeature importances:", rf_model.featureImportances)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Gradient-Boosted Tree Regressor

# COMMAND ----------

from pyspark.ml.regression import GBTRegressor

gbt = GBTRegressor(
    featuresCol="features",
    labelCol="price",
    maxIter=100,
    maxDepth=5,
    stepSize=0.1,
    seed=42,
)
gbt_model = gbt.fit(train)
gbt_preds = gbt_model.transform(test)

print("GBT Regressor:")
gbt_metrics = evaluate_regression(gbt_preds)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generalized Linear Regression (GLM)
# MAGIC
# MAGIC Supports different families (gaussian, binomial, poisson, gamma, tweedie)
# MAGIC and link functions. Useful when you need a specific distributional assumption.

# COMMAND ----------

from pyspark.ml.regression import GeneralizedLinearRegression

glm = GeneralizedLinearRegression(
    featuresCol="features",
    labelCol="price",
    family="gaussian",  # normal distribution
    link="identity",    # identity link → standard linear regression
    maxIter=100,
    regParam=0.01,
)
glm_model = glm.fit(train)
glm_preds = glm_model.transform(test)

print("GLM (Gaussian, identity):")
glm_metrics = evaluate_regression(glm_preds)

# COMMAND ----------

# GLM summary
glm_summary = glm_model.summary
print("Coefficient standard errors:", glm_summary.coefficientStandardErrors)
print("P-values:                   ", glm_summary.pValues)
print("AIC:                        ", glm_summary.aic)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Side-by-side comparison

# COMMAND ----------

from pyspark.sql import Row

results = spark.createDataFrame([
    Row(model="Linear Regression", RMSE=lr_metrics["rmse"], MAE=lr_metrics["mae"], R2=lr_metrics["r2"]),
    Row(model="Decision Tree",     RMSE=dt_metrics["rmse"], MAE=dt_metrics["mae"], R2=dt_metrics["r2"]),
    Row(model="Random Forest",     RMSE=rf_metrics["rmse"], MAE=rf_metrics["mae"], R2=rf_metrics["r2"]),
    Row(model="GBT",               RMSE=gbt_metrics["rmse"], MAE=gbt_metrics["mae"], R2=gbt_metrics["r2"]),
    Row(model="GLM",               RMSE=glm_metrics["rmse"], MAE=glm_metrics["mae"], R2=glm_metrics["r2"]),
])
display(results)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluation metric reference
# MAGIC
# MAGIC | metricName | Formula | Notes |
# MAGIC |-----------|---------|-------|
# MAGIC | `rmse` (default) | sqrt(mean((y - ŷ)²)) | Penalizes large errors |
# MAGIC | `mse` | mean((y - ŷ)²) | Raw squared error |
# MAGIC | `mae` | mean(\|y - ŷ\|) | Robust to outliers |
# MAGIC | `r2` | 1 - SS_res / SS_tot | 1.0 = perfect fit, can be negative |
# MAGIC | `var` | explained variance | Similar to R² but doesn't account for bias |

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC **Next:** [04 — Clustering & Unsupervised Learning](./04_clustering)
