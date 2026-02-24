# Databricks notebook source
# MAGIC %md
# MAGIC # 07 — Databricks Tips & Best Practices for Spark ML
# MAGIC
# MAGIC Practical guidance for running `spark.ml` workloads efficiently on Databricks.
# MAGIC
# MAGIC **Topics:**
# MAGIC - Cluster sizing for ML
# MAGIC - Delta Lake integration
# MAGIC - Caching strategies
# MAGIC - Partitioning and data layout
# MAGIC - Databricks AutoML
# MAGIC - Unity Catalog for models
# MAGIC - Common pitfalls

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cluster sizing for ML workloads
# MAGIC
# MAGIC | Workload | Recommendation |
# MAGIC |----------|---------------|
# MAGIC | **Feature engineering** on large data | Scale **out** — more worker nodes, standard VMs |
# MAGIC | **Model training** (single model) | Scale **up** — fewer nodes with more memory per node |
# MAGIC | **Hyperparameter tuning** | Scale **out** — more workers + set `parallelism` in CrossValidator |
# MAGIC | **Deep learning / GPU** | Use GPU clusters (e.g., `g4dn.xlarge` on AWS, `NC` series on Azure) |
# MAGIC
# MAGIC ### General guidelines
# MAGIC
# MAGIC ```
# MAGIC Data size < 10 GB   →  Single node or 2-4 workers is fine
# MAGIC Data size 10-100 GB →  4-16 workers, memory-optimized VMs
# MAGIC Data size > 100 GB  →  16+ workers, consider Delta caching
# MAGIC
# MAGIC Always use Databricks ML Runtime for spark.ml workloads.
# MAGIC It includes pre-installed ML libraries and MLflow auto-logging.
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Use ML Runtime
# MAGIC
# MAGIC Always select **Databricks Runtime ML** (not the standard runtime) for ML
# MAGIC workloads:
# MAGIC
# MAGIC - Pre-installed: MLflow, scikit-learn, XGBoost, PyTorch, TensorFlow,
# MAGIC   Hugging Face, and more
# MAGIC - MLflow auto-logging enabled by default
# MAGIC - Optimized numpy/pandas via conda
# MAGIC - GPU support in the GPU variant

# COMMAND ----------

# MAGIC %md
# MAGIC ## Delta Lake integration
# MAGIC
# MAGIC Delta Lake is the default storage format on Databricks.  Use it for your ML
# MAGIC datasets to get ACID transactions, schema enforcement, and time travel.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Save training data as a Delta table

# COMMAND ----------

from pyspark.sql import SparkSession
import random

spark = SparkSession.builder.getOrCreate()

# Example: save processed features as a managed Delta table
random.seed(42)
rows = [(random.uniform(30000, 150000), random.randint(0, 20), random.uniform(1, 5), random.randint(0, 1))
        for _ in range(1000)]
df = spark.createDataFrame(rows, ["salary", "years_exp", "satisfaction", "left"])

# Write to Delta (managed table)
# df.write.format("delta").mode("overwrite").saveAsTable("my_catalog.my_schema.training_features")

# Write to Delta (external path)
delta_path = "/tmp/spark_ml_guide/training_features"
df.write.format("delta").mode("overwrite").save(delta_path)

# Read it back
training_df = spark.read.format("delta").load(delta_path)
print(f"Rows: {training_df.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Time travel — reproduce training data
# MAGIC
# MAGIC ```python
# MAGIC # Read the exact version used for training
# MAGIC df_v0 = spark.read.format("delta").option("versionAsOf", 0).load(delta_path)
# MAGIC
# MAGIC # Or by timestamp
# MAGIC df_ts = (spark.read.format("delta")
# MAGIC          .option("timestampAsOf", "2025-01-15T00:00:00")
# MAGIC          .load(delta_path))
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Caching strategies
# MAGIC
# MAGIC ML training iterates over the data multiple times.  Caching avoids
# MAGIC recomputing the feature pipeline each iteration.

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler

assembler = VectorAssembler(
    inputCols=["salary", "years_exp", "satisfaction"],
    outputCol="features",
)
features_df = assembler.transform(training_df)
train, test = features_df.randomSplit([0.8, 0.2], seed=42)

# Cache AFTER feature engineering, BEFORE training
train.cache()
train.count()  # materialize the cache

# Now train — each iteration reads from memory, not disk
from pyspark.ml.classification import RandomForestClassifier
model = RandomForestClassifier(featuresCol="features", labelCol="left", numTrees=100).fit(train)

# Unpersist when done to free memory
train.unpersist()

# COMMAND ----------

# MAGIC %md
# MAGIC ### When to cache vs not
# MAGIC
# MAGIC | Scenario | Cache? |
# MAGIC |----------|--------|
# MAGIC | Training data fits in cluster memory | Yes — `train.cache()` |
# MAGIC | Training data is huge (> 50% of cluster RAM) | No — use Delta with disk caching instead |
# MAGIC | Cross-validation / hyperparameter tuning | Yes — data is reused many times |
# MAGIC | Single `.transform()` call | No — data is read once |
# MAGIC
# MAGIC **Delta Cache (auto):** On Databricks, Delta Cache automatically caches
# MAGIC frequently-read Delta files on local SSDs.  This is transparent and doesn't
# MAGIC require `.cache()`.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Partitioning best practices
# MAGIC
# MAGIC | Tip | Why |
# MAGIC |-----|-----|
# MAGIC | Repartition before training if partitions are skewed | Prevents stragglers |
# MAGIC | Target ~128 MB per partition | Spark's sweet spot for shuffles |
# MAGIC | Use `.coalesce()` after filtering to reduce empty partitions | Avoids wasted tasks |
# MAGIC | Don't over-partition — too many small tasks adds overhead | |

# COMMAND ----------

# Check current partitioning
print(f"Partitions: {train.rdd.getNumPartitions()}")

# Repartition if needed (e.g., after heavy filtering)
# train = train.repartition(32)

# Coalesce after filtering (reduces partitions without a full shuffle)
# filtered = train.filter("salary > 50000").coalesce(16)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Databricks AutoML
# MAGIC
# MAGIC AutoML automatically trains and tunes a suite of models, generates a
# MAGIC leaderboard, and produces editable notebooks for the best runs.
# MAGIC
# MAGIC **When to use AutoML:**
# MAGIC - Quick baseline before investing in manual tuning
# MAGIC - Exploring which algorithm family works best
# MAGIC - Stakeholder demos with minimal setup
# MAGIC
# MAGIC **Important:** AutoML uses scikit-learn, XGBoost, and LightGBM under the hood
# MAGIC (not `spark.ml`).  It collects data to the driver via pandas, so dataset size
# MAGIC should be manageable (typically < 100 GB).

# COMMAND ----------

# MAGIC %md
# MAGIC ### AutoML via Python API

# COMMAND ----------

# Note: Only works on Databricks clusters, not locally
# from databricks import automl
#
# # Classification
# summary = automl.classify(
#     dataset=training_df,        # Spark DataFrame or table name
#     target_col="left",
#     primary_metric="roc_auc",
#     timeout_minutes=15,         # total time budget
#     max_trials=20,              # max models to try
# )
#
# # Access the best model
# print(f"Best trial: {summary.best_trial}")
# print(f"Best metric: {summary.best_trial.metrics['test_roc_auc']:.4f}")
#
# # The best model is logged to MLflow — load it:
# best_model = mlflow.sklearn.load_model(summary.best_trial.model_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ### AutoML via UI
# MAGIC
# MAGIC 1. Go to **Experiments** in the left sidebar
# MAGIC 2. Click **Create AutoML Experiment**
# MAGIC 3. Select your table, target column, and problem type
# MAGIC 4. Click **Start**
# MAGIC 5. Review the leaderboard and generated notebooks

# COMMAND ----------

# MAGIC %md
# MAGIC ## Unity Catalog for model governance
# MAGIC
# MAGIC Unity Catalog provides centralized access control, lineage tracking, and
# MAGIC discovery for ML models alongside your data assets.
# MAGIC
# MAGIC ```
# MAGIC ┌─────────────────────────────────────────┐
# MAGIC │  Unity Catalog                          │
# MAGIC │  ├── my_catalog                         │
# MAGIC │  │   ├── my_schema                      │
# MAGIC │  │   │   ├── training_features  (table) │
# MAGIC │  │   │   ├── attrition_rf       (model) │
# MAGIC │  │   │   └── predictions         (table)│
# MAGIC └─────────────────────────────────────────┘
# MAGIC ```
# MAGIC
# MAGIC ```python
# MAGIC import mlflow
# MAGIC mlflow.set_registry_uri("databricks-uc")
# MAGIC
# MAGIC # Register
# MAGIC mlflow.register_model("runs:/abc123/model", "my_catalog.my_schema.attrition_rf")
# MAGIC
# MAGIC # Grant access
# MAGIC # GRANT EXECUTE ON FUNCTION my_catalog.my_schema.attrition_rf TO `data-scientists`
# MAGIC
# MAGIC # Load for inference
# MAGIC model = mlflow.spark.load_model("models:/my_catalog.my_schema.attrition_rf@champion")
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Common pitfalls & solutions
# MAGIC
# MAGIC | Pitfall | Solution |
# MAGIC |---------|----------|
# MAGIC | **OOM on the driver** during `.toPandas()` or `collect()` | Use `spark.ml` end-to-end, or sample first: `df.sample(0.1).toPandas()` |
# MAGIC | **Slow training with many string columns** | Index and encode in a Pipeline — avoid doing it row-by-row with UDFs |
# MAGIC | **Data leakage in cross-validation** | Put all preprocessing inside the Pipeline so it's re-fit each fold |
# MAGIC | **Skewed classes** | Use `weightCol` parameter (supported by LR, RF, GBT) or oversample the minority class |
# MAGIC | **Model too large to log** | Use `mlflow.spark.log_model()` — it stores the model on the artifact store, not inline |
# MAGIC | **Stale cached data** | Call `.unpersist()` before re-caching after schema/data changes |
# MAGIC | **Feature drift in production** | Log training data stats with MLflow; monitor predictions with Lakehouse Monitoring |
# MAGIC | **StringIndexer fails on unseen labels** | Set `handleInvalid="keep"` or `"skip"` |
# MAGIC | **VectorAssembler fails on nulls** | Impute first, or set `handleInvalid="skip"` |

# COMMAND ----------

# MAGIC %md
# MAGIC ## Handling class imbalance

# COMMAND ----------

from pyspark.sql import functions as F

# Calculate class weights
label_counts = training_df.groupBy("left").count().collect()
total = sum(row["count"] for row in label_counts)
weights = {row["left"]: total / (len(label_counts) * row["count"]) for row in label_counts}
print("Class weights:", weights)

# Add a weight column
weighted_df = training_df.withColumn(
    "weight",
    F.when(F.col("left") == 1, weights[1]).otherwise(weights[0])
)

# Use in training
from pyspark.ml.classification import LogisticRegression
lr = LogisticRegression(
    featuresCol="features",
    labelCol="left",
    weightCol="weight",  # class-weighted training
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Spark configuration tips for ML

# COMMAND ----------

# Useful Spark configs for ML workloads

# Increase driver memory for model collection / broadcast
# spark.conf.set("spark.driver.memory", "16g")

# Increase parallelism for shuffles
# spark.conf.set("spark.sql.shuffle.partitions", "200")

# Enable adaptive query execution (on by default in Databricks)
print("AQE enabled:", spark.conf.get("spark.sql.adaptive.enabled", "not set"))

# Kryo serialization (faster for ML objects)
# spark.conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Workflow summary
# MAGIC
# MAGIC ```
# MAGIC 1. Store data in Delta Lake tables (Unity Catalog)
# MAGIC 2. Explore with AutoML for a quick baseline
# MAGIC 3. Build a spark.ml Pipeline (feature eng → model)
# MAGIC 4. Tune with CrossValidator (set parallelism)
# MAGIC 5. Track everything with MLflow (auto-log or manual)
# MAGIC 6. Register the best model in Unity Catalog
# MAGIC 7. Set alias "champion" and serve via Model Serving or batch inference
# MAGIC 8. Monitor with Lakehouse Monitoring
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC **That's the full guide!** Go back to the [README](../README.md) for the table of contents.
