# Databricks notebook source
# MAGIC %md
# MAGIC # 01 — Data Preparation & Feature Engineering
# MAGIC
# MAGIC Everything in `spark.ml` expects features as a **Vector** column. This notebook
# MAGIC covers the transformers you'll use most often to get your data into that shape.
# MAGIC
# MAGIC **Topics:**
# MAGIC - StringIndexer / IndexToString
# MAGIC - OneHotEncoder
# MAGIC - VectorAssembler
# MAGIC - Imputer
# MAGIC - StandardScaler / MinMaxScaler
# MAGIC - Bucketizer
# MAGIC - SQLTransformer
# MAGIC - Building a full Pipeline

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup — Create a sample dataset

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType

# On Databricks, `spark` is already available. This line is a no-op there,
# but makes the notebook runnable locally too.
spark = SparkSession.builder.getOrCreate()

data = [
    ("Alice",   "Engineering", 85000.0, 5,  "Senior",  None),
    ("Bob",     "Marketing",   62000.0, 2,  "Junior",  28.0),
    ("Charlie", "Engineering", 95000.0, 8,  "Senior",  35.0),
    ("Diana",   "Sales",       58000.0, 1,  "Junior",  24.0),
    ("Eve",     "Marketing",   72000.0, 4,  "Mid",     30.0),
    ("Frank",   "Sales",       68000.0, 3,  "Mid",     None),
    ("Grace",   "Engineering", 110000.0, 12, "Senior", 40.0),
    ("Hank",    "Marketing",   55000.0, 1,  "Junior",  22.0),
    ("Ivy",     "Sales",       75000.0, 6,  "Mid",     33.0),
    ("Jack",    "Engineering", 90000.0, 7,  "Senior",  32.0),
]

schema = StructType([
    StructField("name", StringType()),
    StructField("department", StringType()),
    StructField("salary", DoubleType()),
    StructField("years_exp", IntegerType()),
    StructField("level", StringType()),
    StructField("age", DoubleType()),
])

df = spark.createDataFrame(data, schema)
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## StringIndexer — Encode categorical strings as numeric indices
# MAGIC
# MAGIC `StringIndexer` maps each unique string to an integer (ordered by frequency).
# MAGIC Use `IndexToString` to reverse the mapping later (e.g., on predictions).

# COMMAND ----------

from pyspark.ml.feature import StringIndexer, IndexToString

# Index a single column
dept_indexer = StringIndexer(inputCol="department", outputCol="dept_index")
model = dept_indexer.fit(df)

indexed_df = model.transform(df)
display(indexed_df.select("name", "department", "dept_index"))

# COMMAND ----------

# Check the learned labels
print("Labels:", model.labels)

# COMMAND ----------

# Reverse the index back to the original string
idx_to_str = IndexToString(inputCol="dept_index", outputCol="dept_original", labels=model.labels)
display(idx_to_str.transform(indexed_df).select("name", "dept_index", "dept_original"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Indexing multiple columns at once
# MAGIC
# MAGIC You can pass lists to `inputCols` / `outputCols` to index several columns
# MAGIC in a single step.

# COMMAND ----------

multi_indexer = StringIndexer(
    inputCols=["department", "level"],
    outputCols=["dept_index", "level_index"],
)
indexed_df = multi_indexer.fit(df).transform(df)
display(indexed_df.select("department", "dept_index", "level", "level_index"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## OneHotEncoder — Sparse binary vectors from indices
# MAGIC
# MAGIC After indexing, use `OneHotEncoder` to create sparse one-hot vectors.
# MAGIC By default it drops the last category (k-1 encoding) to avoid
# MAGIC multicollinearity — set `dropLast=False` to keep all k categories.

# COMMAND ----------

from pyspark.ml.feature import OneHotEncoder

encoder = OneHotEncoder(
    inputCols=["dept_index", "level_index"],
    outputCols=["dept_ohe", "level_ohe"],
)
encoded_df = encoder.fit(indexed_df).transform(indexed_df)
display(encoded_df.select("department", "dept_ohe", "level", "level_ohe"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## VectorAssembler — Combine columns into a single feature vector
# MAGIC
# MAGIC This is the most-used transformer in any `spark.ml` workflow. It takes a list
# MAGIC of numeric / vector columns and concatenates them into one `features` column.

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler

assembler = VectorAssembler(
    inputCols=["salary", "years_exp", "dept_ohe", "level_ohe"],
    outputCol="features",
)
assembled_df = assembler.transform(encoded_df)
display(assembled_df.select("name", "features"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Handling nulls in VectorAssembler
# MAGIC
# MAGIC By default VectorAssembler errors on null values.  Set `handleInvalid`
# MAGIC to change this behavior.

# COMMAND ----------

# Options: "error" (default), "skip" (drop rows), "keep" (replace with NaN)
assembler_safe = VectorAssembler(
    inputCols=["salary", "years_exp"],
    outputCol="features_safe",
    handleInvalid="skip",
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Imputer — Fill missing numeric values
# MAGIC
# MAGIC Replaces nulls/NaNs with the **mean** or **median** of each column.

# COMMAND ----------

from pyspark.ml.feature import Imputer

imputer = Imputer(
    inputCols=["age"],
    outputCols=["age_imputed"],
    strategy="median",  # "mean" (default), "median", or "mode"
)
imputed_df = imputer.fit(df).transform(df)
display(imputed_df.select("name", "age", "age_imputed"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## StandardScaler — Zero-mean, unit-variance scaling
# MAGIC
# MAGIC Operates on a **Vector** column. You typically assemble first, then scale.

# COMMAND ----------

from pyspark.ml.feature import StandardScaler

# Build a quick numeric vector
num_assembler = VectorAssembler(inputCols=["salary", "years_exp"], outputCol="num_features")
num_df = num_assembler.transform(df)

scaler = StandardScaler(
    inputCol="num_features",
    outputCol="scaled_features",
    withMean=True,   # center to zero mean (requires dense vectors)
    withStd=True,    # scale to unit variance
)
scaled_df = scaler.fit(num_df).transform(num_df)
display(scaled_df.select("name", "num_features", "scaled_features"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## MinMaxScaler — Scale to [0, 1] range

# COMMAND ----------

from pyspark.ml.feature import MinMaxScaler

mm_scaler = MinMaxScaler(inputCol="num_features", outputCol="minmax_features")
mm_df = mm_scaler.fit(num_df).transform(num_df)
display(mm_df.select("name", "num_features", "minmax_features"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Bucketizer — Discretize continuous values into bins

# COMMAND ----------

from pyspark.ml.feature import Bucketizer

bucketizer = Bucketizer(
    splits=[0.0, 60000.0, 80000.0, 100000.0, float("inf")],
    inputCol="salary",
    outputCol="salary_bucket",
)
bucketed_df = bucketizer.transform(df)
display(bucketed_df.select("name", "salary", "salary_bucket"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## SQLTransformer — Apply arbitrary SQL as a transform step
# MAGIC
# MAGIC Useful for adding computed columns inside a Pipeline without leaving the
# MAGIC `spark.ml` API.  Reference the input DataFrame as `__THIS__`.

# COMMAND ----------

from pyspark.ml.feature import SQLTransformer

sql_tx = SQLTransformer(
    statement="SELECT *, salary / years_exp AS salary_per_year FROM __THIS__"
)
display(sql_tx.transform(df.filter("years_exp > 0")).select("name", "salary", "years_exp", "salary_per_year"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Putting it all together — Pipeline
# MAGIC
# MAGIC A `Pipeline` chains transformers and estimators into a single, reusable object.
# MAGIC Call `.fit()` once to produce a `PipelineModel` that you can `.transform()` on
# MAGIC new data.

# COMMAND ----------

from pyspark.ml import Pipeline

pipeline = Pipeline(stages=[
    # 1. Impute missing age values
    Imputer(inputCols=["age"], outputCols=["age_imputed"], strategy="median"),
    # 2. Index categorical columns
    StringIndexer(inputCols=["department", "level"], outputCols=["dept_idx", "level_idx"]),
    # 3. One-hot encode
    OneHotEncoder(inputCols=["dept_idx", "level_idx"], outputCols=["dept_ohe", "level_ohe"]),
    # 4. Assemble all features
    VectorAssembler(
        inputCols=["salary", "years_exp", "age_imputed", "dept_ohe", "level_ohe"],
        outputCol="features",
    ),
    # 5. Scale
    StandardScaler(inputCol="features", outputCol="scaled_features", withMean=False, withStd=True),
])

pipeline_model = pipeline.fit(df)
final_df = pipeline_model.transform(df)
display(final_df.select("name", "features", "scaled_features"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Pipeline key points
# MAGIC
# MAGIC | Concept | Detail |
# MAGIC |---------|--------|
# MAGIC | **Transformer** | Has `.transform(df)` — produces a new DataFrame |
# MAGIC | **Estimator** | Has `.fit(df)` — produces a Transformer (Model) |
# MAGIC | **Pipeline** | Ordered list of stages (Transformers + Estimators) |
# MAGIC | **PipelineModel** | Result of `Pipeline.fit()` — all stages are now Transformers |
# MAGIC | **Reusability** | Save the PipelineModel and `.transform()` new data at inference time |

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC **Next:** [02 — Classification](./02_classification)
