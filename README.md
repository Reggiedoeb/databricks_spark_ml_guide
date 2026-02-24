# Spark ML Guide for Databricks

A practical, hands-on guide to machine learning with `pyspark.ml` on Databricks. Each notebook is self-contained with runnable examples you can import directly into a Databricks workspace.

## Prerequisites

- Databricks workspace (Community Edition works for most examples)
- Databricks Runtime 13.3 LTS or later (ML Runtime recommended)
- Basic familiarity with PySpark DataFrames

## Guide Structure

| # | Notebook | Topics |
|---|----------|--------|
| 01 | [Data Preparation & Feature Engineering](notebooks/01_data_preparation.py) | VectorAssembler, StringIndexer, OneHotEncoder, Imputer, Scalers, Bucketizer, Pipeline |
| 02 | [Classification](notebooks/02_classification.py) | LogisticRegression, DecisionTree, RandomForest, GBT, evaluation metrics |
| 03 | [Regression](notebooks/03_regression.py) | LinearRegression, tree-based regressors, GBT, evaluation metrics |
| 04 | [Clustering & Unsupervised Learning](notebooks/04_clustering.py) | KMeans, BisectingKMeans, GaussianMixture, LDA |
| 05 | [Hyperparameter Tuning](notebooks/05_tuning.py) | CrossValidator, TrainValidationSplit, ParamGridBuilder, parallelism |
| 06 | [Model Persistence & MLflow](notebooks/06_mlflow_integration.py) | Save/load models, MLflow tracking, Model Registry, serving |
| 07 | [Databricks Tips & Best Practices](notebooks/07_databricks_tips.py) | Cluster sizing, Delta Lake, caching, AutoML, Unity Catalog |

## Quick Start

1. Clone or download this repo.
2. Import the `notebooks/` folder into your Databricks workspace (**Workspace > Import**).
3. Attach a cluster running **Databricks Runtime ML 13.3+**.
4. Run notebooks in order, or jump to the topic you need.

## Key Concepts at a Glance

```
spark.ml workflow
─────────────────
Raw Data
  │
  ▼
Transformers  (StringIndexer, VectorAssembler, Scaler, ...)
  │             ↳ transform(df) → df
  ▼
Estimators    (LogisticRegression, RandomForest, KMeans, ...)
  │             ↳ fit(df) → Model (which is itself a Transformer)
  ▼
Pipeline      (chain Transformers + Estimators into a single workflow)
  │             ↳ fit(df) → PipelineModel
  ▼
Evaluator     (BinaryClassificationEvaluator, RegressionEvaluator, ...)
  │             ↳ evaluate(predictions) → metric
  ▼
Tuning        (CrossValidator / TrainValidationSplit + ParamGridBuilder)
                ↳ fit(df) → best Model
```

## Imports Cheat Sheet

```python
# Core
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    VectorAssembler, StringIndexer, OneHotEncoder,
    StandardScaler, MinMaxScaler, Imputer, Bucketizer,
    SQLTransformer, IndexToString
)

# Classification
from pyspark.ml.classification import (
    LogisticRegression, DecisionTreeClassifier,
    RandomForestClassifier, GBTClassifier,
    MultilayerPerceptronClassifier
)

# Regression
from pyspark.ml.regression import (
    LinearRegression, DecisionTreeRegressor,
    RandomForestRegressor, GBTRegressor,
    GeneralizedLinearRegression
)

# Clustering
from pyspark.ml.clustering import KMeans, BisectingKMeans, GaussianMixture, LDA

# Evaluation
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
    RegressionEvaluator,
    ClusteringEvaluator
)

# Tuning
from pyspark.ml.tuning import CrossValidator, TrainValidationSplit, ParamGridBuilder
```
