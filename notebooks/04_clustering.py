# Databricks notebook source
# MAGIC %md
# MAGIC # 04 — Clustering & Unsupervised Learning
# MAGIC
# MAGIC Discover structure in unlabeled data.
# MAGIC
# MAGIC **Models covered:**
# MAGIC - KMeans
# MAGIC - BisectingKMeans
# MAGIC - Gaussian Mixture Model (GMM)
# MAGIC - Latent Dirichlet Allocation (LDA) — topic modeling
# MAGIC
# MAGIC **Evaluator:** `ClusteringEvaluator` (silhouette score)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup — Synthetic customer dataset

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml import Pipeline
import random

spark = SparkSession.builder.getOrCreate()

# 3 synthetic clusters: budget, mid-range, premium customers
random.seed(42)
rows = []
for _ in range(400):
    # Budget cluster
    rows.append((random.gauss(25, 8), random.gauss(20, 10), random.gauss(2, 1)))
for _ in range(350):
    # Mid-range cluster
    rows.append((random.gauss(45, 10), random.gauss(55, 12), random.gauss(5, 1.5)))
for _ in range(250):
    # Premium cluster
    rows.append((random.gauss(55, 8), random.gauss(85, 10), random.gauss(10, 2)))

df = spark.createDataFrame(rows, ["age", "annual_spend_k", "visits_per_month"])

# Assemble and scale
pipeline = Pipeline(stages=[
    VectorAssembler(inputCols=["age", "annual_spend_k", "visits_per_month"], outputCol="raw_features"),
    StandardScaler(inputCol="raw_features", outputCol="features", withMean=False, withStd=True),
])
prepped = pipeline.fit(df).transform(df)
display(df.describe())

# COMMAND ----------

# MAGIC %md
# MAGIC ## KMeans

# COMMAND ----------

from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

kmeans = KMeans(
    featuresCol="features",
    predictionCol="cluster",
    k=3,
    seed=42,
    maxIter=20,
    initMode="k-means||",  # scalable k-means++ (default)
)
km_model = kmeans.fit(prepped)
km_preds = km_model.transform(prepped)

# Cluster centers
for i, center in enumerate(km_model.clusterCenters()):
    print(f"  Cluster {i}: {center}")

# COMMAND ----------

# Silhouette score (higher is better, range [-1, 1])
evaluator = ClusteringEvaluator(
    featuresCol="features",
    predictionCol="cluster",
    metricName="silhouette",
    distanceMeasure="squaredEuclidean",
)
silhouette = evaluator.evaluate(km_preds)
print(f"Silhouette score: {silhouette:.4f}")

# COMMAND ----------

# Cluster sizes
display(km_preds.groupBy("cluster").count().orderBy("cluster"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Elbow method — choosing k
# MAGIC
# MAGIC Run KMeans for several values of k and plot the cost (within-set sum of
# MAGIC squared errors).  Look for the "elbow" where improvement slows.

# COMMAND ----------

costs = []
silhouettes = []

for k in range(2, 9):
    km = KMeans(featuresCol="features", k=k, seed=42).fit(prepped)
    preds = km.transform(prepped)
    cost = km.summary.trainingCost
    sil = evaluator.evaluate(preds)
    costs.append((k, cost, sil))
    print(f"  k={k}  cost={cost:,.0f}  silhouette={sil:.4f}")

# On Databricks, you can visualize this with display() on a DataFrame
cost_df = spark.createDataFrame(costs, ["k", "cost", "silhouette"])
display(cost_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## BisectingKMeans
# MAGIC
# MAGIC A hierarchical variant that recursively splits the cluster with the
# MAGIC largest cost.  Often faster and more deterministic than standard KMeans.

# COMMAND ----------

from pyspark.ml.clustering import BisectingKMeans

bkm = BisectingKMeans(
    featuresCol="features",
    predictionCol="cluster",
    k=3,
    seed=42,
)
bkm_model = bkm.fit(prepped)
bkm_preds = bkm_model.transform(prepped)

sil = evaluator.evaluate(bkm_preds)
print(f"BisectingKMeans silhouette: {sil:.4f}")
display(bkm_preds.groupBy("cluster").count().orderBy("cluster"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Gaussian Mixture Model (GMM)
# MAGIC
# MAGIC Soft clustering — each point gets a probability distribution over clusters.
# MAGIC Captures elliptical cluster shapes that KMeans cannot.

# COMMAND ----------

from pyspark.ml.clustering import GaussianMixture

gmm = GaussianMixture(
    featuresCol="features",
    predictionCol="cluster",
    probabilityCol="cluster_prob",
    k=3,
    seed=42,
)
gmm_model = gmm.fit(prepped)
gmm_preds = gmm_model.transform(prepped)

# Each row gets a probability vector
display(gmm_preds.select("age", "annual_spend_k", "cluster", "cluster_prob").limit(10))

# COMMAND ----------

# GMM summary
print(f"Log-likelihood: {gmm_model.summary.logLikelihood:.2f}")
for i, gauss in enumerate(gmm_model.gaussiansDF.collect()):
    print(f"  Component {i}: mean={gauss['mean']}, cov={gauss['cov']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Latent Dirichlet Allocation (LDA) — Topic Modeling
# MAGIC
# MAGIC LDA discovers topics in a corpus.  Input is a **sparse vector** of word counts
# MAGIC (bag of words).  Each document is a distribution over topics, and each topic is
# MAGIC a distribution over words.

# COMMAND ----------

from pyspark.ml.clustering import LDA
from pyspark.ml.feature import CountVectorizer, Tokenizer, StopWordsRemover

# Toy corpus
docs = spark.createDataFrame([
    (0, "spark ml machine learning pipeline feature engineering"),
    (1, "deep learning neural network tensorflow pytorch"),
    (2, "spark dataframe sql query optimization catalyst"),
    (3, "machine learning model training evaluation metrics"),
    (4, "sql database table join query performance"),
    (5, "neural network layers activation backpropagation gradient"),
    (6, "spark streaming kafka real time data processing"),
    (7, "model selection cross validation hyperparameter tuning"),
], ["id", "text"])

# Tokenize → remove stop words → count vectorize
tokenizer = Tokenizer(inputCol="text", outputCol="words")
remover = StopWordsRemover(inputCol="words", outputCol="filtered")
cv = CountVectorizer(inputCol="filtered", outputCol="features", vocabSize=100)

text_pipeline = Pipeline(stages=[tokenizer, remover, cv])
text_model = text_pipeline.fit(docs)
corpus = text_model.transform(docs)

# COMMAND ----------

lda = LDA(
    featuresCol="features",
    k=3,             # number of topics
    maxIter=20,
    seed=42,
)
lda_model = lda.fit(corpus)

print(f"Log-likelihood:  {lda_model.logLikelihood(corpus):.2f}")
print(f"Log-perplexity:  {lda_model.logPerplexity(corpus):.2f}")

# COMMAND ----------

# Top words per topic
vocab = text_model.stages[-1].vocabulary  # CountVectorizer vocabulary
topics = lda_model.describeTopics(maxTermsPerTopic=5)

for row in topics.collect():
    topic_words = [vocab[idx] for idx in row["termIndices"]]
    print(f"  Topic {row['topic']}: {topic_words} (weights: {[round(w, 3) for w in row['termWeights']]})")

# COMMAND ----------

# Per-document topic distribution
doc_topics = lda_model.transform(corpus)
display(doc_topics.select("id", "text", "topicDistribution"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Quick reference
# MAGIC
# MAGIC | Algorithm | Type | Key params | Output |
# MAGIC |-----------|------|-----------|--------|
# MAGIC | **KMeans** | Hard clustering | `k`, `maxIter`, `initMode` | `prediction` (cluster ID) |
# MAGIC | **BisectingKMeans** | Hierarchical hard | `k`, `minDivisibleClusterSize` | `prediction` |
# MAGIC | **GaussianMixture** | Soft clustering | `k`, `tol` | `prediction`, `probability` vector |
# MAGIC | **LDA** | Topic modeling | `k`, `maxIter`, `optimizer` | `topicDistribution` per document |
# MAGIC
# MAGIC | Metric | Evaluator | Range | Notes |
# MAGIC |--------|-----------|-------|-------|
# MAGIC | Silhouette | `ClusteringEvaluator` | [-1, 1] | Higher = better separated clusters |
# MAGIC | Training cost | `model.summary.trainingCost` | ≥ 0 | Within-set sum of squared errors |
# MAGIC | Log-likelihood | LDA `model.logLikelihood()` | ≤ 0 | Higher (less negative) = better fit |
# MAGIC | Perplexity | LDA `model.logPerplexity()` | ≥ 0 | Lower = better generalization |

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC **Next:** [05 — Hyperparameter Tuning](./05_tuning)
