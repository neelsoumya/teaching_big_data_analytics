# Big Data Analytics — Course Materials (Markdown)

**Module title**: *Big Data Analytics*

**Assessment split:** **100% coursework** — Technical Report **30%**, Discussion Board **10%**, Big Data Architecture Project **60%**.
Covers data pre-processing, legal/ethical issues, MapReduce/Spark, graph analytics, and future trends. Learning outcomes are targeted across lectures, labs and assessments.

---

## Contents

1. 12-week lecture & lab plan (high level)
2. Practicals (code + student tasks)

   * Practical A — pandas (Data cleaning & EDA)
   * Practical B — PySpark (MapReduce pattern)
   * Practical C — NetworkX (Graph analytics)
   * Requirements snippet
3. Assessments & rubrics

   * Assessment 1 — Technical Paper (30%)
   * Assessment 2 — Discussion Board (10%)
   * Assessment 3 — Big Data Architecture Project (60%)
   * Delivery schedule / milestones
4. Example assessment scaffolds
5. Formative tasks, datasets & resources
6. Example exam-style questions
7. Teaching tips & marking consistency

---

## 1 — 12-week lecture & lab plan (high level)

**Format each week:** 1.5 hr lecture + 1 hr tutorial + 2.5 hr lab/practical (where indicated).

* **Week 1 — Intro & Overview; data stages**
  Definitions, data life-cycle, types of big data, scripting language choice (Python).

* **Week 2 — Data acquisition & storage**
  RDBMS vs NoSQL vs object stores; sharding, partitioning.

* **Week 3 — Data pre-processing I (pandas & pipelines)**
  Cleaning, missing data, type conversion, scaling.

* **Week 4 — Data pre-processing II (ETL & streaming basics)**
  Batch vs streaming, data locality principles.

* **Week 5 — Exploratory data analysis & visualization**
  Aggregation, grouping, plotting for big data summaries.

* **Week 6 — Legal & ethical issues**
  GDPR, data minimisation, bias, provenance, security.

* **Week 7 — MapReduce fundamentals & Hadoop design**
  Map/reduce flow, combiner, partitioner, driver code.

* **Week 8 — Spark & distributed computation**
  RDDs, DataFrames, transformations/actions, performance tuning.

* **Week 9 — Graph theory & network analysis**
  Centrality, random walks, community detection.

* **Week 10 — Graph analytics at scale**
  GraphFrames (Spark), Neo4j overview, use cases (social networks, knowledge graphs).

* **Week 11 — Decision trees, scripted decision systems & ML basics for big data**
  Model training at scale, feature engineering.

* **Week 12 — Maintenance, feasibility & future trends**
  Monitoring, data pipelines, reproducibility, data mesh / lakehouse trends.

> **Recommended readings (per week)** — e.g., McKinney (pandas), Damji et al. (Spark), Spark book, selected papers/extracts. (Add to reading list in course site.)

---

## 2 — Three practicals (with runnable Python code)

> **Note:** Self-contained; uses `pandas`, `matplotlib`, `pyspark`, `networkx`. Replace dataset paths with your chosen CSV/Parquet/Kaggle files.

### Practical A — Data cleaning & EDA with `pandas` (2.5 hr lab)

**Learning goals:** clean a CSV, handle missing values, type casting, aggregation, feature engineering, plotting.

**Filename:** `practical_a_pandas.py`

```python
# practical_a_pandas.py
# Requirements: pandas, matplotlib, numpy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. load (replace path with your dataset)
df = pd.read_csv("data/sample_transactions.csv", parse_dates=["timestamp"])

# 2. quick audit
print(df.info())
print(df.isnull().sum())

# 3. clean: drop duplicates, fill missing amounts with median, parse categories
df = df.drop_duplicates()
df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
df['amount'].fillna(df['amount'].median(), inplace=True)
df['category'] = df['category'].fillna('UNKNOWN')

# 4. feature engineering: day of week, hour, amount_bin
df['dayofweek'] = df['timestamp'].dt.day_name()
df['hour'] = df['timestamp'].dt.hour
df['amount_bin'] = pd.qcut(df['amount'], q=4, labels=['low','med_low','med_high','high'])

# 5. aggregation example: daily totals and top categories
daily = df.groupby(df['timestamp'].dt.date).agg(
    total_amount=('amount','sum'),
    count=('amount','count')
).reset_index().rename(columns={'timestamp':'date'})

top_categories = df.groupby('category')['amount'].sum().sort_values(ascending=False).head(10)

# 6. quick plotting
plt.figure(figsize=(8,4))
plt.plot(daily['timestamp'], daily['total_amount'])
plt.title('Daily total amount')
plt.xlabel('Date'); plt.ylabel('Total amount')
plt.tight_layout()
plt.show()

print("Top categories:\n", top_categories)
```

**Student tasks:** run on provided dataset → produce a 1-page summary (figures + short table) + submit cleaned dataset with brief explanation of cleaning decisions.

---

### Practical B — MapReduce pattern with PySpark (2.5 hr lab)

**Learning goals:** demonstrate MapReduce pattern, Spark DataFrame/RDD APIs, aggregation & wordcount example.

**Filename:** `practical_b_pyspark.py`

```python
# practical_b_pyspark.py
# Requirements: pyspark
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = SparkSession.builder \
    .appName("PracticalB_MapReduce") \
    .master("local[*]") \
    .config("spark.sql.shuffle.partitions", "4") \
    .getOrCreate()

# Example: compute counts by category from a CSV
df = spark.read.csv("data/big_events.csv", header=True, inferSchema=True)

# show schema and a sample
df.printSchema()
df.show(5)

# map-reduce pattern: groupBy (map -> key -> reduce)
counts = df.groupBy("category").agg(
    F.count("*").alias("n_events"),
    F.sum("value").alias("sum_value"),
    F.avg("value").alias("avg_value")
).orderBy(F.desc("n_events"))

counts.show(20, truncate=False)

# Example: wordcount on 'text' column (demonstrates flatMap / reduceByKey)
# (convert DataFrame column to RDD of tokens)
rdd = df.select("text").rdd.flatMap(lambda row: row[0].split() if row[0] else [])
word_counts = rdd.map(lambda w: (w.lower().strip(), 1)).reduceByKey(lambda a,b: a+b)
top = word_counts.takeOrdered(20, key=lambda x: -x[1])
print("Top words:", top)

spark.stop()
```

**Student tasks:** vary `spark.sql.shuffle.partitions` and record runtimes; write short report on partitioning, shuffle cost, and data locality.

---

### Practical C — Graph analytics with NetworkX (2.5 hr lab)

**Learning goals:** build graphs, compute centrality, simulate random walks, small community detection.

**Filename:** `practical_c_networkx.py`

```python
# practical_c_networkx.py
# Requirements: networkx, matplotlib
import networkx as nx
import matplotlib.pyplot as plt
import random

# Build graph (edge list CSV: source,target,weight)
G = nx.read_weighted_edgelist("data/sample_edges.csv", delimiter=",", nodetype=str)

print("Nodes:", G.number_of_nodes(), "Edges:", G.number_of_edges())

# centrality measures
degree_cent = nx.degree_centrality(G)
betweenness = nx.betweenness_centrality(G, normalized=True)
pagerank = nx.pagerank(G)

# top 10 by PageRank
top_pr = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:10]
print("Top PageRank nodes:", top_pr)

# random walk simulation for exploration (simple)
def random_walk(G, start, steps=100):
    path = [start]
    cur = start
    for _ in range(steps):
        nbrs = list(G.neighbors(cur))
        if not nbrs:
            break
        cur = random.choice(nbrs)
        path.append(cur)
    return path

sample_path = random_walk(G, start=list(G.nodes())[0], steps=50)
print("Sample random walk length:", len(sample_path))

# draw small graph (only for small graphs)
if G.number_of_nodes() <= 200:
    plt.figure(figsize=(8,6))
    pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx_nodes(G, pos, node_size=30)
    nx.draw_networkx_edges(G, pos, alpha=0.3)
    plt.title("Graph preview")
    plt.axis('off')
    plt.show()
```

**Student tasks:** compute and compare degree, betweenness, PageRank for a social-network sample and write **500 words** interpreting influence/connectivity and a business application.

---

### Requirements snippet (`requirements.txt`)

```
pandas
numpy
matplotlib
pyspark
networkx
```

---

## 3 — Assessment briefs & rubrics (adapted to module spec)

**Module:** F-1154268957 — *Big Data Analytics*

### Assessment 1 — Technical Paper (30%)

**Brief:** 3,000–3,500 words. Compare & critically review **≥3** databases/tools for big data (e.g., HBase, MongoDB, Cassandra, BigQuery, Delta Lake) **OR** review a hot trend (e.g., streaming, data mesh). Include architecture diagrams, performance considerations, legal/ethical implications, and a recommendation for a named industry scenario.
**LOs assessed:** 3, 5, 6.

**Rubric (30 marks)**

* **Argument & depth (10):** clear thesis, critical assessment, evidence (benchmarks/papers).
* **Technical accuracy (8):** correct internals, trade-offs, scaling.
* **Ethical/legal analysis (4):** GDPR, privacy, bias considered.
* **Presentation & references (4):** diagrams, citations, clarity.
* **Originality & recommendation (4):** actionable recommendation with rationale.

---

### Assessment 2 — Discussion Board (10%)

**Brief:** Active participation across term (min **8** substantive posts: 4 project updates, 2 peer feedback, 2 topical reflections). Posts ≥150 words, reference sources, provide critique.
**LOs assessed:** 1,2,3,4,5.

**Rubric (10 marks)**

* **Quality of contributions (6):** insight, evidence, constructive feedback.
* **Engagement & responsiveness (2):** replies to peers, timely posts.
* **Professionalism (2):** clarity, referencing, tone.

---

### Assessment 3 — Big Data Architecture Project (60%) — group (3–4 people)

**Brief:** Design, implement & present a Big Data Analytics solution to a supplied business problem (or vetted student scenario). Deliverables: architecture document, ETL pipeline code (Spark or equivalent), analysis notebook/visuals, final presentation (10–15 min demo). Emphasise data transformation, scaling, graph analysis (if relevant), and legal/ethical mitigation. Final presentations Week 12.
**LOs assessed:** 1,2,3,4,5,6,7,8.

**Rubric (60 marks)**

* **Architecture & design (16):** clarity, scaling strategy, components.
* **Implementation & code quality (16):** functioning pipeline, reproducibility, docs, tests.
* **Analysis & interpretation (12):** analytics correctness, visuals, business insight.
* **Ethics, security & legal compliance (8):** privacy-by-design, risk assessment.
* **Teamwork & presentation (8):** demo clarity, role distribution, Q&A.

**Delivery schedule (suggested milestones):**

* **Week 4:** Architecture sketch (milestone 1)
* **Week 7:** Working ETL + sample outputs (milestone 2)
* **Week 10:** Draft report
* **Week 12:** Final demo & submission

---

## 4 — Example assessment scaffold documents (copy-paste ready)

### Project milestone 1 (Week 4) — Architecture sketch (10% of project mark)

* One A3 diagram (components, data flows, storage types).
* One-page rationale (why chosen).
* Brief risks & mitigation list.

### Project milestone 2 (Week 7) — Working ETL (25% of project mark)

* Working Spark job or Python ETL script.
* README with run instructions (local).
* Sample outputs (CSV/Parquet).
* Short performance measurements table (time, partitions, memory).

### Final submission (Week 12) — full project (remaining project mark)

* Full code + tests.
* Architecture doc.
* 6–8 slide presentation + recorded 10–15min demo.

---

## 5 — Suggested formative & summative in-class tasks

* **Quick polls (5 min):** Which storage would you choose and why?
* **Minute paper:** One thing unclear, one thing useful (end of lecture).
* **Peer code review session (lab):** swap notebooks + give two improvement suggestions.
* **Ethics scenario workshop:** small groups analyze a re-identification case; present mitigations.

---

## 6 — Suggested datasets & resources

* Small/medium public datasets for labs & projects: transaction logs, social graph subsets, OpenWeather snapshots, Kaggle samples.
* Recommended sources: Kaggle, Stanford SNAP, DBpedia, OpenWeather. (Use subsets/filters so labs run locally.)

---

## 7 — Example exam-style / formative questions

1. **Short answer:** Explain the MapReduce flow and the role of the combiner — when does a combiner improve performance?
2. **List/bullet:** Given a Spark DataFrame `df` with **100M** rows, list **5 strategies** to improve shuffle performance.
3. **Calculation + interpretation:** Given a small network, compute degree centrality vs PageRank for a highlighted node — compare and interpret.

---

## 8 — Teaching tips & marking consistency

* Use explicit rubrics for all submissions; provide annotated feedback linked to rubric items.
* Encourage reproducible environments: `requirements.txt`, Docker, Binder or Colab notebooks.
* For group work, require a short peer-assessment form to adjust marks for unequal contributions.

---
