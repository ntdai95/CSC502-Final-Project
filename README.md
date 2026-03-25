# CSC502-Final-Project

## 1. How to run

### Step 0: Confirm raw data exists

<pre class="overflow-visible! px-0!" data-start="2996" data-end="3038"><div class="relative w-full mt-4 mb-1"><div class=""><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback"><div class="pointer-events-none absolute inset-x-4 top-12 bottom-4"><div class="pointer-events-none sticky z-40 shrink-0 z-1!"><div class="sticky bg-token-border-light"></div></div></div><div class=""><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼd ͼr"><div class="cm-scroller"><div class="cm-content q9tKkq_readonly"><span>Make sure the 'ebird_data_raw.txt' file is in the 'data' folder</span></div></div></div></div></div></div></div></div></div><div class=""><div class=""></div></div></div></div></div></pre>

### Step 1: Process raw data

<pre class="overflow-visible! px-0!" data-start="2996" data-end="3038"><div class="relative w-full mt-4 mb-1"><div class=""><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback"><div class="pointer-events-none absolute inset-x-4 top-12 bottom-4"><div class="pointer-events-none sticky z-40 shrink-0 z-1!"><div class="sticky bg-token-border-light"></div></div></div><div class=""><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼd ͼr"><div class="cm-scroller"><div class="cm-content q9tKkq_readonly"><span>python scripts/process_data.py</span></div></div></div></div></div></div></div></div></div><div class=""><div class=""></div></div></div></div></div></pre>

### Step 2: Run Isolation Forest

<pre class="overflow-visible! px-0!" data-start="3073" data-end="3114"><div class="relative w-full mt-4 mb-1"><div class=""><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback"><div class="pointer-events-none absolute inset-x-4 top-12 bottom-4"><div class="pointer-events-none sticky z-40 shrink-0 z-1!"><div class="sticky bg-token-border-light"></div></div></div><div class=""><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼd ͼr"><div class="cm-scroller"><div class="cm-content q9tKkq_readonly"><span>python -m scripts.run_iforest</span></div></div></div></div></div></div></div></div></div><div class=""><div class=""></div></div></div></div></div></pre>

### Step 3: Run experiments

<pre class="overflow-visible! px-0!" data-start="3144" data-end="3189"><div class="relative w-full mt-4 mb-1"><div class=""><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback"><div class="pointer-events-none absolute inset-x-4 top-12 bottom-4"><div class="pointer-events-none sticky z-40 shrink-0 z-1!"><div class="sticky bg-token-border-light"></div></div></div><div class=""><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼd ͼr"><div class="cm-scroller"><div class="cm-content q9tKkq_readonly"><span>python -m scripts.run_experiments</span></div></div></div></div></div></div></div></div></div></div></div></div></pre>

---

# 2. Dataset and Ground Truth Labels

The processed dataset contains the following columns:

| Column            | Description                              |
| ----------------- | ---------------------------------------- |
| TAXON CONCEPT ID  | Species identifier                       |
| species_frequency | Relative frequency of species in dataset |
| OBSERVATION COUNT | Number of individuals observed           |
| LATITUDE          | Observation latitude                     |
| LONGITUDE         | Observation longitude                    |
| day_sin           | Cyclical encoding of day of year         |
| day_cos           | Cyclical encoding of day of year         |
| REVIEWED          | eBird review flag                        |

The `REVIEWED` column indicates whether an observation was flagged by the eBird review system for manual verification. Observations may be flagged when:

* the species is unusual for the location
* the species is unusual for the time of year
* the observation count is unusually high

This field is used as a  **ground-truth anomaly label for evaluation only**, and is **not used as a feature during model training**.

---


# 3. Data Preprocessing

The raw eBird dataset is processed using Apache Spark to produce a clean dataset suitable for Isolation Forest.

The preprocessing pipeline performs the following steps:

* keep only rows where `CATEGORY = species`
* remove provisional and escaped species observations
* keep only `Traveling` or `Stationary` observation types
* keep only complete checklists
* filter observations with duration ≤ 300 minutes
* filter observations with distance ≤ 10 km
* convert numeric columns to proper numeric types
* encode observation date using cyclic sine and cosine transformations
* compute species frequency as a numerical representation of species identity

The resulting processed dataset is exported as a single CSV file.

---

# 4. Implementation

The Isolation Forest algorithm was implemented from scratch in Python.

The implementation includes the following modules:

| File                 | Purpose                                                        |
| -------------------- | -------------------------------------------------------------- |
| `itree.py`         | Isolation Tree construction                                    |
| `iforest.py`       | Isolation Forest ensemble                                      |
| `iforest_math.py`  | mathematical functions such asc(n)c(n)**c**(**n**) |
| `data_utils.py`    | dataset loading and feature extraction                         |
| `run_pipeline.py`  | anomaly scoring pipeline                                       |
| `experiments.py`   | experiment automation                                          |
| `visualization.py` | result visualization                                           |

The model is trained using the feature columns:

* species_frequency
* observation count
* latitude
* longitude
* seasonal encoding features

The `REVIEWED` column is excluded from training to prevent information leakage.

---

# 5. Experimental Setup

Experiments evaluate the effect of Isolation Forest hyperparameters on anomaly detection performance and computational efficiency.

The following parameters are varied:

### Number of Trees

<pre class="overflow-visible! px-0!" data-start="6019" data-end="6043"><div class="relative w-full mt-4 mb-1"><div class=""><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback"><div class="pointer-events-none absolute end-1.5 top-1 z-2 md:end-2 md:top-1"></div><div class="pe-11 pt-3"><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼd ͼr"><div class="cm-scroller"><div class="cm-content q9tKkq_readonly"><span>25, 50, 100, 200</span></div></div></div></div></div></div></div></div></div><div class=""><div class=""></div></div></div></div></div></pre>

### Subsample Size (ψ)

<pre class="overflow-visible! px-0!" data-start="6069" data-end="6129"><div class="relative w-full mt-4 mb-1"><div class=""><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback"><div class="pointer-events-none absolute end-1.5 top-1 z-2 md:end-2 md:top-1"></div><div class="pe-11 pt-3"><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼd ͼr"><div class="cm-scroller"><div class="cm-content q9tKkq_readonly"><span>2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096</span></div></div></div></div></div></div></div></div></div><div class=""><div class=""></div></div></div></div></div></pre>

---

# 6. Evaluation Metrics

The following metrics are measured:

### ROC AUC

ROC AUC evaluates how well anomaly scores separate reviewed observations from normal observations.

### Runtime Metrics

* training time
* scoring time
* total runtime

These metrics measure the scalability of the algorithm.

---

# 7. Results

The experimental results include:

### Detection Performance

* ROC AUC vs subsample size
* ROC AUC vs number of trees

### Computational Efficiency

* runtime vs subsample size
* runtime vs number of trees

### Anomaly Analysis

* histogram of anomaly scores
* geographic map of top anomalous observations
* ranked list of highest anomaly scores
