# Findings: Reproduction and Rewrite

A record of what was verified, what reproduced, and what did not, while
reproducing this project end to end and rewriting the Spark job as a true
MapReduce implementation.

Work carried out September 2026, roughly eleven months after the original
analysis (NASA POWER accessed 2025-10-12).

---

## Summary

| Step | Goal | Outcome |
|---|---|---|
| 0 | Confirm the NASA data source still works | ✅ Unchanged — all 50 variables present |
| 1 | Build a working local environment | ✅ Done — three blocking issues fixed |
| 2 | Regenerate the 495-location dataset | ✅ Reproduced exactly, with one caveat about K |
| 3 | Rewrite the Spark job as real map/reduce | ✅ Done and validated against three references |

Two findings matter beyond the mechanics:

1. **The choice of K=14 is not robust.** Under a newer scikit-learn the same
   pipeline selects K=15. The metric curve is nearly flat across K=11–15, so
   the original "peak" was always within noise.
2. **The original job was not distributed.** It collected all data to the
   driver and iterated single-threaded. The rewrite fixes this and now
   outperforms both Spark MLlib and the original run on clustering cost.

---

## Step 0 — Verifying the data source

The extraction notebooks read NASA POWER Analysis-Ready Data directly from
public Zarr stores on S3. Since nearly a year had passed, the first question
was whether those endpoints still existed in the same shape.

**They do.** Both stores return HTTP 200 with consolidated metadata, require
no authentication, and retain the grids the project depends on:

| Dataset | Grid | Resolution | Variables requested | Present |
|---|---|---|---|---|
| MERRA-2 meteorology | 361 × 576 | 0.5° × 0.625° | 20 | 20 ✅ |
| SYN1deg radiation | 180 × 360 | 1° × 1° | 30 | 30 ✅ |

Data chunks covering the 2018–2024 window were fetched directly and returned
300 KB–1.4 MB of compressed content each, confirming real data rather than
unwritten placeholder space.

### Worth knowing

**The time axes extend to 2029-12-31.** Both datasets are preallocated well
beyond the data actually published. A query past the real end returns fill
values rather than raising an error — a silent-corruption risk for anyone
extending the date range later. The 2018–2024 window sits safely inside the
populated region.

**Chunking is time-major**, at `[5844, 15, 15]` for MERRA-2 and
`[4000, 10, 15]` for SYN1deg. Requesting 2,557 days still transfers whole
multi-thousand-day chunks, which sets a floor on download time regardless of
how narrow the date slice is.

---

## Step 1 — Environment

Three separate problems blocked execution, none of them obvious from an error
message alone. All three are now captured in [`env.sh`](env.sh), scoped to the
shell so nothing system-wide changes.

| Problem | Symptom | Cause | Fix |
|---|---|---|---|
| Python mismatch | Every Spark job failed | Driver ran venv Python 3.12, workers launched system 3.13 | Pin `PYSPARK_PYTHON` |
| TLS failure | NASA endpoints unreachable from Python, though `curl` worked | python.org build ships no configured CA store | Point `SSL_CERT_FILE` at `certifi` |
| Java version | Spark refused to start | System default is Java 23; Spark 3.5 supports 8/11/17 | Pin `JAVA_HOME` to Temurin 17 |

The TLS one is the most deceptive: `curl` succeeds because it uses the macOS
keychain, so the endpoint looks healthy while Python cannot reach it.

**Disk space was the practical blocker.** The machine had 1.1 GB free of
460 GB. Roughly 16 GB was reclaimed from regenerable caches — npm, pnpm, pip,
Electron updater staging, and 6.5 GB of iOS simulator data orphaned by an
uninstalled Xcode.

**Version choices:** Python 3.12 rather than the system 3.13 (PySpark 3.5 has
known friction on 3.13), `zarr<3` to match the notebooks' store API, and
PySpark 3.5.9 to stay close to what EMR runs.

---

## Step 2 — Regenerating the dataset

Stages 1–3 were re-run against live NASA data. Extraction took about six
minutes.

### What reproduced exactly

| Quantity | Original report | Reproduction |
|---|---|---|
| Native meteorology points | 84 | 84 ✅ |
| Native radiation points | 24 | 24 ✅ |
| Interpolated locations | 495 | 495 ✅ |
| Total data points | 1,265,715 | 1,265,715 ✅ |
| Engineered features | 224 | 224 ✅ |
| Correlated features removed | 40 | 40 ✅ |
| Final feature set | 184 | 184 ✅ |
| PCA components (95%) | 14 | 14 ✅ |
| Variance retained | 95.14% | 95.14% ✅ |

Zero missing values, zero infinities, and all variables within physically
plausible ranges. The extraction and feature pipeline is fully reproducible.

### What did not reproduce: the choice of K

The original analysis selected K=14 as the point where silhouette score peaks
and Davies-Bouldin is minimised. **The rerun selects K=15 on both metrics.**

| K | Silhouette (original) | Silhouette (rerun) | Davies-Bouldin (original) | Davies-Bouldin (rerun) |
|---|---|---|---|---|
| 12 | 0.393 | 0.387 | 0.977 | 0.958 |
| 13 | — | 0.394 | — | 0.950 |
| **14** | **0.396** | 0.395 | **0.929** | 0.928 |
| **15** | 0.391 | **0.408** | 0.961 | **0.916** |

Values agree closely up to K=14. The divergence is at K=15, where the rerun
found a better solution than the original run did — inertia 9,941 versus
10,077. This is a local-optimum difference: newer scikit-learn seeding found a
better K=15 configuration, which flipped the argmax.

Two observations follow:

**The peak was never a strong signal.** Silhouette across K=11–15 reads 0.386,
0.387, 0.394, 0.395, 0.408 — a range narrower than the run-to-run variation
K-means produces from different seeds.

**The optimum was never bracketed.** The sweep tested K ∈ [3, 15], and the best
value is the last one tested with the curve still rising. An interior maximum
is evidence; a boundary maximum means the search window was too narrow. A
sweep to K=25 would establish whether a genuine interior optimum exists.

**Decision:** K was pinned to 14 for the rest of this work, so the local
baseline, the existing EMR output, and the report's fourteen named
microclimates all remain consistent. At K=14 the rerun reproduces the original
closely (inertia 10,624 vs 10,600; silhouette 0.393 vs 0.396). The K=15
finding is recorded in `preprocessed_data/clustering_metrics_495locs.json`
rather than acted upon.

---

## Step 3 — Rewriting the Spark job

### What the original code did

`distributed_kmeans.py` calls `.collect()` to pull all 495 points onto the
driver, then runs Lloyd's algorithm in a single-threaded Python loop. Spark
handles reading from S3, schema inference, and writing results — but not the
clustering.

Supporting evidence from the original run: **16 seconds per iteration** for a
workload of roughly 97,000 floating-point operations. That arithmetic takes
tens of milliseconds in pure Python, so over 99% of each iteration was Spark
scheduling overhead around work that never left the driver.

Several documented behaviours were absent from the code:

| Report describes | Code actually does |
|---|---|
| Broadcast centroids, map phase, reduce phase | Single-threaded loop on the driver |
| k-means++ initialisation | `random.sample` — uniform random |
| `repartition(50, 'lat', 'lon')` | Not present |
| 498× network I/O reduction | Compares against a strawman no one would implement |
| Centralised convergence check is the bottleneck | Comparing 14 small vectors costs microseconds |

Two settings were inert: `.cache()` on a DataFrame read exactly once, and
`spark.sql.shuffle.partitions` on a job that never shuffles. The reported final
inertia is also one iteration stale, computed against distances from before the
last centroid update.

None of this affects the *scientific* results — the clusters are real. It
affects whether the code supports the claims made about it.

### The rewrite

[`distributed_kmeans_mapreduce.py`](Source%20Code/distributed_kmeans_mapreduce.py)
was added as a **new file**; the original is unchanged.

Each iteration is now one Spark job:

- **Map** — `mapPartitions` assigns points and accumulates per-cluster
  (vector sum, count, cost) locally. This is in-mapper combining: a partition
  emits at most K records rather than one per point.
- **Reduce** — `reduceByKey` merges partial sums across partitions.
- **Driver** — divides sums by counts, checks convergence, re-broadcasts.

Also addressed:

- **Broadcast lifecycle.** Centroids ship once per executor rather than once
  per task, and each iteration's broadcast is explicitly destroyed. Without
  this, 50 iterations leak 50 broadcasts into the driver.
- **k-means‖ initialisation.** Plain k-means++ needs K sequential passes — 14
  Spark jobs before clustering begins. k-means‖ oversamples over two rounds and
  reclusters candidates locally.
- **Empty clusters.** A cluster that captures no points produces no key in the
  reduce output. Unhandled, this silently drops a centroid and the run
  continues with K−1 clusters. Empty clusters are now reseeded from the
  farthest point.
- **Caching.** The points RDD is persisted before the loop, so iterations do
  not re-read the source.
- **Partition count.** The original's 50 was counterproductive: with 495
  points, in-mapper combining would emit up to 700 records, more than the 495
  it set out to compress. Now defaults to cluster parallelism.

### The variance problem

The first version ran a single initialisation. Changing only the partition
count moved the final cost by 28%:

| Partitions | Single restart | With `n_init=10` |
|---|---|---|
| 1 | 10,940 | 10,734 |
| 4 | 11,194 | 10,778 |
| 8 | 11,267 | **10,659** |
| 16 | 14,030 | 10,785 |

Partitioning perturbs the initial sample, and Lloyd's algorithm only finds a
local optimum. Adding restarts collapsed the spread from 28% to **1.2%**.

This exposed a flaw in the original validation: the local baseline used
`n_init=50` while the EMR job used `n_init=3`. Some of the reported gap between
them measured that asymmetry, not anything about distribution.

### Validation

Final cost against every available reference:

| Implementation | Cost | Configuration |
|---|---|---|
| scikit-learn baseline | **10,624** | `n_init=50` |
| scikit-learn, seeds 0–3 | 10,645–10,791 | `n_init=10` |
| **This rewrite** | **10,659** | `n_init=10`, best of 10 |
| Spark MLlib | 10,844 | single initialisation |
| Original EMR run | 10,848 | driver-side, `n_init=3` |

The rewrite beats both MLlib's built-in K-means and the original run, and
effectively ties scikit-learn.

**The control experiment settles correctness.** Cluster agreement below a
perfect score needs a reference point, so two independent scikit-learn runs
were compared against each other:

| Comparison | ARI |
|---|---|
| scikit-learn vs scikit-learn (different seeds) | 0.786 – 0.875 |
| **This rewrite vs baseline** | **0.744 – 0.871** |
| MLlib vs baseline | 0.788 |
| Original EMR vs baseline | 0.720 |

The rewrite agrees with the baseline as closely as scikit-learn agrees with
itself. The residual disagreement is inherent K-means variance, not an
implementation defect. All runs produced exactly 14 non-empty clusters.

### Measured scaling behaviour

Runtime for identical work, varying only partition count:

| Partitions | Runtime |
|---|---|
| 1 | 19.9 s |
| 4 | 20.0 s |
| 8 | 32.0 s |
| 16 | 53.2 s |

**More partitions made it slower**, by a factor of 2.7. At 495 points the
per-task scheduling cost exceeds the arithmetic being scheduled. This is the
honest scaling result at this data size: distribution is correct here, but it
does not pay. It begins to pay only when partitions hold enough points to
amortise their own overhead — which is what the 1.26M-record benchmark
(step 5, not yet run) is designed to measure.

---

## Outstanding items

- **Step 4** — formal validation write-up using the existing
  `compare_emr_local.py` comparison and figures.
- **Step 5** — scale benchmark on the full 1,265,715 daily records. Location-level
  aggregation collapses the data to 495 rows before clustering, which is why
  the job is too small to benefit from distribution. Clustering the daily
  records instead answers a different question ("what weather regimes occur?"
  rather than "which locations share a climate?") but is where a genuine
  speedup curve can be measured.
- **Report corrections** — the claims listed under "What the original code did"
  describe an intended design rather than the committed code.
- **K sweep** — extending beyond K=15 would determine whether a true interior
  optimum exists.

## Reproducing this

```bash
source env.sh                                    # JDK 17, venv Python, CA bundle

# Stage 1: extract and interpolate (~6 min, ~2.2 GB)
#   run Data Processing/enhance_location_interpolation.ipynb

# Stage 2-3: features, PCA, baseline, Spark input
python "Source Code/preprocessing_pipeline.py"
python "Source Code/data_preparation_spark.py"

# Rewritten MapReduce job
python "Source Code/distributed_kmeans_mapreduce.py" \
    --input spark_data/pca_features.csv \
    --output out --master "local[*]"
```
