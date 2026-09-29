#!/usr/bin/env python3
"""
Scaling benchmark: distributed map/reduce vs driver-side aggregation.

Measures where distributing K-means starts to pay off, and how it scales with
cores. Both modes import their geometry from distributed_kmeans_mapreduce, so
the inner loop is provably identical and the only variable is *where* the
aggregation happens:

  distributed   mapPartitions -> reduceByKey across executors
  driver        collect() once, then a single-threaded Python loop
                (what the original distributed_kmeans.py does)

scikit-learn is timed alongside as a reference line. It is vectorised C rather
than pure Python, so it is not a like-for-like comparison -- it is there to
show the absolute cost of the no-NumPy constraint, and to answer the question
"is Spark warranted at this size at all?"

Controls
--------
* Both modes start from identical centroids (fixed seed, plain random).
  k-means|| is deliberately not used: it runs Spark jobs, which would
  contaminate the driver-side path.
* A fixed iteration count with early exit disabled, so every configuration
  performs exactly the same work.
* The first iteration is discarded (JVM codegen, JIT warm-up, cache
  materialisation all land there).
* Only the iteration loop is timed -- not Spark start-up, CSV reads, PCA, or
  output writing.
* Each configuration runs several times and the median is reported.

Usage
-----
  python "Source Code/benchmark_scaling.py" --prepare   # build the dataset
  python "Source Code/benchmark_scaling.py"             # run the sweep
"""

import argparse
import json
import os
import random
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from distributed_kmeans_mapreduce import (  # noqa: E402
    accumulate_partition,
    closest_centroid,
    merge_partials,
)

DAILY_CSV = "spark_data/daily_pca_features.csv"
SAMPLE_DIR = "spark_data/benchmark"
SIZES = [495, 5_000, 50_000, 500_000, None]  # None => the full dataset
CORE_SWEEP = [1, 2, 4, 6]
K = 14
ITERATIONS = 10
REPEATS = 3
SEED = 42


# =============================================================================
# DATA PREPARATION
# =============================================================================


def prepare(log):
    """PCA the 1.26M daily records down to the same 14 dimensions.

    The location-level pipeline collapses 1,265,715 daily observations into 495
    rows before clustering, which is exactly why the Spark job is too small to
    benefit from distribution. Clustering the daily records instead keeps the
    data at full size.

    Note this answers a different question from the main analysis: "what daily
    weather regimes occur?" rather than "which locations share a climate?".
    """
    import numpy as np
    import pandas as pd
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    source = "nasa_power_data/combined_climate_data_dense.csv"
    log("reading {} ...".format(source))
    header = pd.read_csv(source, nrows=0).columns.tolist()
    params = [c for c in header if c not in ("time", "lat", "lon")]
    log("  {} parameters".format(len(params)))

    df = pd.read_csv(source, usecols=["lat", "lon"] + params)
    log("  {:,} rows loaded".format(len(df)))

    X = df[params].to_numpy(dtype="float64")
    log("standardising and fitting PCA ...")
    Xz = StandardScaler().fit_transform(X)
    del X

    pca = PCA(n_components=14, random_state=SEED)
    Xp = pca.fit_transform(Xz)
    del Xz
    log("  variance retained: {:.4f}".format(pca.explained_variance_ratio_.sum()))

    out = pd.DataFrame({"lat": df["lat"].to_numpy(), "lon": df["lon"].to_numpy()})
    for i in range(Xp.shape[1]):
        out["PC{}".format(i + 1)] = Xp[:, i]
    del Xp

    os.makedirs("spark_data", exist_ok=True)
    out.to_csv(DAILY_CSV, index=False)
    log("wrote {} ({:.0f} MB)".format(
        DAILY_CSV, os.path.getsize(DAILY_CSV) / 1e6))

    # Fixed subsamples so both modes and every repeat see identical data.
    os.makedirs(SAMPLE_DIR, exist_ok=True)
    shuffled = out.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    for n in SIZES:
        if n is None:
            continue
        path = os.path.join(SAMPLE_DIR, "n{}.csv".format(n))
        shuffled.head(n).to_csv(path, index=False)
        log("  sample n={:,} -> {}".format(n, path))

    full = os.path.join(SAMPLE_DIR, "full.csv")
    shuffled.to_csv(full, index=False)
    log("  sample n={:,} -> {}".format(len(shuffled), full))


def sample_path(n):
    return os.path.join(SAMPLE_DIR, "full.csv" if n is None
                        else "n{}.csv".format(n))


# =============================================================================
# TIMED LOOPS
# =============================================================================


def initial_centroids(rows, k, seed):
    """Plain random init, identical for every mode at a given size."""
    rng = random.Random(seed)
    picked = rng.sample(range(len(rows)), k)
    return [tuple(rows[i][2]) for i in picked]


def _update(stats, centroids, k):
    """Shared centroid update; keeps an empty cluster where it is."""
    out = []
    for j in range(k):
        entry = stats.get(j)
        if entry is None:
            out.append(centroids[j])
        else:
            vec_sum, count, _ = entry
            out.append(tuple(v / count for v in vec_sum))
    return out


def time_distributed(sc, rdd, centroids, iterations, counter=None):
    """mapPartitions -> reduceByKey, one Spark job per iteration."""
    timings = []
    for i in range(iterations + 1):  # +1 warm-up, discarded
        start = time.perf_counter()
        bc = sc.broadcast(centroids)
        try:
            if counter is None:
                stats = (rdd
                         .mapPartitions(lambda rows: accumulate_partition(
                             rows, bc.value))
                         .reduceByKey(merge_partials)
                         .collectAsMap())
            else:
                def counted(rows, _bc=bc, _acc=counter):
                    emitted = accumulate_partition(rows, _bc.value)
                    _acc.add(len(emitted))
                    return emitted

                stats = (rdd.mapPartitions(counted)
                         .reduceByKey(merge_partials)
                         .collectAsMap())
        finally:
            bc.destroy()
        centroids = _update(stats, centroids, len(centroids))
        timings.append(time.perf_counter() - start)
    return timings[1:], centroids


def time_driver(rows, centroids, iterations):
    """collect()-then-loop: the original job's approach, single-threaded."""
    timings = []
    k = len(centroids)
    for i in range(iterations + 1):
        start = time.perf_counter()
        stats = {}
        for row in rows:
            point = row[2]
            cluster, dist = closest_centroid(point, centroids)
            entry = stats.get(cluster)
            if entry is None:
                stats[cluster] = [list(point), 1, dist]
            else:
                running = entry[0]
                for idx, value in enumerate(point):
                    running[idx] += value
                entry[1] += 1
                entry[2] += dist
        stats = {c: (tuple(v[0]), v[1], v[2]) for c, v in stats.items()}
        centroids = _update(stats, centroids, k)
        timings.append(time.perf_counter() - start)
    return timings[1:], centroids


def time_sklearn(rows, centroids, iterations):
    """Reference line: vectorised C, same init, same iteration count."""
    import numpy as np
    from sklearn.cluster import KMeans

    X = np.asarray([r[2] for r in rows], dtype="float64")
    init = np.asarray(centroids, dtype="float64")
    timings = []
    for i in range(min(iterations, 3) + 1):
        start = time.perf_counter()
        KMeans(n_clusters=len(centroids), init=init, n_init=1,
               max_iter=iterations, tol=0.0).fit(X)
        timings.append((time.perf_counter() - start) / iterations)
    return timings[1:]


# =============================================================================
# SWEEPS
# =============================================================================


def ship_module(spark):
    """Send the geometry module to the executors.

    sys.path on the driver does not reach the workers, and cloudpickle
    serialises module-level functions by reference, so without this every
    task fails with ModuleNotFoundError.
    """
    spark.sparkContext.addPyFile(
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "distributed_kmeans_mapreduce.py"))


def load_rows(spark, path, partitions):
    df = spark.read.csv(path, header=True, inferSchema=True)
    pcs = [c for c in df.columns if c.startswith("PC")]
    d = len(pcs)
    rdd = df.select("lat", "lon", *pcs).rdd.map(
        lambda r: (float(r[0]), float(r[1]),
                   tuple(float(r[i]) for i in range(2, 2 + d))))
    if partitions:
        rdd = rdd.repartition(partitions)
    rdd = rdd.persist()
    rdd.count()  # materialise before timing
    return rdd


def size_sweep(spark, cores, log):
    sc = spark.sparkContext
    results = []
    for n in SIZES:
        path = sample_path(n)
        rdd = load_rows(spark, path, cores)
        rows = rdd.collect()
        label = len(rows)
        centroids = initial_centroids(rows, K, SEED)

        counter = sc.accumulator(0)
        dist_runs, seq_runs = [], []
        for _ in range(REPEATS):
            t, _ = time_distributed(sc, rdd, centroids, ITERATIONS,
                                    counter if not dist_runs else None)
            dist_runs.append(statistics.median(t))
            t, _ = time_driver(rows, centroids, ITERATIONS)
            seq_runs.append(statistics.median(t))

        sk = statistics.median(time_sklearn(rows, centroids, ITERATIONS))

        dist = statistics.median(dist_runs)
        seq = statistics.median(seq_runs)
        emitted = counter.value / (ITERATIONS + 1)

        results.append({
            "n": label, "distributed_s": dist, "driver_s": seq,
            "sklearn_s": sk, "speedup": seq / dist if dist else 0.0,
            "partitions": rdd.getNumPartitions(),
            "map_records_emitted": emitted,
        })
        log("  n={:>9,}  distributed {:7.4f}s  driver {:7.4f}s  "
            "sklearn {:7.4f}s  speedup {:5.2f}x".format(
                label, dist, seq, sk, seq / dist if dist else 0.0))
        rdd.unpersist()
    return results


def core_sweep(log):
    from pyspark.sql import SparkSession

    results = []
    path = sample_path(None)
    for cores in CORE_SWEEP:
        spark = (SparkSession.builder.appName("bench-c{}".format(cores))
                 .master("local[{}]".format(cores))
                 .config("spark.ui.enabled", "false")
                 .config("spark.driver.memory", "8g")
                 .getOrCreate())
        spark.sparkContext.setLogLevel("ERROR")
        ship_module(spark)
        try:
            rdd = load_rows(spark, path, cores)
            rows = rdd.collect()
            centroids = initial_centroids(rows, K, SEED)
            runs = []
            for _ in range(REPEATS):
                t, _ = time_distributed(spark.sparkContext, rdd, centroids,
                                        ITERATIONS)
                runs.append(statistics.median(t))
            per_iter = statistics.median(runs)
            results.append({"cores": cores, "seconds": per_iter})
            log("  cores={}  {:7.4f}s/iteration".format(cores, per_iter))
            rdd.unpersist()
        finally:
            spark.stop()
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prepare", action="store_true",
                    help="build the daily PCA dataset and subsamples")
    ap.add_argument("--cores", type=int, default=6,
                    help="cores for the size sweep (default 6: the M2 Pro's "
                         "performance cores, avoiding the slower efficiency "
                         "cores)")
    ap.add_argument("--out", default="benchmark_results.json")
    args = ap.parse_args()

    def log(msg):
        print(msg)
        sys.stdout.flush()

    if args.prepare or not os.path.exists(sample_path(None)):
        log("=" * 62)
        log("PREPARING BENCHMARK DATA")
        log("=" * 62)
        prepare(log)
        if args.prepare:
            return

    from pyspark.sql import SparkSession

    log("\n" + "=" * 62)
    log("SIZE SWEEP  (K={}, {} iterations, {} repeats, local[{}])".format(
        K, ITERATIONS, REPEATS, args.cores))
    log("=" * 62)

    spark = (SparkSession.builder.appName("bench-size")
             .master("local[{}]".format(args.cores))
             .config("spark.ui.enabled", "false")
             .config("spark.driver.memory", "8g")
             .getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    ship_module(spark)
    try:
        sizes = size_sweep(spark, args.cores, log)
    finally:
        spark.stop()

    log("\n" + "=" * 62)
    log("CORE SWEEP  (full dataset)")
    log("=" * 62)
    cores = core_sweep(log)

    payload = {"config": {"k": K, "iterations": ITERATIONS,
                          "repeats": REPEATS, "cores": args.cores},
               "size_sweep": sizes, "core_sweep": cores}
    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=2)
    log("\nwrote {}".format(args.out))


if __name__ == "__main__":
    main()
