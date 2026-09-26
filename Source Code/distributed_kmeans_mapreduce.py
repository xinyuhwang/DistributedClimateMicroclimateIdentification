#!/usr/bin/env python3
"""
Distributed K-Means Clustering on Spark (MapReduce formulation)

This is a true map/reduce implementation of the algorithm in
distributed_kmeans.py. The original collects the dataset to the driver and
iterates in a single-threaded Python loop; here every iteration is one Spark
job spread across the executors:

  MAP     mapPartitions  Assign each point to its nearest centroid and
                         accumulate per-cluster (vector sum, count, cost)
                         locally. This is in-mapper combining: a partition
                         emits at most K records instead of one per point.

  REDUCE  reduceByKey    Merge the partial sums across partitions.

  DRIVER                 Divide sums by counts to get the new centroids,
                         test convergence, re-broadcast.

Centroids travel to the executors in a broadcast variable (one copy per
executor rather than one per task) and each iteration's broadcast is
explicitly destroyed to keep the driver's block manager from growing.

Initialisation uses k-means|| (scalable k-means++). Plain k-means++ needs K
sequential passes over the data -- 14 Spark jobs before clustering starts --
whereas k-means|| oversamples over a small fixed number of rounds and then
reclusters the candidates locally.

Written in pure Python (no NumPy) so it runs on a stock EMR cluster with no
bootstrap action or custom Python environment, matching the constraint the
original was built under.

Usage
-----
  # Local, against the prepared dataset
  spark-submit distributed_kmeans_mapreduce.py \
      --input spark_data/pca_features.csv --output out --master "local[*]"

  # On EMR (defaults point at S3)
  spark-submit --deploy-mode cluster \
      s3://<bucket>/distributed_kmeans_mapreduce.py
"""

import argparse
import math
import random
import sys
import time

# =============================================================================
# CONFIGURATION
# =============================================================================


class Config:
    K = 14
    MAX_ITERATIONS = 50
    CONVERGENCE_THRESHOLD = 1e-4
    SEED = 42

    # Restarts. Lloyd's algorithm only finds a local optimum, so a single run
    # is high-variance: on this dataset, changing nothing but the partition
    # count (which perturbs the initial sample) moved the final cost between
    # 10,940 and 14,030. scikit-learn defaults to n_init=10 and the local
    # baseline in preprocessing_pipeline.py uses 50. Each restart is
    # independent, so the best-of-N is what should be compared against it.
    N_INIT = 10

    # k-means|| initialisation. Spark MLlib defaults to 2 rounds; each round
    # costs two Spark jobs (one to total the cost, one to sample), so this is
    # the main knob trading initialisation quality against startup latency.
    INIT_ROUNDS = 2
    INIT_OVERSAMPLE = 2.0  # candidates drawn per round = OVERSAMPLE * K

    S3_BUCKET = "mydatahwbucket"
    INPUT_DATA = "s3://{}/pca_features.csv".format(S3_BUCKET)
    OUTPUT_PATH = "s3://{}/output".format(S3_BUCKET)

    # None => fall back to the cluster's default parallelism. The original's
    # value of 50 is counterproductive at this scale: with 495 points,
    # in-mapper combining would emit up to 50*14 = 700 records, more than the
    # 495 it set out to compress.
    NUM_PARTITIONS = None


# =============================================================================
# PURE-PYTHON GEOMETRY
#
# Rows are (lat, lon, features); features is a tuple of floats.
# =============================================================================


def _vec(row):
    return row[2]


def squared_distance(p, q):
    total = 0.0
    for a, b in zip(p, q):
        diff = a - b
        total += diff * diff
    return total


def closest_centroid(point, centroids):
    """Return (index, squared_distance) of the nearest centroid.

    Squared distance is enough to rank candidates, so the square root is
    skipped. The inner loop abandons a centroid as soon as its running total
    exceeds the best seen so far.
    """
    best_index = 0
    best_dist = float("inf")
    for i, centroid in enumerate(centroids):
        total = 0.0
        for a, b in zip(point, centroid):
            diff = a - b
            total += diff * diff
            if total >= best_dist:
                break
        if total < best_dist:
            best_dist = total
            best_index = i
    return best_index, best_dist


# =============================================================================
# MAP AND REDUCE
# =============================================================================


def accumulate_partition(rows, centroids):
    """MAP phase with in-mapper combining.

    Emits (cluster_id, (vector_sum, count, cost)) -- at most K records per
    partition, regardless of how many points the partition holds.
    """
    partials = {}
    for row in rows:
        point = _vec(row)
        cluster, dist = closest_centroid(point, centroids)
        entry = partials.get(cluster)
        if entry is None:
            partials[cluster] = [list(point), 1, dist]
        else:
            running = entry[0]
            for i, value in enumerate(point):
                running[i] += value
            entry[1] += 1
            entry[2] += dist
    return [(k, (tuple(v[0]), v[1], v[2])) for k, v in partials.items()]


def merge_partials(a, b):
    """REDUCE phase: combine two partial (sum, count, cost) triples."""
    sum_a, count_a, cost_a = a
    sum_b, count_b, cost_b = b
    return (
        tuple(x + y for x, y in zip(sum_a, sum_b)),
        count_a + count_b,
        cost_a + cost_b,
    )


# =============================================================================
# K-MEANS|| INITIALISATION
# =============================================================================


def _sample_candidates(rows, centers, factor, seed):
    """Keep each point with probability proportional to its squared distance."""
    rng = random.Random(seed)
    chosen = []
    for row in rows:
        _, dist = closest_centroid(_vec(row), centers)
        if dist > 0.0 and rng.random() < factor * dist:
            chosen.append(_vec(row))
    return chosen


def _weighted_local_kmeans(candidates, weights, k, seed, max_iter=30):
    """Recluster the k-means|| candidates down to exactly k centroids.

    Weighted k-means++ seeding followed by weighted Lloyd iterations. Runs on
    the driver: len(candidates) is roughly INIT_ROUNDS * INIT_OVERSAMPLE * K,
    so a few dozen points at most.
    """
    rng = random.Random(seed)

    # --- weighted k-means++ seeding ---
    total_weight = sum(weights)
    centers = [candidates[0]]
    if total_weight > 0.0:
        target = rng.random() * total_weight
        cumulative = 0.0
        for cand, w in zip(candidates, weights):
            cumulative += w
            if cumulative >= target:
                centers = [cand]
                break

    while len(centers) < k:
        costs = [weights[i] * closest_centroid(c, centers)[1]
                 for i, c in enumerate(candidates)]
        total = sum(costs)
        if total <= 0.0:
            # Remaining candidates coincide with chosen centres; pad.
            centers.append(candidates[rng.randrange(len(candidates))])
            continue
        target = rng.random() * total
        cumulative = 0.0
        pick = candidates[-1]
        for cand, cost in zip(candidates, costs):
            cumulative += cost
            if cumulative >= target:
                pick = cand
                break
        centers.append(pick)

    # --- weighted Lloyd refinement ---
    dims = len(candidates[0])
    for _ in range(max_iter):
        sums = [[0.0] * dims for _ in range(k)]
        counts = [0.0] * k
        for cand, w in zip(candidates, weights):
            idx, _ = closest_centroid(cand, centers)
            row = sums[idx]
            for i, value in enumerate(cand):
                row[i] += value * w
            counts[idx] += w

        moved = False
        for j in range(k):
            if counts[j] == 0.0:
                continue
            updated = tuple(v / counts[j] for v in sums[j])
            if squared_distance(updated, centers[j]) > 0.0:
                moved = True
            centers[j] = updated
        if not moved:
            break

    return [tuple(c) for c in centers]


def kmeans_parallel_init(sc, points, k, seed, log):
    """k-means|| (scalable k-means++)."""
    log("  k-means|| initialisation ({} rounds, oversample {}x, seed {})".format(
        Config.INIT_ROUNDS, Config.INIT_OVERSAMPLE, seed))

    first = points.takeSample(False, 1, seed=seed)[0]
    centers = [tuple(_vec(first))]

    for round_index in range(Config.INIT_ROUNDS):
        bc = sc.broadcast(centers)
        try:
            total_cost = points.map(
                lambda row: closest_centroid(_vec(row), bc.value)[1]
            ).sum()

            if total_cost <= 0.0:
                log("    round {}: cost is zero, stopping early".format(
                    round_index + 1))
                break

            factor = Config.INIT_OVERSAMPLE * k / total_cost
            round_seed = seed + round_index * 977
            new_centers = points.mapPartitionsWithIndex(
                lambda idx, rows: _sample_candidates(
                    rows, bc.value, factor, round_seed + idx)
            ).collect()
        finally:
            bc.destroy()

        centers.extend(tuple(c) for c in new_centers)
        log("    round {}: cost={:.2f}, {} candidates".format(
            round_index + 1, total_cost, len(centers)))

    if len(centers) <= k:
        log("    only {} candidates; using them directly".format(len(centers)))
        while len(centers) < k:
            centers.append(centers[-1])
        return centers[:k]

    # Weight each candidate by how many points it captures, then recluster.
    bc = sc.broadcast(centers)
    try:
        counts = (points
                  .map(lambda row: (closest_centroid(_vec(row), bc.value)[0], 1))
                  .reduceByKey(lambda a, b: a + b)
                  .collectAsMap())
    finally:
        bc.destroy()

    weights = [float(counts.get(i, 0)) for i in range(len(centers))]
    log("    reclustering {} candidates -> {} centroids".format(
        len(centers), k))
    return _weighted_local_kmeans(centers, weights, k, seed)


# =============================================================================
# EMPTY-CLUSTER RECOVERY
# =============================================================================


def farthest_point(points, centroids):
    """Return the point with the greatest squared distance to its centroid.

    Only called when a cluster loses every member, which is rare, so this pays
    for an extra Spark job rather than burdening the per-iteration shuffle.
    """
    def keyed(row):
        return (closest_centroid(_vec(row), centroids)[1], _vec(row))

    return points.map(keyed).max(key=lambda t: t[0])[1]


# =============================================================================
# LLOYD ITERATIONS
# =============================================================================


def lloyd(sc, points, k, centroids, log):
    """Run Lloyd's algorithm to convergence from the given starting centroids.

    Returns (centroids, cost, iterations, converged).
    """
    cost = float("nan")
    iteration = 0
    converged = False

    for iteration in range(1, Config.MAX_ITERATIONS + 1):
        bc = sc.broadcast(centroids)
        try:
            stats = (points
                     .mapPartitions(lambda rows_: accumulate_partition(
                         rows_, bc.value))
                     .reduceByKey(merge_partials)
                     .collectAsMap())
        finally:
            # Without this the driver accumulates one broadcast per iteration.
            bc.destroy()

        cost = sum(v[2] for v in stats.values())

        updated = []
        empty = []
        for j in range(k):
            entry = stats.get(j)
            if entry is None:
                empty.append(j)
                updated.append(centroids[j])  # placeholder, replaced below
            else:
                vec_sum, count, _ = entry
                updated.append(tuple(v / count for v in vec_sum))

        # A cluster that captured no points produces no key in the reduce
        # output. Left unhandled this silently drops a centroid and the run
        # continues with k-1 clusters.
        for j in empty:
            replacement = farthest_point(points, updated)
            log("      iter {}: cluster {} empty, reseeding".format(
                iteration, j))
            updated[j] = tuple(replacement)

        shift = max(math.sqrt(squared_distance(o, u))
                    for o, u in zip(centroids, updated))
        centroids = updated

        if shift < Config.CONVERGENCE_THRESHOLD:
            converged = True
            break

    return centroids, cost, iteration, converged


# =============================================================================
# MAIN
# =============================================================================


def run(spark, args, log):
    sc = spark.sparkContext
    started = time.time()

    log("\n[1/4] Loading {}".format(args.input))
    df = spark.read.csv(args.input, header=True, inferSchema=True)
    pca_cols = [c for c in df.columns if c.startswith("PC")]
    if not pca_cols:
        raise SystemExit("no PC* feature columns found in {}".format(args.input))

    n_features = len(pca_cols)
    rows = df.select("lat", "lon", *pca_cols).rdd.map(
        lambda r: (float(r[0]), float(r[1]),
                   tuple(float(r[i]) for i in range(2, 2 + n_features)))
    )

    partitions = args.partitions or Config.NUM_PARTITIONS
    if partitions:
        rows = rows.repartition(partitions)

    # Cache before the loop: without this every iteration re-reads the source.
    points = rows.persist()
    n = points.count()  # materialise the cache
    log("  {} points, {} features, {} partitions".format(
        n, n_features, points.getNumPartitions()))

    log("\n[2/4] Clustering (K={}, {} restarts, max {} iterations)".format(
        args.k, args.n_init, Config.MAX_ITERATIONS))

    best = None
    for attempt in range(args.n_init):
        seed = Config.SEED + attempt * 7919
        log("  restart {}/{}".format(attempt + 1, args.n_init))
        start = kmeans_parallel_init(sc, points, args.k, seed, log)
        centers, cost, iters, converged = lloyd(
            sc, points, args.k, start, log)
        log("    cost={:.2f}  iterations={}{}".format(
            cost, iters, "" if converged else " (not converged)"))

        if best is None or cost < best[1]:
            best = (centers, cost, iters, converged)
            log("    new best")

    centroids, cost, iteration, converged = best
    log("\n[3/4] Best of {} restarts: cost={:.2f} ({} iterations)".format(
        args.n_init, cost, iteration))

    log("\n[4/4] Writing assignments to {}".format(args.output))
    bc = sc.broadcast(centroids)
    try:
        assignments = points.map(
            lambda row: (row[0], row[1],
                         closest_centroid(_vec(row), bc.value)[0])
        )
        result = spark.createDataFrame(assignments, ["lat", "lon", "cluster"])
        result.coalesce(1).write.mode("overwrite").csv(
            args.output + "/cluster_assignments", header=True)

        sizes = (result.groupBy("cluster").count()
                 .orderBy("cluster").collect())
    finally:
        bc.destroy()

    points.unpersist()

    log("\nCluster sizes:")
    for row in sizes:
        log("  cluster {:2d}: {:4d}".format(row["cluster"], row["count"]))

    elapsed = time.time() - started
    log("\n{}".format("=" * 62))
    log("iterations : {}{}".format(iteration,
                                   "" if converged else " (not converged)"))
    log("final cost : {:.2f}".format(cost))
    log("runtime    : {:.2f}s".format(elapsed))
    log("=" * 62)

    return {"iterations": iteration, "cost": cost, "centroids": centroids,
            "converged": converged, "runtime": elapsed}


def parse_args(argv):
    p = argparse.ArgumentParser(description="Distributed K-means on Spark")
    p.add_argument("--input", default=Config.INPUT_DATA)
    p.add_argument("--output", default=Config.OUTPUT_PATH)
    p.add_argument("--k", type=int, default=Config.K)
    p.add_argument("--n-init", type=int, default=Config.N_INIT,
                   help="independent restarts; the lowest-cost run wins")
    p.add_argument("--partitions", type=int, default=None)
    p.add_argument("--master", default=None,
                   help="Spark master; omit on EMR to use the cluster default")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv if argv is not None else sys.argv[1:])

    def log(message):
        print(message)
        sys.stdout.flush()

    from pyspark.sql import SparkSession

    builder = SparkSession.builder.appName("Climate-K-Means")
    if args.master:
        builder = builder.master(args.master)
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    log("=" * 62)
    log("DISTRIBUTED K-MEANS  (Spark {})".format(spark.version))
    log("K={}  n_init={}  max_iter={}  tol={}".format(
        args.k, args.n_init, Config.MAX_ITERATIONS,
        Config.CONVERGENCE_THRESHOLD))
    log("=" * 62)

    try:
        return run(spark, args, log)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
