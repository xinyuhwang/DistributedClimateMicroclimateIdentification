#!/usr/bin/env python3
"""
Distributed K-Means Clustering - Pure Python (No NumPy)
Works on any EMR cluster without additional dependencies
"""

import sys
import time
import random
import math

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    K = 14
    MAX_ITERATIONS = 50
    CONVERGENCE_THRESHOLD = 0.01
    N_INIT = 3
    
    S3_BUCKET = "mydatahwbucket"
    INPUT_DATA = f"s3://{S3_BUCKET}/pca_features.csv"
    OUTPUT_PATH = f"s3://{S3_BUCKET}/output/"
    NUM_PARTITIONS = 50

print("="*80)
print("DISTRIBUTED K-MEANS CLUSTERING")
print("="*80)
print(f"Config: K={Config.K}, Max Iter={Config.MAX_ITERATIONS}")
print(f"Input: {Config.INPUT_DATA}")
print(f"Output: {Config.OUTPUT_PATH}")
print("="*80)
sys.stdout.flush()

# =============================================================================
# PURE PYTHON MATH FUNCTIONS (NO NUMPY)
# =============================================================================

def euclidean_distance(p1, p2):
    """Compute Euclidean distance between two points"""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

def mean_of_points(points):
    """Compute mean of list of points"""
    if not points:
        return None
    n = len(points)
    d = len(points[0])
    return [sum(p[i] for p in points) / n for i in range(d)]

def assign_to_cluster(features, centroids):
    """Assign point to nearest centroid"""
    min_dist = float('inf')
    cluster = 0
    
    for i, centroid in enumerate(centroids):
        dist = euclidean_distance(features, centroid)
        if dist < min_dist:
            min_dist = dist
            cluster = i
    
    return cluster, min_dist

# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    start_time = time.time()
    
    # Import PySpark
    print("\n[1/6] Importing PySpark...")
    sys.stdout.flush()
    try:
        from pyspark.sql import SparkSession
        from pyspark.sql.functions import col, array as spark_array
        print("✓ PySpark imported")
        sys.stdout.flush()
    except ImportError as e:
        print(f"✗ Failed to import PySpark: {e}")
        sys.exit(1)
    
    # Initialize Spark
    print("\n[2/6] Initializing Spark...")
    sys.stdout.flush()
    try:
        spark = SparkSession.builder \
            .appName("Climate-K-Means") \
            .config("spark.sql.shuffle.partitions", str(Config.NUM_PARTITIONS)) \
            .getOrCreate()
        
        print(f"✓ Spark {spark.version} initialized")
        sys.stdout.flush()
    except Exception as e:
        print(f"✗ Failed to initialize Spark: {e}")
        sys.exit(1)
    
    # Load data
    print(f"\n[3/6] Loading data from S3...")
    sys.stdout.flush()
    try:
        data_df = spark.read.csv(Config.INPUT_DATA, header=True, inferSchema=True)
        
        # Get PCA columns
        pca_cols = [c for c in data_df.columns if c.startswith('PC')]
        n_rows = data_df.count()
        
        print(f"✓ Loaded {n_rows} rows with {len(pca_cols)} PCA features")
        sys.stdout.flush()
        
        # Create features array
        data_df = data_df.withColumn('features', spark_array(*pca_cols))
        data_df = data_df.select('lat', 'lon', 'features')
        data_df.cache()
        
    except Exception as e:
        print(f"✗ Failed to load data: {e}")
        import traceback
        traceback.print_exc()
        spark.stop()
        sys.exit(1)
    
    # Collect data
    print(f"\n[4/6] Collecting data for clustering...")
    sys.stdout.flush()
    try:
        data_list = data_df.collect()
        print(f"✓ Collected {len(data_list)} points")
        sys.stdout.flush()
    except Exception as e:
        print(f"✗ Failed to collect data: {e}")
        spark.stop()
        sys.exit(1)
    
    # Extract features as lists
    features_list = [list(row['features']) for row in data_list]
    n_samples = len(features_list)
    n_features = len(features_list[0])
    
    print(f"  Feature matrix: {n_samples} × {n_features}")
    sys.stdout.flush()
    
    # K-means clustering
    print(f"\n[5/6] Running K-means (K={Config.K})...")
    sys.stdout.flush()
    
    best_inertia = float('inf')
    best_centroids = None
    best_labels = None
    
    random.seed(42)
    
    for init_num in range(Config.N_INIT):
        print(f"\n  Init {init_num + 1}/{Config.N_INIT}")
        sys.stdout.flush()
        
        # Random initialization
        indices = random.sample(range(n_samples), Config.K)
        centroids = [features_list[i][:] for i in indices]
        
        # Iterate
        for iteration in range(Config.MAX_ITERATIONS):
            # Assignment step
            assignments = []
            distances = []
            
            for features in features_list:
                cluster, dist = assign_to_cluster(features, centroids)
                assignments.append(cluster)
                distances.append(dist)
            
            # Update step
            new_centroids = []
            for k in range(Config.K):
                cluster_points = [features_list[i] for i in range(n_samples) if assignments[i] == k]
                
                if cluster_points:
                    new_centroids.append(mean_of_points(cluster_points))
                else:
                    # Empty cluster - reinitialize
                    new_centroids.append(features_list[random.randint(0, n_samples-1)][:])
            
            # Check convergence
            max_shift = max(euclidean_distance(old, new) for old, new in zip(centroids, new_centroids))
            centroids = new_centroids
            
            if iteration % 10 == 0:
                inertia = sum(d**2 for d in distances)
                print(f"    Iter {iteration:3d}: Inertia={inertia:12.2f}, Shift={max_shift:.6f}")
                sys.stdout.flush()
            
            if max_shift < Config.CONVERGENCE_THRESHOLD:
                print(f"    ✓ Converged at iteration {iteration}")
                sys.stdout.flush()
                break
        
        # Final inertia
        final_inertia = sum(d**2 for d in distances)
        print(f"    Final inertia: {final_inertia:.2f}")
        sys.stdout.flush()
        
        if final_inertia < best_inertia:
            best_inertia = final_inertia
            best_centroids = centroids
            best_labels = assignments
            print(f"    ✓ New best!")
            sys.stdout.flush()
    
    print(f"\n✓ K-means complete! Best inertia: {best_inertia:.2f}")
    sys.stdout.flush()
    
    # Create results
    print(f"\n[6/6] Saving results...")
    sys.stdout.flush()
    
    try:
        results_data = []
        for i, row in enumerate(data_list):
            results_data.append({
                'lat': float(row['lat']),
                'lon': float(row['lon']),
                'cluster': int(best_labels[i])
            })
        
        results_df = spark.createDataFrame(results_data)
        
        # Show distribution
        print("\nCluster distribution:")
        sys.stdout.flush()
        cluster_counts = results_df.groupBy('cluster').count().orderBy('cluster').collect()
        for row in cluster_counts:
            print(f"  Cluster {row['cluster']}: {row['count']} locations")
            sys.stdout.flush()
        
        # Save to S3
        output_path = f"{Config.OUTPUT_PATH}/cluster_assignments"
        results_df.coalesce(1).write.mode('overwrite').csv(output_path, header=True)
        
        print(f"✓ Results saved to: {output_path}")
        sys.stdout.flush()
        
    except Exception as e:
        print(f"✗ Failed to save: {e}")
        import traceback
        traceback.print_exc()
        spark.stop()
        sys.exit(1)
    
    # Done
    elapsed = time.time() - start_time
    print("\n" + "="*80)
    print("CLUSTERING COMPLETE!")
    print("="*80)
    print(f"Runtime: {elapsed:.2f} seconds")
    print(f"Results: {Config.OUTPUT_PATH}/cluster_assignments")
    print("="*80)
    sys.stdout.flush()
    
    spark.stop()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
