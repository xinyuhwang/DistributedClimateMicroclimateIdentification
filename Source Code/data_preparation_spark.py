"""
Data Preparation for Distributed K-Means Clustering
Converts preprocessed data to Parquet format for Spark/HDFS
Fixed version - avoids PyArrow extension type issues
"""

import pandas as pd
import numpy as np
import pickle
import os

print("="*80)
print("PREPARING DATA FOR DISTRIBUTED CLUSTERING")
print("="*80)

# =============================================================================
# STEP 1: LOAD PREPROCESSED DATA
# =============================================================================

print("\nSTEP 1: Loading preprocessed data...")

# Load PCA features (use full PCA file if available)
print("  Loading PCA features...")
if os.path.exists("preprocessed_data/pca_features_full.csv"):
    df_pca = pd.read_csv("preprocessed_data/pca_features_full.csv")
    print(f"  ✓ Loaded FULL PCA features from pca_features_full.csv")
else:
    df_pca = pd.read_csv("preprocessed_data/clustering_results_495locs.csv")
    print(f"  ⚠ Using clustering_results file (may have fewer PCA components)")
print(f"  ✓ Loaded {len(df_pca)} locations with PCA coordinates")

# Load engineered features (for reference)
print("  Loading engineered features...")
df_features = pd.read_csv("preprocessed_data/engineered_features_495locs.csv")
print(f"  ✓ Loaded {len(df_features)} locations with {len(df_features.columns)} features")

# Load scaler and PCA model
with open("preprocessed_data/scaler_495locs.pkl", "rb") as f:
    scaler = pickle.load(f)
print("  ✓ Loaded scaler")

with open("preprocessed_data/pca_model_495locs.pkl", "rb") as f:
    pca = pickle.load(f)
print(f"  ✓ Loaded PCA model ({pca.n_components_} components)")

# =============================================================================
# STEP 2: PREPARE DATA FOR SPARK (AVOID NESTED ARRAYS)
# =============================================================================

print("\nSTEP 2: Preparing data for Spark...")

# Extract PCA features as separate columns (not nested arrays)
pca_cols = [col for col in df_pca.columns if col.startswith('PC')]
print(f"  Found {len(pca_cols)} PCA columns: {pca_cols}")

# Create DataFrame with flat structure (each PC as separate column)
df_spark = pd.DataFrame({
    'lat': df_pca['lat'].astype('float64'),
    'lon': df_pca['lon'].astype('float64')
})

# Add each PCA component as separate column
for pc_col in pca_cols:
    df_spark[pc_col] = df_pca[pc_col].astype('float64')

print(f"  ✓ Created {len(df_spark)} records with {len(pca_cols)} PCA features")
print(f"  Columns: {df_spark.columns.tolist()}")

# Verify data types
print(f"\n  Data types:")
print(df_spark.dtypes)

# Check for any NaN values
if df_spark.isnull().any().any():
    print(f"\n  ⚠ Warning: Found NaN values, filling with 0")
    df_spark = df_spark.fillna(0.0)

# =============================================================================
# STEP 3: CONVERT TO PARQUET (SIMPLE FORMAT)
# =============================================================================

print("\nSTEP 3: Converting to Parquet format...")

# Create output directory
output_dir = "spark_data"
os.makedirs(output_dir, exist_ok=True)

# Skip Parquet due to PyArrow compatibility issues
# Use CSV format instead (fully compatible with Spark)
output_file = os.path.join(output_dir, "pca_features.csv")

print(f"  Using CSV format for maximum compatibility...")
df_spark.to_csv(output_file, index=False)
file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
print(f"  ✓ Saved to: {output_file}")
print(f"  File size: {file_size_mb:.2f} MB")

# =============================================================================
# STEP 4: SKIP (CSV already created in Step 3)
# =============================================================================

# =============================================================================
# STEP 5: CREATE SAMPLE DATASETS FOR SCALABILITY TESTING
# =============================================================================

print("\nSTEP 4: Creating sample datasets for scalability testing...")

sample_sizes = [0.1, 0.25, 0.5]
for fraction in sample_sizes:
    sample_df = df_spark.sample(frac=fraction, random_state=42)
    
    # CSV format only
    sample_csv = os.path.join(output_dir, f"pca_features_{int(fraction*100)}pct.csv")
    sample_df.to_csv(sample_csv, index=False)
    
    print(f"  ✓ Created {int(fraction*100)}% sample: {len(sample_df)} records")

# =============================================================================
# STEP 6: SAVE METADATA
# =============================================================================

print("\nSTEP 5: Saving metadata...")

metadata = {
    'n_locations': int(len(df_spark)),
    'n_features': int(len(pca_cols)),
    'pca_variance_retained': float(pca.explained_variance_ratio_.sum()),
    'original_features': int(len(df_features.columns) - 3),  # Exclude lat, lon, cluster
    'pca_components': int(pca.n_components_),
    'feature_columns': pca_cols,
    'lat_range': [float(df_spark['lat'].min()), float(df_spark['lat'].max())],
    'lon_range': [float(df_spark['lon'].min()), float(df_spark['lon'].max())]
}

import json
metadata_file = os.path.join(output_dir, "metadata.json")
with open(metadata_file, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"  ✓ Saved metadata to: {metadata_file}")

# =============================================================================
# STEP 7: VERIFY FILES CAN BE READ
# =============================================================================

print("\nSTEP 6: Verifying files...")

try:
    # Test reading CSV
    df_test_csv = pd.read_csv(output_file)
    print(f"  ✓ Successfully read CSV: {len(df_test_csv)} rows, {len(df_test_csv.columns)} columns")
    print(f"    Columns: {df_test_csv.columns.tolist()}")
except Exception as e:
    print(f"  ✗ Error reading CSV: {e}")

# =============================================================================
# STEP 8: CREATE FEATURE ARRAY FILE FOR SPARK
# =============================================================================

print("\nSTEP 7: Creating feature array file for Spark...")

# Create a version with features as array column (for Spark MLlib compatibility)
df_spark_array = df_spark.copy()

# Convert PCA columns to list of lists
feature_arrays = df_spark[pca_cols].values.tolist()

# Create new dataframe with array column
df_spark_with_array = pd.DataFrame({
    'lat': df_spark['lat'],
    'lon': df_spark['lon'],
    'features_str': [','.join(map(str, row)) for row in feature_arrays]  # String representation
})

# Save as CSV (easy to parse in Spark)
array_csv = os.path.join(output_dir, "pca_features_array.csv")
df_spark_with_array.to_csv(array_csv, index=False)
print(f"  ✓ Saved array format to: {array_csv}")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "="*80)
print("DATA PREPARATION COMPLETE!")
print("="*80)

print(f"\n�� Dataset Summary:")
print(f"  • Locations: {metadata['n_locations']}")
print(f"  • PCA features: {metadata['n_features']}")
print(f"  • Variance retained: {metadata['pca_variance_retained']:.4f}")
print(f"  • Original features: {metadata['original_features']}")

print(f"\n📁 Output Files:")
print(f"  Main dataset:")
print(f"    • {output_file} (CSV format)")
print(f"    • {array_csv} (CSV with feature arrays)")
print(f"  ")
print(f"  Sample datasets:")
for fraction in sample_sizes:
    print(f"    • spark_data/pca_features_{int(fraction*100)}pct.csv")
print(f"  ")
print(f"  Metadata:")
print(f"    • {metadata_file}")

print(f"\n🚀 Next Steps:")
print(f"  1. Copy to HDFS:")
print(f"     hdfs dfs -mkdir -p /user/climate/")
print(f"     hdfs dfs -put {output_file} /user/climate/")
print(f"  ")
print(f"  2. Run distributed K-means:")
print(f"     spark-submit --master yarn distributed_kmeans.py")
print(f"  ")
print(f"  3. Monitor progress:")
print(f"     Spark UI: http://<master-node>:4040")

print("\n" + "="*80)

# Print sample of data
print("\n📋 Data Sample (first 5 rows):")
print(df_spark.head())
print("\n" + "="*80)
