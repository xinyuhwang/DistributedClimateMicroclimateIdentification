"""
Updated Data Preprocessing Pipeline for High-Density Climate Data
Processes 495 locations with 1.26M data points
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import warnings
import os

warnings.filterwarnings('ignore')

print("="*80)
print("CLIMATE MICROCLIMATE CLUSTERING - PREPROCESSING PIPELINE")
print("="*80)

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("\nSTEP 1: LOADING DATA")
print("-"*80)

input_file = "nasa_power_data/combined_climate_data_dense.csv"
print(f"Loading: {input_file}")
print("This may take 30-60 seconds for 1.26M rows...")

df = pd.read_csv(input_file)

print(f"\n✓ Loaded successfully!")
print(f"  Total rows: {len(df):,}")
print(f"  Unique locations: {df[['lat', 'lon']].drop_duplicates().shape[0]:,}")
print(f"  Date range: {df['time'].min()} to {df['time'].max()}")
print(f"  Total columns: {len(df.columns)}")

# Identify numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
# Remove lat, lon from features (keep as identifiers)
feature_cols = [col for col in numeric_cols if col not in ['lat', 'lon']]

print(f"  Numeric parameters: {len(feature_cols)}")

# ============================================================================
# STEP 2: DATA QUALITY CHECK
# ============================================================================
print("\n" + "="*80)
print("STEP 2: DATA QUALITY ASSESSMENT")
print("-"*80)

# Check for missing values
missing = df[feature_cols].isnull().sum()
if missing.sum() == 0:
    print("✓ No missing values found!")
else:
    print(f"⚠ Found {missing.sum()} missing values:")
    print(missing[missing > 0])

# Check for infinite values
inf_count = np.isinf(df[feature_cols]).sum().sum()
if inf_count > 0:
    print(f"⚠ Found {inf_count} infinite values - replacing with NaN")
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    df = df.dropna()

# Statistical summary
print("\nStatistical Summary (first 10 parameters):")
print(df[feature_cols[:10]].describe().T[['mean', 'std', 'min', 'max']])

# ============================================================================
# STEP 3: FEATURE ENGINEERING - AGGREGATE BY LOCATION
# ============================================================================
print("\n" + "="*80)
print("STEP 3: FEATURE ENGINEERING")
print("-"*80)

print("\nAggregating temporal statistics for each location...")
print("  Computing: mean, std, min, max for all parameters")

# Group by location and compute statistics
agg_dict = {col: ['mean', 'std', 'min', 'max'] for col in feature_cols}
df_agg = df.groupby(['lat', 'lon']).agg(agg_dict)

# Flatten column names
df_agg.columns = ['_'.join(col).strip() for col in df_agg.columns.values]
df_agg = df_agg.reset_index()

print(f"✓ Created {len(df_agg.columns) - 2} aggregated features")
print(f"  Locations: {len(df_agg)}")

# Add seasonal aggregations for key parameters
print("\nComputing seasonal statistics...")
df['time'] = pd.to_datetime(df['time'])
df['season'] = df['time'].dt.month.map({
    12: 'winter', 1: 'winter', 2: 'winter',
    3: 'spring', 4: 'spring', 5: 'spring',
    6: 'summer', 7: 'summer', 8: 'summer',
    9: 'fall', 10: 'fall', 11: 'fall'
})

# Key parameters for seasonal analysis
seasonal_params = [
    'T2M', 'PRECTOTCORR', 'RH2M', 'WS2M', 
    'ALLSKY_SFC_SW_DWN', 'CLOUD_AMT'
]

# Compute seasonal means
df_seasonal = df.groupby(['lat', 'lon', 'season'])[seasonal_params].mean()
df_seasonal = df_seasonal.reset_index()

# Pivot to get seasons as columns
df_seasonal_pivot = df_seasonal.pivot_table(
    index=['lat', 'lon'],
    columns='season',
    values=seasonal_params
)
df_seasonal_pivot.columns = ['_'.join(col).strip() for col in df_seasonal_pivot.columns.values]
df_seasonal_pivot = df_seasonal_pivot.reset_index()

print(f"✓ Created {len(df_seasonal_pivot.columns) - 2} seasonal features")

# Merge aggregated and seasonal features
df_features = pd.merge(df_agg, df_seasonal_pivot, on=['lat', 'lon'], how='inner')

print(f"\n✓ Total engineered features: {len(df_features.columns) - 2}")
print(f"  Locations: {len(df_features)}")

# Check for NaN after aggregation
nan_cols = df_features.isnull().sum()
if nan_cols.sum() > 0:
    print(f"\n⚠ Found {nan_cols.sum()} NaN values after aggregation")
    print("Columns with NaN:")
    print(nan_cols[nan_cols > 0])
    print("Filling NaN with column means...")
    df_features = df_features.fillna(df_features.mean())

# ============================================================================
# STEP 4: REMOVE HIGHLY CORRELATED FEATURES
# ============================================================================
print("\n" + "="*80)
print("STEP 4: REMOVING REDUNDANT FEATURES")
print("-"*80)

# Get feature columns (exclude lat, lon)
feature_cols_eng = [col for col in df_features.columns if col not in ['lat', 'lon']]

print(f"Analyzing correlations for {len(feature_cols_eng)} features...")
correlation_matrix = df_features[feature_cols_eng].corr().abs()

# Find highly correlated features (>0.99)
upper_triangle = correlation_matrix.where(
    np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
)

to_drop = [column for column in upper_triangle.columns 
           if any(upper_triangle[column] > 0.99)]

print(f"\nFound {len(to_drop)} highly correlated features (r > 0.99):")
if len(to_drop) > 0:
    for col in to_drop[:10]:  # Show first 10
        print(f"  - {col}")
    if len(to_drop) > 10:
        print(f"  ... and {len(to_drop) - 10} more")
    
    print("\nRemoving redundant features...")
    df_features = df_features.drop(columns=to_drop)
    feature_cols_eng = [col for col in df_features.columns if col not in ['lat', 'lon']]
    print(f"✓ Reduced to {len(feature_cols_eng)} features")
else:
    print("✓ No highly correlated features found")

# ============================================================================
# STEP 5: NORMALIZATION
# ============================================================================
print("\n" + "="*80)
print("STEP 5: Z-SCORE NORMALIZATION")
print("-"*80)

X = df_features[feature_cols_eng].values
locations = df_features[['lat', 'lon']].values

print(f"Feature matrix shape: {X.shape}")
print(f"  Locations: {X.shape[0]}")
print(f"  Features: {X.shape[1]}")

# Fit scaler
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

print(f"\n✓ Normalization complete")
print(f"  Mean: {X_normalized.mean():.10f} (should be ~0)")
print(f"  Std:  {X_normalized.std():.10f} (should be ~1)")

# Save scaler
os.makedirs("preprocessed_data", exist_ok=True)
import pickle
with open("preprocessed_data/scaler_495locs.pkl", "wb") as f:
    pickle.dump(scaler, f)
print("✓ Saved scaler to: preprocessed_data/scaler_495locs.pkl")

# ============================================================================
# STEP 6: PCA DIMENSIONALITY REDUCTION
# ============================================================================
print("\n" + "="*80)
print("STEP 6: PCA DIMENSIONALITY REDUCTION")
print("-"*80)

# Fit PCA to analyze variance
pca_full = PCA()
pca_full.fit(X_normalized)

# Determine components for variance thresholds
variance_thresholds = [0.90, 0.95, 0.99]
for threshold in variance_thresholds:
    n_components = np.argmax(np.cumsum(pca_full.explained_variance_ratio_) >= threshold) + 1
    print(f"  Components for {int(threshold*100)}% variance: {n_components}")

# Use 95% variance
target_variance = 0.95
n_components = np.argmax(np.cumsum(pca_full.explained_variance_ratio_) >= target_variance) + 1

print(f"\n✓ Using {n_components} components (95% variance)")

# Transform data
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_normalized)

print(f"  Original dimensions: {X_normalized.shape[1]}")
print(f"  Reduced dimensions: {X_pca.shape[1]}")
print(f"  Compression ratio: {X_normalized.shape[1] / X_pca.shape[1]:.1f}x")
print(f"  Variance retained: {pca.explained_variance_ratio_.sum():.4f}")

# Save PCA model
with open("preprocessed_data/pca_model_495locs.pkl", "wb") as f:
    pickle.dump(pca, f)
print("✓ Saved PCA model to: preprocessed_data/pca_model_495locs.pkl")

# ============================================================================
# STEP 7: DETERMINE OPTIMAL K
# ============================================================================
print("\n" + "="*80)
print("STEP 7: DETERMINING OPTIMAL K")
print("-"*80)

k_range = range(3, 16)
inertias = []
silhouettes = []
db_scores = []

print(f"Testing K from {min(k_range)} to {max(k_range)}...")
print("Running multiple initializations per K (n_init=20)...")

for k in k_range:
    kmeans = KMeans(n_clusters=k, n_init=20, random_state=42, max_iter=300)
    labels = kmeans.fit_predict(X_pca)
    
    inertias.append(kmeans.inertia_)
    silhouettes.append(silhouette_score(X_pca, labels))
    db_scores.append(davies_bouldin_score(X_pca, labels))
    
    print(f"  K={k:2d}: Inertia={kmeans.inertia_:8.2f}, "
          f"Silhouette={silhouettes[-1]:.3f}, "
          f"Davies-Bouldin={db_scores[-1]:.3f}")

# Find optimal K
optimal_k_silhouette = k_range[np.argmax(silhouettes)]
optimal_k_db = k_range[np.argmin(db_scores)]

print(f"\n✓ Optimal K by Silhouette Score: {optimal_k_silhouette}")
print(f"✓ Optimal K by Davies-Bouldin Index: {optimal_k_db}")

# ============================================================================
# STEP 8: FINAL CLUSTERING
# ============================================================================
print("\n" + "="*80)
print("STEP 8: FINAL CLUSTERING")
print("-"*80)

# Use silhouette-based optimal K
optimal_k = optimal_k_silhouette
print(f"Performing final K-means clustering with K={optimal_k}...")

kmeans_final = KMeans(n_clusters=optimal_k, n_init=50, random_state=42, max_iter=500)
labels = kmeans_final.fit_predict(X_pca)

print(f"\n✓ Clustering complete!")
print(f"\nCluster distribution:")
unique, counts = np.unique(labels, return_counts=True)
for cluster_id, count in zip(unique, counts):
    print(f"  Cluster {cluster_id}: {count} locations ({count/len(labels)*100:.1f}%)")

# ============================================================================
# STEP 9: SAVE RESULTS
# ============================================================================
print("\n" + "="*80)
print("STEP 9: SAVING RESULTS")
print("-"*80)

# Create results dataframe
df_results = pd.DataFrame({
    'lat': locations[:, 0],
    'lon': locations[:, 1],
    'cluster': labels
})

# Add PCA coordinates for visualization
for i in range(min(3, X_pca.shape[1])):
    df_results[f'PC{i+1}'] = X_pca[:, i]

# Save results
output_file = "preprocessed_data/clustering_results_495locs.csv"
df_results.to_csv(output_file, index=False)
print(f"✓ Saved clustering results: {output_file}")

# Save feature matrix
df_features_output = df_features.copy()
df_features_output['cluster'] = labels
df_features_output.to_csv("preprocessed_data/engineered_features_495locs.csv", index=False)
print(f"✓ Saved engineered features: preprocessed_data/engineered_features_495locs.csv")

# Save metrics
metrics = {
    'k_range': list(k_range),
    'inertias': inertias,
    'silhouettes': silhouettes,
    'davies_bouldin': db_scores,
    'optimal_k': optimal_k
}
import json
with open("preprocessed_data/clustering_metrics_495locs.json", "w") as f:
    json.dump(metrics, f, indent=2)
print(f"✓ Saved metrics: preprocessed_data/clustering_metrics_495locs.json")

# ============================================================================
# STEP 10: VISUALIZATIONS
# ============================================================================
print("\n" + "="*80)
print("STEP 10: CREATING VISUALIZATIONS")
print("-"*80)

os.makedirs("preprocessed_data/visualizations", exist_ok=True)

# 1. Elbow plot
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Number of Clusters (K)', fontsize=12)
plt.ylabel('Inertia (WCSS)', fontsize=12)
plt.title('Elbow Method', fontsize=14)
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
plt.plot(k_range, silhouettes, 'go-', linewidth=2, markersize=8)
plt.axvline(optimal_k, color='r', linestyle='--', alpha=0.7, label=f'Optimal K={optimal_k}')
plt.xlabel('Number of Clusters (K)', fontsize=12)
plt.ylabel('Silhouette Score', fontsize=12)
plt.title('Silhouette Analysis', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 3)
plt.plot(k_range, db_scores, 'ro-', linewidth=2, markersize=8)
plt.xlabel('Number of Clusters (K)', fontsize=12)
plt.ylabel('Davies-Bouldin Index', fontsize=12)
plt.title('Davies-Bouldin Index (lower is better)', fontsize=14)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('preprocessed_data/visualizations/optimal_k_analysis.png', dpi=150, bbox_inches='tight')
print("✓ Saved: optimal_k_analysis.png")

# 2. Geographic cluster map
plt.figure(figsize=(14, 10))
scatter = plt.scatter(df_results['lon'], df_results['lat'], 
                     c=df_results['cluster'], cmap='tab10', 
                     s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
plt.colorbar(scatter, label='Cluster ID')
plt.xlabel('Longitude', fontsize=14)
plt.ylabel('Latitude', fontsize=14)
plt.title(f'Climate Microclimates of Washington State (K={optimal_k})', fontsize=16, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('preprocessed_data/visualizations/cluster_map.png', dpi=150, bbox_inches='tight')
print("✓ Saved: cluster_map.png")

# 3. PCA scatter (if we have at least 2 components)
if X_pca.shape[1] >= 2:
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], 
                         c=labels, cmap='tab10', 
                         s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
    plt.colorbar(scatter, label='Cluster ID')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', fontsize=12)
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)', fontsize=12)
    plt.title('Clusters in PCA Space', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('preprocessed_data/visualizations/pca_clusters.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: pca_clusters.png")

print("\n" + "="*80)
print("PREPROCESSING COMPLETE!")
print("="*80)
print(f"\n📊 SUMMARY:")
print(f"  • Processed {len(df_features)} locations")
print(f"  • Engineered {len(feature_cols_eng)} features")
print(f"  • Reduced to {n_components} PCA components")
print(f"  • Optimal clusters: K={optimal_k}")
print(f"  • Silhouette score: {silhouettes[optimal_k - min(k_range)]:.3f}")

print(f"\n📁 OUTPUT FILES:")
print(f"  • preprocessed_data/clustering_results_495locs.csv")
print(f"  • preprocessed_data/engineered_features_495locs.csv")
print(f"  • preprocessed_data/scaler_495locs.pkl")
print(f"  • preprocessed_data/pca_model_495locs.pkl")
print(f"  • preprocessed_data/clustering_metrics_495locs.json")
print(f"  • preprocessed_data/visualizations/")

print(f"\n🎯 NEXT STEPS:")
print(f"  1. Review cluster_map.png - Do clusters make geographic sense?")
print(f"  2. Analyze cluster characteristics by examining centroids")
print(f"  3. Validate against Köppen climate classification")
print(f"  4. If results look good, proceed to MapReduce implementation")
print("="*80)
