"""
Compare EMR Distributed Results with Local Results
Validates that distributed clustering matches local clustering
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import os

print("="*80)
print("COMPARING EMR vs LOCAL CLUSTERING RESULTS")
print("="*80)

# =============================================================================
# LOAD DATA
# =============================================================================

print("\n[1] Loading results...")

# Load EMR results
emr_file = "emr_result.csv"
df_emr = pd.read_csv(emr_file)
print(f"✓ EMR results: {len(df_emr)} locations")

# Load local results
local_file = "preprocessed_data/clustering_results_495locs.csv"
df_local = pd.read_csv(local_file)
print(f"✓ Local results: {len(df_local)} locations")

# Merge on location
df_compare = pd.merge(
    df_emr, 
    df_local[['lat', 'lon', 'cluster']], 
    on=['lat', 'lon'], 
    suffixes=('_emr', '_local')
)

print(f"✓ Matched {len(df_compare)} locations")

# =============================================================================
# CLUSTER DISTRIBUTION COMPARISON
# =============================================================================

print("\n" + "="*80)
print("CLUSTER DISTRIBUTIONS")
print("="*80)

print("\nEMR Clustering:")
emr_counts = df_compare['cluster_emr'].value_counts().sort_index()
for cluster, count in emr_counts.items():
    print(f"  Cluster {cluster}: {count} locations ({count/len(df_compare)*100:.1f}%)")

print("\nLocal Clustering:")
local_counts = df_compare['cluster_local'].value_counts().sort_index()
for cluster, count in local_counts.items():
    print(f"  Cluster {cluster}: {count} locations ({count/len(df_compare)*100:.1f}%)")

# =============================================================================
# SIMILARITY METRICS
# =============================================================================

print("\n" + "="*80)
print("SIMILARITY METRICS")
print("="*80)

# Adjusted Rand Index (1.0 = perfect match, 0.0 = random)
ari = adjusted_rand_score(df_compare['cluster_local'], df_compare['cluster_emr'])
print(f"\nAdjusted Rand Index: {ari:.4f}")
if ari > 0.8:
    print("  ✓ Excellent agreement!")
elif ari > 0.5:
    print("  ✓ Good agreement")
elif ari > 0.3:
    print("  ⚠ Moderate agreement")
else:
    print("  ⚠ Low agreement - may need investigation")

# Normalized Mutual Information (1.0 = perfect match)
nmi = normalized_mutual_info_score(df_compare['cluster_local'], df_compare['cluster_emr'])
print(f"\nNormalized Mutual Information: {nmi:.4f}")
if nmi > 0.8:
    print("  ✓ Excellent information sharing!")
elif nmi > 0.5:
    print("  ✓ Good information sharing")
else:
    print("  ⚠ Moderate information sharing")

# =============================================================================
# CONFUSION MATRIX
# =============================================================================

print("\n" + "="*80)
print("CLUSTER MAPPING")
print("="*80)

# Create confusion matrix
confusion = pd.crosstab(
    df_compare['cluster_local'], 
    df_compare['cluster_emr'],
    rownames=['Local'],
    colnames=['EMR']
)

print("\nConfusion Matrix (Local vs EMR):")
print("Rows = Local clusters, Columns = EMR clusters")
print(confusion)

# Find best mapping
print("\nBest cluster mappings (Local → EMR):")
for local_cluster in confusion.index:
    emr_cluster = confusion.loc[local_cluster].idxmax()
    overlap = confusion.loc[local_cluster, emr_cluster]
    total = confusion.loc[local_cluster].sum()
    print(f"  Local Cluster {local_cluster} → EMR Cluster {emr_cluster} ({overlap}/{total} = {overlap/total*100:.1f}%)")

# =============================================================================
# VISUALIZATIONS
# =============================================================================

print("\n" + "="*80)
print("CREATING VISUALIZATIONS")
print("="*80)

os.makedirs("emr_comparison", exist_ok=True)

# 1. Side-by-side maps
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

# Local results
scatter1 = ax1.scatter(df_compare['lon'], df_compare['lat'], 
                      c=df_compare['cluster_local'], cmap='tab20',
                      s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
ax1.set_xlabel('Longitude', fontsize=12)
ax1.set_ylabel('Latitude', fontsize=12)
ax1.set_title('Local Clustering Results', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
plt.colorbar(scatter1, ax=ax1, label='Cluster ID')

# EMR results
scatter2 = ax2.scatter(df_compare['lon'], df_compare['lat'], 
                      c=df_compare['cluster_emr'], cmap='tab20',
                      s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
ax2.set_xlabel('Longitude', fontsize=12)
ax2.set_ylabel('Latitude', fontsize=12)
ax2.set_title('EMR Distributed Clustering Results', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
plt.colorbar(scatter2, ax=ax2, label='Cluster ID')

plt.tight_layout()
plt.savefig('emr_comparison/local_vs_emr_maps.png', dpi=150, bbox_inches='tight')
print("✓ Saved: local_vs_emr_maps.png")
plt.close()

# 2. Confusion matrix heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', 
            cbar_kws={'label': 'Number of Locations'})
plt.title('Clustering Comparison: Local vs EMR\n(Higher diagonal = better match)', 
         fontsize=14, fontweight='bold', pad=20)
plt.xlabel('EMR Cluster', fontsize=12)
plt.ylabel('Local Cluster', fontsize=12)
plt.tight_layout()
plt.savefig('emr_comparison/confusion_matrix.png', dpi=150, bbox_inches='tight')
print("✓ Saved: confusion_matrix.png")
plt.close()

# 3. Agreement map (where do they agree/disagree?)
df_compare['agreement'] = (df_compare['cluster_local'] == df_compare['cluster_emr'])

fig, ax = plt.subplots(figsize=(14, 10))
colors = ['red' if not agree else 'green' for agree in df_compare['agreement']]
scatter = ax.scatter(df_compare['lon'], df_compare['lat'], 
                    c=colors, s=120, alpha=0.6, edgecolors='black', linewidth=0.5)

# Create legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='green', label=f'Agreement ({df_compare["agreement"].sum()} locations)'),
    Patch(facecolor='red', label=f'Disagreement ({(~df_compare["agreement"]).sum()} locations)')
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=11)

ax.set_xlabel('Longitude', fontsize=12)
ax.set_ylabel('Latitude', fontsize=12)
ax.set_title(f'Clustering Agreement Map\nAgreement Rate: {df_compare["agreement"].mean()*100:.1f}%', 
            fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('emr_comparison/agreement_map.png', dpi=150, bbox_inches='tight')
print("✓ Saved: agreement_map.png")
plt.close()

# 4. Cluster size comparison
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(max(len(emr_counts), len(local_counts)))
width = 0.35

# Prepare data
emr_sizes = [emr_counts.get(i, 0) for i in range(14)]
local_sizes = [local_counts.get(i, 0) for i in range(14)]

bars1 = ax.bar(x - width/2, local_sizes, width, label='Local', alpha=0.8)
bars2 = ax.bar(x + width/2, emr_sizes, width, label='EMR', alpha=0.8)

ax.set_xlabel('Cluster ID', fontsize=12)
ax.set_ylabel('Number of Locations', fontsize=12)
ax.set_title('Cluster Size Comparison: Local vs EMR', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('emr_comparison/cluster_sizes.png', dpi=150, bbox_inches='tight')
print("✓ Saved: cluster_sizes.png")
plt.close()

# =============================================================================
# SUMMARY REPORT
# =============================================================================

print("\n" + "="*80)
print("VALIDATION SUMMARY")
print("="*80)

agreement_rate = df_compare['agreement'].mean()
print(f"\n📊 Overall Statistics:")
print(f"  • Total locations: {len(df_compare)}")
print(f"  • Agreement rate: {agreement_rate*100:.1f}%")
print(f"  • Disagreement: {(1-agreement_rate)*100:.1f}%")
print(f"  • Adjusted Rand Index: {ari:.4f}")
print(f"  • Normalized Mutual Information: {nmi:.4f}")

print(f"\n�� Output Files:")
print(f"  • emr_comparison/local_vs_emr_maps.png")
print(f"  • emr_comparison/confusion_matrix.png")
print(f"  • emr_comparison/agreement_map.png")
print(f"  • emr_comparison/cluster_sizes.png")

print(f"\n✅ Validation Assessment:")
if ari > 0.8 and agreement_rate > 0.8:
    print("  EXCELLENT: EMR distributed clustering matches local results very well!")
    print("  Your MapReduce implementation is validated. ✓")
elif ari > 0.5 and agreement_rate > 0.6:
    print("  GOOD: EMR clustering is consistent with local results.")
    print("  Small differences are expected due to random initialization.")
elif ari > 0.3:
    print("  MODERATE: Some consistency but notable differences.")
    print("  This is normal for K-means with different initializations.")
else:
    print("  LOW: Significant differences detected.")
    print("  Check if K value or parameters match between local and EMR.")

print("\n" + "="*80)
print("COMPARISON COMPLETE!")
print("="*80)
