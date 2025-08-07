import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
import joblib
import os

# 1. Setup directories
os.makedirs("data/processed", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("reports/figures", exist_ok=True)

# 2. Load preprocessed data
print("Loading data...")
df = pd.read_csv("data/processed/preprocessed_students.csv")

# 3. PCA Dimensionality Reduction
print("Running PCA...")
pca = PCA(n_components=0.95, random_state=42)
X_pca = pca.fit_transform(df)
print(f"Reduced from {df.shape[1]} to {X_pca.shape[1]} dimensions")

# 4. t-SNE Embedding
print("Running t-SNE...")
tsne = TSNE(
    n_components=2,
    perplexity=min(30, len(df)-1),
    early_exaggeration=12,
    learning_rate=200,
    random_state=42,
    n_jobs=-1
)
X_tsne = tsne.fit_transform(X_pca)

# 5. DBSCAN Clustering with Auto-tuned Epsilon
print("Optimizing DBSCAN...")
neigh = NearestNeighbors(n_neighbors=10)
neigh.fit(X_tsne)
distances, _ = neigh.kneighbors(X_tsne)
k_dist = np.sort(distances[:, -1])
optimal_eps = k_dist[int(len(k_dist)*0.15)]  # 15th percentile rule

dbscan = DBSCAN(
    eps=optimal_eps,
    min_samples=max(5, int(0.001*len(df)))  # THIS LINE IS NOW CORRECTLY CLOSED
)
clusters = dbscan.fit_predict(X_tsne)

# 6. Evaluation
score = silhouette_score(X_tsne, clusters)
print("\n=== Results ===")
print(f"Optimal eps: {optimal_eps:.2f}")
print(f"Silhouette Score: {score:.3f}")
print("Cluster Distribution:")
print(pd.Series(clusters).value_counts().sort_index())

# 7. Save Outputs
df['Cluster'] = clusters
df.to_csv("data/processed/students_clusters_final.csv", index=False)
joblib.dump({'pca': pca, 'tsne': tsne, 'dbscan': dbscan}, 
            "models/clustering_pipeline.pkl")

# 8. Professional Visualization
print("Generating visualization...")
plt.style.use('seaborn-v0_8-notebook')
plt.figure(figsize=(12, 8))

# Main clusters
scatter = plt.scatter(
    X_tsne[:, 0], X_tsne[:, 1],
    c=clusters,
    cmap='viridis',
    s=25,
    alpha=0.8,
    edgecolor='w',
    linewidth=0.3
)

# Highlight noise (if exists)
if -1 in clusters:
    noise_mask = clusters == -1
    plt.scatter(
        X_tsne[noise_mask, 0], X_tsne[noise_mask, 1],
        c='red',
        s=15,
        label='Noise',
        alpha=0.5
    )
    plt.legend()

# Annotations
plt.title(f"Student Clusters\nSilhouette Score: {score:.2f}", pad=20, fontsize=14)
plt.xlabel("t-SNE Dimension 1", labelpad=10)
plt.ylabel("t-SNE Dimension 2", labelpad=10)
plt.colorbar(scatter, label='Cluster ID', pad=0.02)

# Save and show
plt.tight_layout()
plt.savefig("reports/figures/final_clusters.png", 
            dpi=300, 
            bbox_inches='tight',
            transparent=False)
print("Visualization saved to reports/figures/final_clusters.png")
plt.close()

print("\n=== Pipeline Complete ===")