import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import joblib
import matplotlib.pyplot as plt
import os

# Ensure directories exist
os.makedirs("models", exist_ok=True)
os.makedirs("reports/figures", exist_ok=True)

# Load preprocessed data
df = pd.read_csv("D:\Projects\personalized-study-path\data\processed\preprocessed_students.csv")

# Dimensionality Reduction (PCA - retain 95% variance)
pca = PCA(n_components=0.95, random_state=42)
X_pca = pca.fit_transform(df)
print(f"Reduced to {X_pca.shape[1]} dimensions")

# Optimize GMM hyperparameters
gmm = GaussianMixture(
    n_components=3,               # Start with 3 clusters (adjust based on metrics)
    covariance_type='full',        # Best for flexible cluster shapes
    random_state=42
)
gmm.fit(X_pca)
clusters = gmm.predict(X_pca)

# Evaluate
print(f"Silhouette Score: {silhouette_score(X_pca, clusters):.3f}")

# Save artifacts
df['Cluster'] = clusters
df.to_csv("data/processed/students_clustered.csv", index=False)
joblib.dump(gmm, "models/gmm_model.pkl")
joblib.dump(pca, "models/pca_model.pkl")

# Visualize clusters (2D PCA)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.6)
plt.title("Student Clusters (PCA-Reduced)")
plt.colorbar(label='Cluster')
plt.savefig("reports/figures/gmm_clusters.png", dpi=300, bbox_inches='tight')
plt.close()