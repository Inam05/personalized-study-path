import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import BayesianGaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE  # Non-linear dimensionality reduction
import matplotlib.pyplot as plt
import joblib

# Load data
df = pd.read_csv("data/processed/preprocessed_students.csv")

# Step 1: Hybrid Dimensionality Reduction
pca = PCA(n_components=0.95, whiten=True, random_state=42)
X_pca = pca.fit_transform(df)

# Step 2: t-SNE for Non-linear Structure (critical for behavioral data)
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_embedded = tsne.fit_transform(X_pca)

# Step 3: Bayesian GMM with Dirichlet Prior
bgmm = BayesianGaussianMixture(
    n_components=3,  # Force fewer clusters
    weight_concentration_prior_type="dirichlet_process",
    weight_concentration_prior=0.01,  # Stronger regularization
    covariance_type="tied",  # Shared covariance for stability
    max_iter=500,
    random_state=42
)
bgmm.fit(X_embedded)
clusters = bgmm.predict(X_embedded)

# Step 4: Evaluate
print(f"Silhouette Score: {silhouette_score(X_embedded, clusters):.3f}")
print(f"Cluster Distribution:\n{pd.Series(clusters).value_counts()}")

# Step 5: Save Outputs
df['Cluster'] = clusters
df.to_csv("data/processed/students_clusters_final.csv", index=False)
joblib.dump(bgmm, "models/bgmm_final_model.pkl")

# Step 6: Visualize
plt.figure(figsize=(10, 6))
plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=clusters, cmap='viridis', alpha=0.7, s=50)
plt.colorbar(label='Cluster')
plt.title("t-SNE + Bayesian GMM Clustering (Optimal)")
plt.savefig("reports/figures/final_clusters.png", dpi=300, bbox_inches='tight')