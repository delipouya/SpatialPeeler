import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern, RationalQuadratic
from scipy.stats import pearsonr
import pandas as pd

# Simulate spatial coordinates
n_cells = 300
coords = np.random.uniform(0, 10, size=(n_cells, 2))

# Simulate a hidden spatial field (e.g., unmodeled biology) for residuals
def hidden_field(xy):
    return np.sin(xy[:, 0] / 2) + np.cos(xy[:, 1] / 2)

residual_signal = hidden_field(coords)
residual_noise = np.random.normal(0, 0.5, size=n_cells)
residuals = residual_signal + residual_noise

# Simulate gene expressions (some spatial, some not)
n_genes = 20
gene_expr_matrix = np.zeros((n_cells, n_genes))

# First few genes are spatially correlated with residual field
for i in range(5):
    gene_expr_matrix[:, i] = residual_signal + np.random.normal(0, 0.3, size=n_cells)

# Next few are spatial but different
for i in range(5, 10):
    gene_expr_matrix[:, i] = np.sin(coords[:, 0] * (i - 4) / 10) + np.random.normal(0, 0.3, size=n_cells)

# The rest are pure noise
for i in range(10, n_genes):
    gene_expr_matrix[:, i] = np.random.normal(0, 1, size=n_cells)

# Choose kernel
# Uncomment the kernel you want to test
# kernel = 1.0 * RBF(length_scale=2.0) + WhiteKernel(noise_level=0.1)
# kernel = 1.0 * Matern(length_scale=2.0, nu=1.5) + WhiteKernel(noise_level=0.1)
# kernel = 1.0 * RationalQuadratic(length_scale=2.0, alpha=1.0) + WhiteKernel(noise_level=0.1)
kernel = 1.0 * RBF(length_scale=2.0) + WhiteKernel(noise_level=0.1)  # default kernel

# Fit GP to residuals
gp_r = GaussianProcessRegressor(kernel=kernel, alpha=1e-4, normalize_y=True)
gp_r.fit(coords, residuals)
residual_gp_mean = gp_r.predict(coords)

# Fit GP for each gene and compare with residual GP mean
similarity_scores = []
for g in range(n_genes):
    gp_g = GaussianProcessRegressor(kernel=kernel, alpha=1e-4, normalize_y=True)
    gp_g.fit(coords, gene_expr_matrix[:, g])
    gene_gp_mean = gp_g.predict(coords)

    corr, _ = pearsonr(gene_gp_mean, residual_gp_mean)
    similarity_scores.append(corr)

# Package results
gene_names = [f"Gene_{i+1}" for i in range(n_genes)]
results = pd.DataFrame({
    "Gene": gene_names,
    "Similarity_to_residual_GP": similarity_scores
}).sort_values(by="Similarity_to_residual_GP", ascending=False)

# Visualize top spatial matches
top_genes = results["Gene"].values[:3]
top_indices = [int(name.split("_")[1]) - 1 for name in top_genes]

fig, axs = plt.subplots(1, 4, figsize=(20, 4))

axs[0].scatter(coords[:, 0], coords[:, 1], c=residual_gp_mean, cmap="viridis")
axs[0].set_title("Residual GP mean")
axs[0].axis('off')

for i, idx in enumerate(top_indices):
    gp_g = GaussianProcessRegressor(kernel=kernel, alpha=1e-4, normalize_y=True)
    gp_g.fit(coords, gene_expr_matrix[:, idx])
    gene_gp_mean = gp_g.predict(coords)
    axs[i + 1].scatter(coords[:, 0], coords[:, 1], c=gene_gp_mean, cmap="viridis")
    axs[i + 1].set_title(f"{gene_names[idx]} GP mean")
    axs[i + 1].axis('off')

plt.tight_layout()
plt.show()

# Print results
print(results)