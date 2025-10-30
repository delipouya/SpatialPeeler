#!/usr/bin/env python3
# Per-sample GA: evolve a binary spatial mask to maximize DE genes (Welch t-test p<0.05)
# Initial population for each sample = binarized p-hat masks from GOF factors only.

import os
os.environ.setdefault("OMP_NUM_THREADS","1")
os.environ.setdefault("MKL_NUM_THREADS","1")
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
os.environ.setdefault("NUMEXPR_NUM_THREADS","1")

import numpy as np
import scanpy as sc
import scipy.sparse as sp
from scipy.stats import ttest_ind
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pickle
import pygad
import pandas as pd

# Your plotting utilities
from SpatialPeeler import plotting as plot

# ----------------------------
# Config
# ----------------------------
RAND_SEED = 28
np.random.seed(RAND_SEED)

# Choose dataset variant
results_path = '/home/delaram/SpatialPeeler/Data/Remyelin_Slide-seq/results_Remyelin_uncropped_t18.pkl'
adata_path   = '/home/delaram/SpatialPeeler/Data/Remyelin_Slide-seq/Remyelin_NMF_30_uncropped_t18.h5ad'

# choose factor set
GOF_index = [18, 12, 9, 20, 14]   # uncomment for t18
# GOF_index = [21, 1, 9, 22, 24]  # uncomment for t3_7

#### uncropped - t3_7 indices
#GOF_index = [21, 1, 9, 22, 24]
#LOF_index = [0, 27, 12, 5, 19]

#### uncropped - t18 indices
#GOF_index = [18, 12, 9, 20, 14]
#LOF_index = [3, 19, 23, 25]

GA_COL = "GA_mask_GOF_ttest"  # where we store the final per-spot binary mask

# ----------------------------
# Load
# ----------------------------
with open(results_path, 'rb') as f:
    results = pickle.load(f)      # list-like of 30 dicts; each has 'p_hat'
adata = sc.read_h5ad(adata_path)

# Ensure log1p (once, global) – avoids repeated preprocessing
if 'log1p' not in adata.uns_keys():
    sc.pp.log1p(adata)

# Prepare container for final mask
adata.obs[GA_COL] = np.nan  # will fill per sample with 0/1

# ----------------------------
# ----------------------------
def binarize_phat(phat_1d: np.ndarray) -> np.ndarray:
    """
    Cluster p-hat into 2 groups using KMeans; label 1 as the higher-mean cluster.
    """
    km = KMeans(n_clusters=2, random_state=RAND_SEED).fit(phat_1d.reshape(-1, 1))
    centers = km.cluster_centers_.ravel()
    remap = {np.argmin(centers): 0, np.argmax(centers): 1}
    return np.vectorize(remap.get)(km.labels_).astype(np.uint8)

def de_score_ttest(mask_bool: np.ndarray, X: np.ndarray) -> int:
    """
    Welch t-test per gene between inside vs outside.
    Fitness = count of genes with p < 0.05.
    """
    inside  = mask_bool
    outside = ~inside
    n_in, n_out = int(inside.sum()), int(outside.sum())
    if n_in == 0 or n_out == 0:
        return 0
    Xin, Xout = X[inside, :], X[outside, :]
    _, pvals = ttest_ind(Xin, Xout, axis=0, equal_var=False)
    return int(np.sum(pvals < 0.05))

# ----------------------------
# Run GA per sample
# ----------------------------
sample_ids = adata.obs['sample_id'].unique().tolist()
print(f"Found {len(sample_ids)} samples.")

for sid in sample_ids:
    print(f"\n=== Sample: {sid} ===")
    idx = (adata.obs['sample_id'].values == sid)
    ad_s = adata[idx].copy()
    n_obs_s = ad_s.n_obs

    # Dense expression for this sample; drop 0-var genes for stability
    Xs = ad_s.X.toarray() if sp.issparse(ad_s.X) else np.asarray(ad_s.X)
    varmask = (Xs.var(axis=0) > 0)
    Xs = Xs[:, varmask]
    if Xs.shape[1] == 0:
        print("No variable genes after filtering; skipping.")
        continue

    # --- Initial population: binarize p-hat for GOF factors on THIS sample only ---
    init = []
    for i in GOF_index:
        phat_full = np.asarray(results[i]['p_hat'])
        phat_s = phat_full[idx]                    # restrict to this sample
        bin_mask = binarize_phat(phat_s)           # 0/1 per spot in this sample
        init.append(bin_mask)

    initial_population = np.stack(init, axis=0).astype(np.uint8)  # (len(GOF_index), n_obs_s)

    # --- Fitness wrapper for pygad ---
    def fitness_func(ga, sol, _):
        mask = sol.astype(bool, copy=False)
        return float(de_score_ttest(mask, Xs))

    # --- GA config (small & fast) ---
    ga = pygad.GA(
        fitness_func=fitness_func,
        num_generations=100,
        num_parents_mating=min(5, initial_population.shape[0]),
        sol_per_pop=initial_population.shape[0],
        num_genes=n_obs_s,
        gene_space=[0, 1],
        gene_type=np.uint8,
        initial_population=initial_population,

        parent_selection_type="tournament",
        K_tournament=3,
        keep_parents=min(2, initial_population.shape[0]),

        crossover_type="two_points",
        crossover_probability=0.9,

        mutation_type="random",
        mutation_probability=0.10,
        mutation_num_genes=max(1, int(0.005 * n_obs_s)),  # ~0.5%

        stop_criteria=["saturate_30"],
        random_seed=RAND_SEED,
    )

    def on_generation(g):
        if g.generations_completed in (1, 5, 10, 20, 50) or g.generations_completed % 10 == 0:
            best, fit, _ = g.best_solution()
            print(f"Gen {g.generations_completed:3d} | Fitness={fit:.1f} | Mask size={int(best.sum())}")
    ga.on_generation = on_generation

    # --- Run GA for this sample ---
    ga.run()
    best_vec, best_fit, _ = ga.best_solution()
    print(f"Best fitness (sample {sid}): {best_fit}  | Mask size: {int(best_vec.sum())}")

    # Write back to main adata.obs
    adata.obs.loc[idx, GA_COL] = best_vec

# Cast GA mask to category/string (0/1) for plotting
adata.obs[GA_COL] = adata.obs[GA_COL].astype('Int64').astype(str)

# ----------------------------
# Plot per sample
# ----------------------------
adata_by_sample = {sid: adata[adata.obs['sample_id'] == sid].copy() for sid in sample_ids}

plot.plot_grid_upgrade(
    adata_by_sample, sample_ids,
    key=GA_COL,
    title_prefix="GA mask (GOF, Welch t-test p<0.05)",
    from_obsm=False, discrete=True,
    palette={"0": "#4C78A8", "1": "#F58518"},
    dot_size=2, figsize=(25, 10)
)

# Optional: visualize fitness trajectories? (we ran a GA per sample; not stored globally)
# If you want to save per-sample ga.best_solutions_fitness, collect them in a dict within the loop.