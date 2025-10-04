import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import scanpy as sc
import pandas as pd
import anndata as ad

rng = np.random.default_rng(42)

# sanity images
def show_marker(adata_like, gene, title_suffix=""):

    if gene not in adata_like.var_names:
        print(f"[warn] {gene} not in var_names")
        return
    gi = adata_like.var_names.get_loc(gene)
    col = adata_like.X[:, gi]
    if sp.issparse(col):
        col = col.toarray().ravel()
    else:
        col = np.asarray(col).ravel()
    H, W = 200, 300        # grid height × width
    img = col.reshape(H, W)

    plt.figure(figsize=(6, 4))
    plt.imshow(img, origin="lower", interpolation="nearest")
    plt.title(f"{gene} expression {title_suffix}".strip())
    plt.axis("off")
    plt.colorbar(label="expression")
    plt.show()

marker_list = ['IL32','CD2','MS4A1','CD19','PRF1','GZMA','GZMK',
              'NKG7','PDCD1','TIGIT','CD79A','CD79B']
# ---------------------------
seed_t = ad.read_h5ad("/home/delaram/SpatialPeeler/Data/seed_top_cd4_inside_b_outside.h5ad")
seed_b = ad.read_h5ad("/home/delaram/SpatialPeeler/Data/seed_bottom_cd4_inside_b_outside.h5ad")
ground_truth = ad.read_h5ad("/home/delaram/SpatialPeeler/Data/ground_truth_cd4_inside_b_outside.h5ad")

for marker in ['IL32','CD2']:
    show_marker(seed_t, marker, "(seed top)")
    show_marker(seed_b, marker, "(seed bottom)")
    show_marker(ground_truth, marker,"(ground truth)")



#### genetic algorithm tutorial with pygad 
import pygad

