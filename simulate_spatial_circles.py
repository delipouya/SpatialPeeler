##################################################################################
# we aim to create a spatial transcriptomics-like dataset using real single-cell data (PBMC3k)
# with CD4 T cells inside a circle and B cells outside the circle
##################################################################################
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import scanpy as sc
import pandas as pd

# sanity images
def show_marker(adata_like, gene, H, W, title_suffix=""):
    if gene not in adata_like.var_names:
        print(f"[warn] {gene} not in var_names")
        return
    gi = adata_like.var_names.get_loc(gene)
    col = adata_like.X[:, gi]
    if sp.issparse(col):
        col = col.toarray().ravel()
    else:
        col = np.asarray(col).ravel()
    img = col.reshape(H, W)

    plt.figure(figsize=(6, 4))
    plt.imshow(img, origin="lower", interpolation="nearest")
    plt.title(f"{gene} expression {title_suffix}".strip())
    plt.axis("off")
    plt.colorbar(label="expression")
    plt.show()


# ---------------------------
# Load PBMC and sanity check
adata = sc.datasets.pbmc3k_processed()  # 2638 cells × 1838 genes
print(adata.shape) #n_obs × n_vars = 2638 × 1838
print(adata.obs.louvain.value_counts()) # 8 clusters
#louvain
#CD4 T cells          1144
#CD14+ Monocytes       480
#B cells               342

ct_in  = "CD4 T cells"   # inside circle
ct_out = "B cells"       # outside circle
assert ct_in in adata.obs["louvain"].unique(),  f"Missing {ct_in}"
assert ct_out in adata.obs["louvain"].unique(), f"Missing {ct_out}"

# ---------------------------
# Make a spatial grid + circle mask
rng = np.random.default_rng(42)
H, W = 200, 300        # grid height × width
y, x = np.ogrid[:H, :W]

cy, cx = H / 2, W / 2
r = min(H, W) * 0.25   # circle radius
## formula for a circle centered at (c_x,c_y): (x - c_x)^2 + (y - c_y)^2 = r^2
circle_mask = (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2

# ---------------------------
# Sample cell indices from adata based on the mask
is_cd4 = (adata.obs["louvain"].values == ct_in)
is_b   = (adata.obs["louvain"].values == ct_out)
cd4_idx = np.where(is_cd4)[0]
b_idx   = np.where(is_b)[0]
print(cd4_idx.size > 0 and b_idx.size > 0)

inside_flat = circle_mask.ravel()      # flatten the array
n_inside    = int(inside_flat.sum())
n_outside   = int((inside_flat==False).sum())

pick_inside  = rng.choice(cd4_idx, size=n_inside,  replace=True) ## randomly sample from the cd4 indices
pick_outside = rng.choice(b_idx,   size=n_outside, replace=True)

picked_all = np.empty(H * W, dtype=int)
picked_all[inside_flat]  = pick_inside
picked_all[~inside_flat] = pick_outside ## vectorized grid with indices of cells from adata

# ---------------------------
# Build AnnData from the vectorized grid
X_spots = adata.X[picked_all]  
if not sp.issparse(X_spots):
    X_spots = sp.csr_matrix(X_spots)

yy, xx = np.indices((H, W))
obs = pd.DataFrame({
    "y": yy.ravel(),
    "x": xx.ravel(),
    "region": np.where(inside_flat, "inside_circle", "outside_circle"),
    "assigned_cell_type": np.where(inside_flat, ct_in, ct_out),
    "orig_cell_type": adata.obs["louvain"].values[picked_all],
    "orig_cell_barcode": adata.obs_names.values[picked_all],
})

ground_truth = sc.AnnData(X_spots, obs=obs, var=adata.var.copy())
ground_truth.uns["description"] = "ground_truth: inside=CD4 T cells, outside=B cells (sampled with replacement from PBMC3k)."

#ground_truth.uns["spatial_layout"] = {
#    "height": H, "width": W,
#    "circle_center": (cy, cx),
#    "circle_radius": r
#}

print(ground_truth)
print(ground_truth.obs["assigned_cell_type"].value_counts())
# B cells        52155
# CD4 T cells     7845

# ---------------------------

markers_to_vis = ["IL32", "CD2", "MS4A1", 'CD19']
immune_markers = ['PRF1', 'GZMA', 'GZMK', 'NKG7', 'CCR7', 'IL7R', 'TCF7', 
                  'SELL', 'CXCL13', 'CTLA4', 'PDCD1', 'LAG3', 'TIGIT',
                  'CD19', 'CD79A', 'CD79B', 'MS4A1', 'CCR7', 'CD38', 'PRDM1']
for a_marker in immune_markers:
    print(a_marker in adata.var_names, a_marker in ground_truth.var_names)
    if a_marker in ground_truth.var_names and a_marker not in markers_to_vis:
        markers_to_vis.append(a_marker)

# ---------------------------

# IL7R ~ CD4 T cells (should light up INSIDE); MS4A1 ~ B cells (should light up OUTSIDE)
for g in markers_to_vis:  # T cells inside, B cells outside
    show_marker(ground_truth, g, H, W, "(ground truth)")
# ---------------------------
ground_truth.write_h5ad("/home/delaram/SpatialPeeler/Data/ground_truth_cd4_inside_b_outside.h5ad")


################ creating a similr field with half a circle on the top/bottom insead of full circle
# ---------------------------
# Make a spatial grid + circle mask
rng = np.random.default_rng(42)
H, W = 200, 300        # grid height × width
y, x = np.ogrid[:H, :W]

cy, cx = H / 2, W / 2
r = min(H, W) * 0.25   # circle radius
## formula for a circle centered at (c_x,c_y): (x - c_x)^2 + (y - c_y)^2 = r^2
circle_mask = (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2 # full circle: (200, 300) -> 7845 true elements
half_circle_mask = circle_mask & (y > cy)  # half circle on top of cy -> 3872 true elements
half_circle_mask = circle_mask & (y < cy)  # half circle on bottom of cy -> 3872 true elements

# ---------------------------
# Sample cell indices from adata based on the mask
is_cd4 = (adata.obs["louvain"].values == ct_in)
is_b   = (adata.obs["louvain"].values == ct_out)
cd4_idx = np.where(is_cd4)[0]
b_idx   = np.where(is_b)[0]
print(cd4_idx.size > 0 and b_idx.size > 0)

inside_flat = half_circle_mask.ravel()      # flatten the array
n_inside    = int(inside_flat.sum())
n_outside   = int((inside_flat==False).sum())

pick_inside  = rng.choice(cd4_idx, size=n_inside,  replace=True) ## randomly sample from the cd4 indices
pick_outside = rng.choice(b_idx,   size=n_outside, replace=True)

picked_all = np.empty(H * W, dtype=int)
picked_all[inside_flat]  = pick_inside
picked_all[~inside_flat] = pick_outside ## vectorized grid with indices of cells from adata

# ---------------------------
# Build AnnData from the vectorized grid
X_spots = adata.X[picked_all]  
if not sp.issparse(X_spots):
    X_spots = sp.csr_matrix(X_spots)

yy, xx = np.indices((H, W))
obs = pd.DataFrame({
    "y": yy.ravel(),
    "x": xx.ravel(),
    "region": np.where(inside_flat, "inside_circle", "outside_circle"),
    "assigned_cell_type": np.where(inside_flat, ct_in, ct_out),
    "orig_cell_type": adata.obs["louvain"].values[picked_all],
    "orig_cell_barcode": adata.obs_names.values[picked_all],
})

seed = sc.AnnData(X_spots, obs=obs, var=adata.var.copy())
seed.uns["description"] = "seed: inside=CD4 T cells, outside=B cells (sampled with replacement from PBMC3k)."

print(seed)
print(seed.obs["assigned_cell_type"].value_counts())

# IL7R ~ CD4 T cells (should light up INSIDE); MS4A1 ~ B cells (should light up OUTSIDE)
for g in markers_to_vis:  # T cells inside, B cells outside
    show_marker(seed, g, H, W, "(seed)")
# ---------------------------
seed.write_h5ad("/home/delaram/SpatialPeeler/Data/seed_top_cd4_inside_b_outside.h5ad")
seed.write_h5ad("/home/delaram/SpatialPeeler/Data/seed_bottom_cd4_inside_b_outside.h5ad")
