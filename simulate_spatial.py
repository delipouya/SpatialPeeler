import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng(42)

# Grid (H x W)
H, W = 200, 300
y, x = np.ogrid[:H, :W]

# Circle parameters
cy, cx, r = H/2, W/2, min(H, W) * 0.25
circle_mask = (x - cx)**2 + (y - cy)**2 <= r**2

# Half-circle masks (split by horizontal axis through center)
top_half    = circle_mask & (y < cy)
bottom_half = circle_mask & (y >= cy)

# Gaussian parameters
mu_top, sigma_top       = 2.0, 0.5   # top half
mu_bottom, sigma_bottom = -1.0, 0.5  # bottom half
mu_out, sigma_out       = 0.0, 0.5   # outside circle

# SEED1 = Sample field - a half-circle on the bottom field
field1 = np.empty((H, W), dtype=float)
field1[~circle_mask] = rng.normal(mu_out, sigma_out, size=(~circle_mask).sum()) ## Outside circle
field1[top_half]    = rng.normal(mu_out, sigma_out, size=top_half.sum()) ## Top half same as outside
field1[bottom_half] = rng.normal(mu_bottom, sigma_bottom, size=bottom_half.sum()) ## Bottom half 

# Visualize
plt.figure(figsize=(6, 4))
plt.imshow(field1, origin='lower', interpolation='nearest', cmap="coolwarm")
plt.title("Upper Gaussian half-circles")
plt.clim(-3, 3) 
plt.colorbar(label='Value')
plt.axis('off')
plt.show()

# SEED2 = Sample field - a half-circle on the top field
field2 = np.empty((H, W), dtype=float)
field2[~circle_mask] = rng.normal(mu_out, sigma_out, size=(~circle_mask).sum()) ## Outside circle
field2[bottom_half] = rng.normal(mu_out, sigma_out, size=bottom_half.sum()) ## Bottom half same as outside
field2[top_half]    = rng.normal(mu_top, sigma_top, size=top_half.sum()) ## Top half

# Visualize
plt.figure(figsize=(6, 4))
plt.imshow(field2, origin='lower', interpolation='nearest', cmap="coolwarm")
plt.title("Bottom Gaussian half-circles")
plt.clim(-3, 3) 
plt.colorbar(label='Value')
plt.axis('off')
plt.show()

# GROUND TRUTH = Sample field - a full circle in the middle of field
field3 = np.empty((H, W), dtype=float)
field3[~circle_mask] = rng.normal(mu_out, sigma_out, size=(~circle_mask).sum()) ## Outside circle
field3[top_half]    = rng.normal(mu_top, sigma_top, size=top_half.sum()) ## Top half
field3[bottom_half] = rng.normal(mu_bottom, sigma_bottom, size=bottom_half.sum()) ## Bottom half

# Visualize
plt.figure(figsize=(6, 4))
plt.imshow(field3, origin='lower', interpolation='nearest', cmap="coolwarm")
plt.title("Full circle in the middle of field")
plt.clim(-3, 3) 
plt.colorbar(label='Value')
plt.axis('off')
plt.show()

##################################


import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import scanpy as sc
import pandas as pd

# ---------------------------
# 0) Load PBMC and sanity check
# ---------------------------
adata = sc.datasets.pbmc3k_processed()  # 2638 cells × 1838 genes
print(adata.shape) #n_obs × n_vars = 2638 × 1838
print(adata.obs.louvain.value_counts()) # 8 clusters
#louvain
#CD4 T cells          1144
#CD14+ Monocytes       480
#B cells               342

# Required clusters
ct_in  = "CD4 T cells"   # inside circle
ct_out = "B cells"       # outside circle
assert ct_in in adata.obs["louvain"].unique(),  f"Missing {ct_in}"
assert ct_out in adata.obs["louvain"].unique(), f"Missing {ct_out}"

# ---------------------------
# 1) Make a spatial grid + circle mask
# ---------------------------
rng = np.random.default_rng(42)

H, W = 200, 300        # grid height × width
y, x = np.ogrid[:H, :W]

cy, cx = H / 2, W / 2
r = min(H, W) * 0.25   # circle radius
circle_mask = (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2

# ---------------------------
# 2) Sample cell indices in raster order
# ---------------------------
is_cd4 = (adata.obs["louvain"].values == ct_in)
is_b   = (adata.obs["louvain"].values == ct_out)
cd4_idx = np.where(is_cd4)[0]
b_idx   = np.where(is_b)[0]
assert cd4_idx.size > 0 and b_idx.size > 0

inside_flat = circle_mask.ravel()             # raster order mask
n_inside    = int(inside_flat.sum())
n_outside   = int((~inside_flat).sum())

pick_inside  = rng.choice(cd4_idx, size=n_inside,  replace=True)
pick_outside = rng.choice(b_idx,   size=n_outside, replace=True)

picked_all = np.empty(H * W, dtype=int)
picked_all[inside_flat]  = pick_inside
picked_all[~inside_flat] = pick_outside

# ---------------------------
# 3) Build AnnData in raster order (CRUCIAL)
# ---------------------------
X_spots = adata.X[picked_all]  # preserves raster order
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

seed1 = sc.AnnData(X_spots, obs=obs, var=adata.var.copy())
seed1.uns["spatial_layout"] = {
    "height": H, "width": W,
    "circle_center": (cy, cx),
    "circle_radius": r
}
seed1.uns["description"] = "SEED1: inside=CD4 T cells, outside=B cells (sampled with replacement from PBMC3k)."

print(seed1)
print(seed1.obs["assigned_cell_type"].value_counts())

# ---------------------------
# 4) Quick sanity images
# ---------------------------
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

# IL7R ~ CD4 T cells (should light up INSIDE); MS4A1 ~ B cells (should light up OUTSIDE)
for g in ["LTB", "IL32", "CD2", "MS4A1"]:  # T cells inside, B cells outside
    show_marker(seed1, g, H, W, "(SEED1)")
# ---------------------------
# 5) (Optional) Save
# ---------------------------
# seed1.write_h5ad("SEED1_cd4_inside_b_outside.h5ad")