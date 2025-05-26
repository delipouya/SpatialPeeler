import scanpy as sc
import numpy as np
import scipy.sparse
import matplotlib.pyplot as plt
import scanpy as sc
import anndata as ad
import seaborn as sns
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from matplotlib.patches import Patch
from sklearn.cluster import KMeans
from scipy.special import logit


RAND_SEED = 28
CASE_COND = 1
def standalone_logistic(X, y):
    #clf = LogisticRegression(random_state=RAND_SEED, penalty='none').fit(X, y)
    clf = LogisticRegression(random_state=RAND_SEED, penalty=None).fit(X, y)
    predicted_label = clf.predict(X)
    predicted_prob = clf.predict_proba(X)
    return predicted_prob[:,1]

def PCA_logistic_kmeans_residual(adata, num_pcs):
    """
    Runs logistic regression and KMeans clustering on the PCA of the residual expression matrix.
    """
    # Use PCA scores computed from residuals  
    #### X_pca changed to X_pca_residual
    X_pca = adata.obsm["X_pca_residual"][:, :num_pcs]

    # Step 1: Logistic regression
    p_hat = standalone_logistic(X_pca, adata.obs['status'].values)

    # Step 2: Prepare DataFrame
    df_p_hat = pd.DataFrame({
        'p_hat': p_hat,
        'status': adata.obs['status'].values,
        'sample_id': adata.obs['sample_id'].values
    })
    adata.obs['p_hat'] = p_hat

    # Step 3: Compute logit
    logit_p_hat = logit(p_hat)
    df_p_hat['logit_p_hat'] = logit_p_hat
    adata.obs['logit_p_hat'] = logit_p_hat

    # Step 4: KMeans clustering among cases only
    case_mask = (adata.obs['status'] == CASE_COND).values
    kmeans_case = KMeans(n_clusters=2, random_state=0).fit(
        logit_p_hat[case_mask].reshape(-1, 1)
    )

    # Identify which cluster has lower mean p_hat
    mean0 = p_hat[case_mask][kmeans_case.labels_ == 0].mean()
    mean1 = p_hat[case_mask][kmeans_case.labels_ == 1].mean()
    zero_lab_has_lower_mean = mean0 < mean1

    # Step 5: Assign new labels
    df_p_hat_clust_case = df_p_hat.copy()
    df_p_hat_clust_case['kmeans'] = 0
    df_p_hat_clust_case.loc[case_mask, 'kmeans'] = [
        1 if x == int(zero_lab_has_lower_mean) else 0
        for x in kmeans_case.labels_
    ]

    # Step 6: Plot logit(p_hat)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    sns.histplot(x=logit_p_hat, hue='status', data=df_p_hat, ax=ax1)
    ax1.set_title(f'logit(p_hat), original labels (residual PCs)')
    sns.histplot(x=logit_p_hat, hue='kmeans', data=df_p_hat_clust_case, ax=ax2)
    ax2.set_title(f'logit(p_hat), new labels (residual PCs)')
    plt.show()

    # Step 7: Plot raw p_hat
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    sns.histplot(x=p_hat, hue='status', data=df_p_hat, ax=ax1)
    ax1.set_title(f'p_hat, original labels (residual PCs)')
    sns.histplot(x=p_hat, hue='kmeans', data=df_p_hat_clust_case, ax=ax2)
    ax2.set_title(f'p_hat, new labels (residual PCs)')
    plt.show()

    return p_hat, df_p_hat_clust_case['kmeans']



# Load the merged data
adata = sc.read_h5ad("/home/delaram/LabelCorrection/data_PSC/PSC_merged.h5ad")

######### Regressing out PC2 from the expression matrix #########

# Step 1: Extract dense expression matrix
if scipy.sparse.issparse(adata.X):
    X = adata.X.A  # convert to dense
else:
    X = adata.X.copy()

# Step 2: Extract PC2 and build design matrix
pc2 = adata.obsm["X_pca"][:, 1].reshape(-1, 1)  # PC2 is 2nd component
design = np.hstack([np.ones_like(pc2), pc2])    # add intercept

# Step 3: Regress out PC2 (OLS solution)
beta_hat = np.linalg.lstsq(design, X, rcond=None)[0]  # shape (2, n_genes)
fitted = design @ beta_hat                            # predicted expression
residuals = X - fitted                                # get residuals

# Step 4: Store residuals in .layers
adata.layers["residual_PC2"] = residuals.astype(np.float32)

# Step 5: Run PCA on residuals
sc.pp.scale(adata, layer="residual_PC2", max_value=10)
sc.tl.pca(adata, layer="residual_PC2", svd_solver='arpack')

# Step 6: Visualize PCA
sc.pl.pca_scatter(adata, color="sample_id", title="PCA after regressing out PC2", size=20)
sc.pl.pca_scatter(adata, color="disease", title="PCA after regressing out PC2", size=20)



# Double check you're using the PCA of residuals
assert adata.uns["pca"]["params"]["layer"] == "residual_PC2", "PCA not run on residuals!"


# Assign all PCs to adata.obs
for pc_idx in range(adata.obsm["X_pca"].shape[1]):
    adata.obs[f"PC{pc_idx+1}"] = adata.obsm["X_pca"][:, pc_idx]

# Now split by sample AFTER assigning PCs
adata_by_sample = {
    sid: adata[adata.obs["sample_id"] == sid].copy()
    for sid in adata.obs["sample_id"].unique().tolist()
}
# Prepare sample-wise views
adata_by_sample = {
    sid: adata[adata.obs["sample_id"] == sid].copy()
    for sid in adata.obs["sample_id"].unique().tolist()
}

for pc_idx in range(20):
    pc_name = f"PC{pc_idx+1}"
    adata.obs[pc_name] = adata.obsm["X_pca"][:, pc_idx]

    # Compute global color scale
    vals = adata.obs[pc_name].values
    vmin = np.quantile(vals, 0.01)
    vmax = np.quantile(vals, 0.99)

    # Plot all samples in 2x4 grid
    fig, axs = plt.subplots(2, 4, figsize=(16, 8))
    axs = axs.flatten()

    for i, (sid, a) in enumerate(adata_by_sample.items()):
        spatial = a.obsm["spatial"]
        pc_vals = a.obs[pc_name].clip(lower=vmin, upper=vmax)

        axs[i].scatter(spatial[:, 0], spatial[:, 1],
                       c=pc_vals, cmap="coolwarm", s=10, vmin=vmin, vmax=vmax)
        axs[i].set_title(sid, fontsize=10)
        axs[i].axis("off")

    # Shared colorbar
    cbar_ax = fig.add_axes([0.92, 0.25, 0.015, 0.5])
    sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(vmin=vmin, vmax=vmax))
    fig.colorbar(sm, cax=cbar_ax, label=pc_name)

    plt.suptitle(f"Residual {pc_name} across spatial coordinates", fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    plt.show()

###############################################################
######### Correlation between original and residual PCs #########

# Step 1: Run PCA on original and residuals
sc.pp.scale(adata)
sc.tl.pca(adata, n_comps=50, svd_solver='arpack')
X_pca_original = adata.obsm["X_pca"].copy()  # Save original

sc.pp.scale(adata, layer="residual_PC2")
sc.tl.pca(adata, layer="residual_PC2", n_comps=50, svd_solver='arpack')
X_pca_residual = adata.obsm["X_pca"].copy()  # Save residual

# Step 2: Compute correlation matrix
num_pc = 15
corr_matrix = np.corrcoef(X_pca_original.T[:num_pc], X_pca_residual.T[:num_pc])
corr_df = pd.DataFrame(
    corr_matrix[:num_pc, num_pc:],  # shape: (num_pc, num_pc)
    index=[f"PC{i+1}_orig" for i in range(num_pc)],
    columns=[f"PC{i+1}_resid" for i in range(num_pc)]
)

# Step 3: Plot
plt.figure(figsize=(12, 9))
sns.heatmap(
    corr_df,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    square=True,
    linewidths=0.5,
    cbar_kws={"label": "Pearson Correlation"},
    annot_kws={"size": 9}
)
plt.xticks(rotation=45, ha="right", fontsize=11)
plt.yticks(rotation=0, fontsize=11)
plt.title("Correlation Between Original and Residual PCs", fontsize=16, weight='bold')
plt.tight_layout()
plt.show()


######################## Identifying informative PCs using logistic regression ########################

# 1. Extract features (PCs) and target (disease label)
X = X_pca_residual
y_raw = adata.obs['disease'].values

# 2. Encode the disease label into binary (0/1)
le = LabelEncoder()
y = le.fit_transform(y_raw)  # e.g., "normal"=0, "PSC"=1

# 3. Fit logistic regression
model = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000)
model.fit(X, y)


# 4. Extract feature importance (absolute value of coefficients)
coefs = model.coef_[0]
pc_labels = [f'PC{i+1}' for i in range(50)]
importance_df = pd.DataFrame({
    'PC': pc_labels,
    'Coefficient': coefs,
    'AbsCoefficient': np.abs(coefs)
})

# 5. Sort by importance and plot
importance_df_sorted = importance_df.sort_values(by='AbsCoefficient', ascending=False)

plt.figure(figsize=(10, 5))
sns.barplot(x='PC', y='AbsCoefficient', data=importance_df_sorted.head(15), palette="viridis")
plt.title("Top 15 Most Informative PCs for Predicting Disease")
plt.ylabel("Absolute Logistic Regression Coefficient")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Step 1: Extract top 5 informative PCs
top5_pcs = importance_df_sorted['PC'].head(5).tolist()
print("Top 5 PCs:", top5_pcs)

# Step 2: Get indices of those PCs
top5_indices = [int(pc[2:]) - 1 for pc in top5_pcs]  # Convert 'PC2' → 1, etc.

# Step 3: Extract PC values and sample IDs
top5_df = pd.DataFrame(adata.obsm["X_pca"][:, top5_indices], columns=top5_pcs)
top5_df["sample_id"] = adata.obs["sample_id"].values

# Step 4: Melt to long format
top5_long = top5_df.melt(id_vars="sample_id", var_name="PC", value_name="value")


plt.figure(figsize=(12, 6))
sns.violinplot(
    x="PC", y="value", hue="sample_id",
    data=top5_long, inner="box", palette="Set2"
)
plt.title("Distribution of Top 5 Informative PCs Across Samples")
plt.legend(title="Sample ID", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()

# Step 1: Extract top 5 informative PCs
top5_pcs = importance_df_sorted['PC'].head(5).tolist()
print("Top 5 PCs:", top5_pcs)

top5_indices = [int(pc[2:]) - 1 for pc in top5_pcs]  # Convert 'PC2' → 1, etc.

# Step 3: Extract PC values and sample IDs
top5_df = pd.DataFrame(adata.obsm["X_pca"][:, top5_indices], columns=top5_pcs)
top5_df["sample_id"] = adata.obs["sample_id"].values

# Step 4: Melt to long format
top5_long = top5_df.melt(id_vars="sample_id", var_name="PC", value_name="value")

plt.figure(figsize=(12, 6))
sns.violinplot(
    x="PC", y="value", hue="sample_id",
    data=top5_long, inner="box", palette="Set2"
)
plt.title("Distribution of Top 5 Informative PCs Across Samples")
plt.legend(title="Sample ID", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()



##################################################################
########### Running HiDDEN on the residuals ######################

from hiddensc import models
# Recompute PCA on residual expression
adata.obsm["X_pca_residual"] = X_pca_residual
adata.obsm["X_pca"] = X_pca_residual

## run the model.py script within the hiddensc package
num_pcs, ks, ks_pval = determine_pcs_heuristic_ks_residual(adata=adata, 
                                                                  orig_label="binary_label", 
                                                                  max_pcs=60)
optimal_num_pcs_ks = num_pcs[np.argmax(ks)]

plt.figure(figsize=(4, 2))
plt.scatter(num_pcs, ks, s=10, c='k', alpha=0.5)
plt.axvline(x = optimal_num_pcs_ks, color = 'r', alpha=0.2)
plt.scatter(optimal_num_pcs_ks, ks[np.argmax(ks)], s=20, c='r', marker='*', edgecolors='k')
plt.xticks(np.append(np.arange(0, 60, 15), optimal_num_pcs_ks), fontsize=18)
plt.xlabel('Number of PCs')
plt.ylabel('KS')
plt.show()

# Select optimal number of PCs
num_pcs, ks, _ = models.determine_pcs_heuristic_ks(adata, 
                                                   orig_label="binary_label",
                                                    max_pcs=60)
optimal_num_pcs = num_pcs[np.argmax(ks)]
print(f"Optimal number of PCs: {optimal_num_pcs}")

# Run HiDDEN
adata.obsm["X_pca"] = adata.obsm["X_pca_residual"][:, :optimal_num_pcs]

NUM_PCS = optimal_num_pcs_ks
adata.obs['status'] = adata.obs['binary_label'].astype('int').values
## 0=False=normal=control
## 1=True=PSC=case
p_hat, new_labels = PCA_logistic_kmeans_residual(adata, num_pcs=NUM_PCS)

df_combined = adata.obs.copy()

# Add p_hat and new_labels as new columns
df_combined['p_hat'] = p_hat
df_combined['new_label'] = new_labels.values  # make sure it's aligned and has the right length

# Show shape and a preview
print(df_combined.shape)
df_combined.head()

adata.obs['new_label'] = new_labels.values
adata.obs['new_label'] = adata.obs['new_label'].astype('category')
adata.obs['p_hat'] = p_hat
adata.obs['p_hat'] = adata.obs['p_hat'].astype('float32')

sample_ids = adata.obs['sample_id'].unique().tolist()
adata_by_sample = {sample_id: adata[adata.obs['sample_id'] == sample_id].copy()
                   for sample_id in sample_ids}

def plot_spatial_p_hat(a, sample_id):
    spatial = a.obsm["spatial"]
    p_hat = a.obs["p_hat"].values

    # Optionally cap extreme values
    upper = np.quantile(p_hat, 0.99)
    p_hat = np.minimum(p_hat, upper)

    plt.figure(figsize=(5, 5))
    sc = plt.scatter(spatial[:, 0], spatial[:, 1], c=p_hat, cmap="viridis", s=10)
    plt.axis("equal")
    plt.title(f"HiDDEN prediction – {sample_id}", fontsize=14)
    plt.colorbar(sc, label="p_hat")
    plt.xticks([]); plt.yticks([])
    plt.tight_layout()
    plt.show()


for i in range(8):
    plot_spatial_p_hat(adata_by_sample[sample_ids[i]], sample_ids[i])


############# plotting all samples in a grid
# Get global 99th percentile for consistent color scale
p_hat_all = adata.obs["p_hat"].values
vmax = np.quantile(p_hat_all, 0.99)

# Plot 2x4 grid
fig, axs = plt.subplots(2, 4, figsize=(16, 8))
axs = axs.flatten()

for i, sid in enumerate(sample_ids):
    a = adata_by_sample[sid]
    spatial = a.obsm["spatial"]
    p_hat = a.obs["p_hat"].clip(upper=vmax)

    axs[i].scatter(spatial[:, 0], spatial[:, 1], c=p_hat, cmap="viridis", s=10, vmin=0, vmax=vmax)
    axs[i].set_title(sid, fontsize=10)
    axs[i].axis("off")

# Add shared colorbar
cbar_ax = fig.add_axes([0.92, 0.25, 0.015, 0.5])
sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=0, vmax=vmax))
fig.colorbar(sm, cax=cbar_ax, label="p_hat")

plt.suptitle("HiDDEN predictions (p_hat) across spatial coordinates", fontsize=16)
plt.tight_layout(rect=[0, 0, 0.9, 0.95])
plt.show()



df_violin = adata.obs[["sample_id", "p_hat"]].copy()
plt.figure(figsize=(10, 5))
sns.violinplot(
    x="sample_id", y="p_hat", hue="sample_id", data=df_violin,
    palette="Set2", density_norm="width", inner=None, legend=False
)
sns.boxplot(
    x="sample_id", y="p_hat", data=df_violin,
    color="white", width=0.1, fliersize=0
)
plt.xticks(rotation=45, ha="right")
plt.title("Distribution of p_hat per sample")
plt.tight_layout()
plt.show()
