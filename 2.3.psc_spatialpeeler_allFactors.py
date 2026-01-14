import os
import sys
root_path = os.path.abspath('./..')
sys.path.insert(0, root_path)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.sparse as sp
import hiddensc
from hiddensc import utils, vis
import scanpy as sc
import scvi
import anndata
from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.simplefilter("ignore", category=ConvergenceWarning)
from scipy.sparse import issparse
from functools import reduce

from SpatialPeeler import helpers as hlps
from SpatialPeeler import case_prediction as cpred
from SpatialPeeler import plotting as plot
from SpatialPeeler import gene_identification as gid

from sklearn.cluster import KMeans

import statsmodels.api as sm
from scipy.special import logit
import pickle
RAND_SEED = 28
CASE_COND = 1
np.random.seed(RAND_SEED)
utils.set_random_seed(utils.RANDOM_SEED)
utils.print_module_versions([sc, anndata, scvi, hiddensc])
vis.visual_settings()


# file_name = '/home/delaram/SpatialPeeler/Data/PSC_liver/PSC_NMF_30.h5ad'
#file_name = '/home/delaram/SpatialPeeler/Data/PSC_liver/PSC_NMF_30_varScale_2000HVG.h5ad'
#file_name = '/home/delaram/SpatialPeeler/Data/PSC_liver/PSC_NMF_30_varScale_2000HVG_NMF10.h5ad'
#file_name = '/home/delaram/SpatialPeeler/Data/PSC_liver/PSC_NMF_30_revLog_varScale_2000HVG_NMF10.h5ad'
file_name = '/home/delaram/SpatialPeeler/Data/PSC_liver/PSC_NMF_10_varScale_2000HVG_filtered.h5ad'


adata = sc.read_h5ad(file_name)

sample_ids = adata.obs['sample_id'].unique().tolist()
# Create a dictionary splitting the merged data by sample
adata_by_sample = {
    sid: adata[adata.obs['sample_id'] == sid].copy()
    for sid in adata.obs['sample_id'].unique()
}
sample_ids = list(adata_by_sample.keys())


sid=sample_ids[0]
plot.plot_spatial_nmf(adata_by_sample[sid], 3, sample_id=sid, figsize=(6, 5))



for i in range(len(sample_ids)): #len(sample_ids)
    sid = sample_ids[i]
    print(f"Plotting for sample: {sid}")
    plot.plot_spatial_nmf(adata_by_sample[sid], 0, sample_id=sid, figsize=(6, 5))


total_factors = adata.obsm["X_nmf"].shape[1]

num_factors = 10
nmf_factors = adata.obsm['X_nmf'][:, :num_factors]
nmf_df = pd.DataFrame(nmf_factors, 
                      columns=[f'NMF{i+1}' for i in range(nmf_factors.shape[1])])

#nmf_df['sample_id'] = adata.obs['sample_id'].values
#nmf_df['sex'] = adata.obs['sex'].values
#nmf_df['donor_id'] = adata.obs['donor_id'].values
nmf_df['sample_id'] = adata.obs['disease'].values

nmf_long = nmf_df.melt(id_vars='sample_id', 
                       var_name='Factor', 
                       value_name='Score')
plt.figure(figsize=(16, 6))
sns.violinplot(x="Factor", y="Score", hue="sample_id", data=nmf_long, 
               inner="box", palette="Set2")
plt.title("Distribution of NMF Across Samples")
plt.legend(title="Sample ID", bbox_to_anchor=(1.05, 1), 
           loc='upper left')
plt.tight_layout()
plt.show()

total_factors = adata.obsm["X_nmf"].shape[1]
nmf_factors = adata.obsm['X_nmf'][:, :8]
nmf_df = pd.DataFrame(nmf_factors, 
                      columns=[f'NMF{i+1}' for i in range(nmf_factors.shape[1])])
nmf_df['sample_id'] = adata.obs['sample_id'].values
nmf_long = nmf_df.melt(id_vars='sample_id', 
                       var_name='Factor', 
                       value_name='Score')


plt.figure(figsize=(16, 6))
sns.violinplot(x="Factor", y="Score", hue="sample_id", data=nmf_long, inner="box", palette="Set2")
plt.title("Distribution of NMF Across Samples")
plt.legend(title="Sample ID", bbox_to_anchor=(1.05, 1), 
           loc='upper left')
plt.tight_layout()
plt.show()



##################################################################
########### Running logistic regression on all the NMF factors ######################

#adata.obsm["X_nmf"] 
#adata.uns["nmf_components"]

optimal_num_pcs_ks = 10#30
print(f"Optimal number of PCs/KS: {optimal_num_pcs_ks}")
adata.obsm["X_pca"] = adata.obsm["X_nmf"][:, :optimal_num_pcs_ks]
adata.obs['binary_label'] = adata.obs['disease'].apply(lambda x: 1 if x == 'primary sclerosing cholangitis' else 0)
adata.obs['status'] = adata.obs['binary_label'].astype(int).values

factor_key = "X_nmf"
y = adata.obs["disease"].values
y_int = (np.asarray(y) == "primary sclerosing cholangitis").astype(int)

X = adata.obsm[factor_key]
#sub_k = 10
#X_sub = X[:, :10]
#print(X_sub.shape)
# Add intercept explicitly
X_with_intercept = sm.add_constant(X)

# Fit model using statsmodels for inference
model = sm.Logit(y_int, X_with_intercept)
result = model.fit(disp=False)

p_hat = result.predict(X_with_intercept)  # returns P(y=1|x)
coef = result.params          # includes intercept and weights
stderr = result.bse           # standard errors for each beta
pvals = result.pvalues        # p-values (Wald test)


plt.figure(figsize=(5, 5))
sns.histplot(p_hat, bins=30, kde=True)
plt.title("p_hat Distribution")
plt.xlabel("p-hat for all spots")
plt.ylabel("Count")
plt.show()

### visualize the p_hat distribution for all samples across conditions violin plot
df_p_hat = pd.DataFrame({
    'disease': adata.obs['disease'],
    'sample_id': adata.obs['sample_id'],
    'p_hat': p_hat
})

plt.figure(figsize=(10, 10))
sns.violinplot(
    y="p_hat",
    x="disease",
    data=df_p_hat,
    inner="box",
    cut=0,
    density_norm="count",
    order=["primary sclerosing cholangitis", "normal"],
    palette={"primary sclerosing cholangitis": "skyblue", "normal": "salmon"}
)
plt.title(f"p_hat Distribution for each primary sclerosing cholangitis vs control")
plt.tight_layout()
plt.show()


normal_samples = adata.obs[adata.obs['disease']=='normal']['sample_id'].unique().tolist()
psc_samples = adata.obs[adata.obs['disease']=='primary sclerosing cholangitis']['sample_id'].unique().tolist()

plt.figure(figsize=(15, 10))
sns.violinplot(
    y="p_hat",
    x="sample_id",
    data=df_p_hat,
    inner="box",
    density_norm="count",
    cut=0,
    order=normal_samples + psc_samples,
    palette={sid: "skyblue" for sid in normal_samples}| {sid: "salmon" for sid in psc_samples}
)
plt.title(f"p_hat Distribution for each sample")
plt.tight_layout()
plt.show()

### visualize p-hat values on spatial maps for each sample, for non-high-expression cluster points p_hat=NA
adata.obs['p_hat'] = pd.NA
adata.obs['p_hat'] = p_hat.astype('float32')
#adata.obs['p_hat'] = adata.obs['p_hat'].astype('float32')
# Copy adata per sample for plotting
sample_ids = adata.obs['sample_id'].unique().tolist()
adata_by_sample = {
    sample_id: adata[adata.obs['sample_id'] == sample_id].copy()
    for sample_id in sample_ids
}
# Plot spatial maps for the first 8 samples
plot.plot_grid_upgrade(adata_by_sample, sample_ids, key="p_hat", from_obsm=False, 
    title_prefix=f"p-hat predictions",  
    figsize=(43, 20), fontsize=45, dot_size=60) #figsize=(42, 30), fontsize=45


##################################################################

##################################################################
CASE_COND_NAME = "primary sclerosing cholangitis"
p_hat_case = p_hat[adata.obs['disease'] == CASE_COND_NAME]
p_hat_case_df = pd.DataFrame(p_hat_case, columns=['p_hat'])

plt.figure(figsize=(5, 5))
sns.histplot(p_hat_case_df['p_hat'], bins=30, kde=True)
plt.title(f"p-hat distribution for case samples")
plt.xlabel("p-hat for case samples")
plt.ylabel("Count")
plt.show()
    
kmeans = KMeans(n_clusters=2, random_state=RAND_SEED)
kmeans.fit(p_hat_case.reshape(-1, 1))

### add cluster information to adata - to the case observations with the correct order
mask_case = (adata.obs['disease'].values == CASE_COND_NAME)

# Sanity check: lengths must match
print(mask_case.sum() == len(kmeans.labels_))

# Remap labels so that 1 = higher p-hat, 0 = lower p-hat
centers = kmeans.cluster_centers_.ravel()
order = np.argsort(centers)               # [low_center_label, high_center_label]
remap = {order[0]: 0, order[1]: 1}
labels_remapped = np.vectorize(remap.get)(kmeans.labels_).astype(int)

# Calculate the threshold as the midpoint between the centroids
threshold = np.mean(centers)
print(f"Centroids: {centers.flatten()}")
print(f"Binarization Threshold: {threshold}")

### add the threshold to the density plot of case-phat values
plt.figure(figsize=(5, 5))
sns.histplot(p_hat_case_df['p_hat'], bins=30, kde=True)
plt.axvline(x=threshold, color="r", linestyle="--")
plt.title(f"p-hat distribution for case samples")
plt.xlabel("p-hat for case samples")
plt.ylabel("Count")
plt.show()

# Create an obs column name that encodes which factor you clustered 
obs_col = "phat_cluster"
adata.obs["phat"] = p_hat

# initialize and fill only case rows
adata.obs[obs_col] = np.nan
adata.obs.loc[mask_case, obs_col] = labels_remapped  # 0/1 for cases

# map to desired labels
adata.obs[obs_col] = adata.obs[obs_col].map({0: "case_0", 1: "case_1"})
adata.obs[obs_col] = adata.obs[obs_col].fillna("control")

# enforce x-axis order
x_order = ["control", "case_0", "case_1"]
adata.obs[obs_col] = pd.Categorical(adata.obs[obs_col], categories=x_order, ordered=True)

plt.figure(figsize=(6, 5))
sns.violinplot(
    x=obs_col, y="phat", data=adata.obs,
    order=x_order,
    density_norm="count",
    inner="box"  # optional, matches your prior style better than default
)
plt.axhline(y=threshold, color="r", linestyle="--")
plt.title("Violin plot of p-hat scores")
plt.xlabel("Cluster")
plt.ylabel("p-hat")
plt.show()

### split the obs_col values equal to 'case_nan' into 'control_0' and 'control_1' based on the threshold derived from case samples
def assign_control_label(row):
    if row[obs_col].startswith('case_0') or row[obs_col].startswith('case_1'):
        return row[obs_col]
    else:
        if row['phat'] >= threshold:
            return 'control_1'
        else:
            return 'control_0'

obs_col_v2 = "phat_cluster"
# assign labels
adata.obs[obs_col_v2] = adata.obs.apply(assign_control_label, axis=1)
x_order = ['control_0', 'control_1', 'case_0', 'case_1']

# make ordered categorical (DO NOT astype(str) afterward)
adata.obs[obs_col_v2] = pd.Categorical(
    adata.obs[obs_col_v2],
    categories=x_order,
    ordered=True
)
# sanity check
print(adata.obs[obs_col_v2].dtype)
print(adata.obs[obs_col_v2].value_counts().reindex(x_order, fill_value=0))


plt.figure(figsize=(8, 5))
sns.violinplot(
    x=obs_col_v2,
    y='phat',
    data=adata.obs,
    density_norm="count",
    order=x_order,
    inner="box"   # optional but consistent with your other plot
)
plt.axhline(y=threshold, color='r', linestyle='--')
plt.title("Violin plot of p-hat scores")
plt.xlabel("Cluster")
plt.ylabel("p-hat")
plt.show()

### visualize the table of counts per cluster
cluster_counts = adata.obs[obs_col_v2].value_counts().reindex(x_order, fill_value=0)
## bar plot
plt.figure(figsize=(8, 5))
sns.barplot(x=cluster_counts.index, y=cluster_counts.values)
plt.title("Cell counts per cluster")
plt.xlabel("Cluster")
plt.ylabel("Number of cells")
plt.show()





### I added this mapping to resolve the color issue in plotting the clusters
###############################################################################
# Map string labels to numeric codes (ensure no NaNs remain)
code_map = {'control_0': 0, 'control_1': 1, 'case_0': 2, 'case_1': 3}
obs_col_num = obs_col + "_num"
adata.obs[obs_col_num] = adata.obs[obs_col].map(code_map).astype(float)
# Sanity check: if you see NaN here, some labels didn't match code_map exactly
print("Unique numeric codes:", adata.obs[obs_col_num].unique())
assert not pd.isna(adata.obs[obs_col_num]).any(), "Unmapped labels → extend code_map."
# Numeric palette (keys must match the codes you just wrote)
palette_num = {
    0: "#54A24B",  # control_0
    1: "#E45756",  # control_1
    2: "#4C78A8",  # case_0
    3: "#F58518",  # case_1
}   

sample_ids = adata.obs['sample_id'].unique().tolist()
adata_by_sample = {
    sample_id: adata[adata.obs['sample_id'] == sample_id].copy()
    for sample_id in sample_ids
}

plot.plot_grid_upgrade(
    adata_by_sample, sample_ids, key=obs_col_num,
    title_prefix=f"Clusters",
    from_obsm=False, discrete=True,
    palette=palette_num,
    figsize=(43, 20), fontsize=45, dot_size=60
)

#plot.plot_grid_upgrade(adata_by_sample, sample_ids, key=obs_col,
#        title_prefix="Clusters (factor "+str(factor_idx+1)+")", 
#        from_obsm=False, discrete=True,
#        dot_size=2, figsize=(25, 10),
#        palette={'case_0': "#4C78A8", 'case_1': "#F58518", 
#                 'control_0': "#54A24B", 'control_1': "#E45756"})
##

plot.plot_grid_upgrade(adata_by_sample, sample_ids, key='phat',
                        title_prefix="p-hat", 
                        from_obsm=False, discrete=False,
                        figsize=(43, 20), fontsize=45, dot_size=60)

