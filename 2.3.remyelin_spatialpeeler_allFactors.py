import os
import sys
from sklearn.cluster import KMeans
root_path = os.path.abspath('./..')
sys.path.insert(0, root_path)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.sparse as sp
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

import statsmodels.api as sm
from scipy.special import logit
import pickle

RAND_SEED = 28
CASE_COND = 1
np.random.seed(RAND_SEED)
#file_name = '/home/delaram/SpatialPeeler/Data/Remyelin_Slide-seq/Remyelin_NMF_30_uncropped_t3_7_PreprocV2_samplewise.h5ad'
file_name = '/home/delaram/SpatialPeeler/Data/Remyelin_Slide-seq/Remyelin_NMF_30_uncropped_t3_PreprocV2_samplewise_ALLGENES.h5ad'

adata_cropped = sc.read_h5ad('/home/delaram/SpatialPeeler/Data/Remyelin_Slide-seq/Remyelin_NMF_30.h5ad')
adata = sc.read_h5ad(file_name)
### cropped data
adata.obs['Condition'].value_counts() 
# LPC       28582
# Saline     9522
adata.obs['Animal'].value_counts()
#['M140','M261','M257','M255','M259','EG102','EG98','EG100','M254','M258','EG103','EG99']
adata.obs['Timepoint'].value_counts() # [12, 18, 3, 7]

adata.obs['sample_id'] = adata.obs['puck_id'] #orig.ident
#adata.obs['sample_id'] = adata.obs['orig.ident']
spatial = {'x': adata.obs['x'].values.astype(float).tolist(), 
           'y': adata.obs['y'].values.astype(float).tolist()}
#adata.obsm["spatial"] = pd.DataFrame(spatial, index=adata.obs.index).values
adata.obs['cropped'] = adata.obs['barcode'].isin(adata_cropped.obs['bead'])


sample_ids = adata.obs['sample_id'].unique().tolist()


# Create a dictionary splitting the merged data by sample
adata_by_sample = {
    sid: adata[adata.obs['sample_id'] == sid].copy()
    for sid in adata.obs['sample_id'].unique()
}
sample_ids = list(adata_by_sample.keys())

metadata = {'sample_id': adata.obs['sample_id'].values.tolist(),
            'Timepoint': adata.obs['Timepoint'].values.tolist(),
            'Animal': adata.obs['Animal'].values.tolist(),
            'Condition': adata.obs['Condition'].values.tolist()}
metadata_df = pd.DataFrame(metadata, index=adata.obs.index)
metadata_df = metadata_df.reset_index(drop=True)
duplicate_rows_mask = metadata_df.duplicated()
metadata_df = metadata_df[~duplicate_rows_mask]
print(metadata_df)
saline_samples = metadata_df[metadata_df['Condition'] == 'Saline']['sample_id'].tolist()
lpc_samples = metadata_df[metadata_df['Condition'] == 'LPC']['sample_id'].tolist()
print("Saline samples: ", saline_samples)
print("LPC samples: ", lpc_samples)

# Plot spatial maps for the first 8 samples
#for i in range(min(8, len(sample_ids))):
#    plot_spatial_p_hat(adata_by_sample[sample_ids[i]], sample_ids[i])

plot.plot_grid_upgrade(adata_by_sample, sample_ids, key="cropped", 
               title_prefix="Cropped regions", 
               from_obsm=False, figsize=(43, 30), fontsize=45, 
               dot_size=50) #figsize=(42, 30), fontsize=45

for i in range(0, 3): #len(sample_ids)
    sid = sample_ids[i]
    print(f"Plotting for sample: {sid}")
    plot.plot_spatial_nmf(adata_by_sample[sid], 0, sample_id=sid, figsize=(10, 10))
    plot.plot_spatial_nmf(adata_by_sample[sid], 1, sample_id=sid, figsize=(10, 10))
    plot.plot_spatial_nmf(adata_by_sample[sid], 2, sample_id=sid, figsize=(10, 10))

total_factors = adata.obsm["X_nmf"].shape[1]
num_factors = 10
nmf_factors = adata.obsm['X_nmf'][:, :num_factors]
nmf_df = pd.DataFrame(nmf_factors, 
                      columns=[f'NMF{i+1}' for i in range(nmf_factors.shape[1])])

nmf_df['sample_id'] = adata.obs['Animal'].values
nmf_df['sample_id'] = adata.obs['Timepoint'].values
nmf_df['sample_id'] = adata.obs['sample_id'].values
nmf_df['sample_id'] = adata.obs['Condition'].values

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




##################################################################
########### Running logistic regression on all the NMF factors ######################

#adata.obsm["X_nmf"] 
#adata.uns["nmf_components"]

optimal_num_pcs_ks = 30
print(f"Optimal number of PCs/KS: {optimal_num_pcs_ks}")
adata.obsm["X_pca"] = adata.obsm["X_nmf"][:, :optimal_num_pcs_ks]
adata.obs['binary_label'] = adata.obs['Condition'].apply(lambda x: 1 if x == 'LPC' else 0)
adata.obs['status'] = adata.obs['binary_label'].astype(int).values

factor_key = "X_nmf"
y = adata.obs["Condition"].values
y_int = (np.asarray(y) == "LPC").astype(int)

regress_factor_1 = False
X = adata.obsm[factor_key]
print(X.shape)

if regress_factor_1:
    #### remove the first factor from X
    X = X[:, 1:]
    print(X.shape)

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
    'disease': adata.obs['Condition'],
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
    order=["Saline", "LPC"],
    palette={"Saline": "skyblue", "LPC": "salmon"}
)
plt.title(f"p_hat Distribution for each LPC vs Saline")
plt.tight_layout()
plt.show()

plt.figure(figsize=(15, 10))
sns.violinplot(
    y="p_hat",
    x="sample_id",
    data=df_p_hat,
    inner="box",
    density_norm="count",
    cut=0,
    order=saline_samples + lpc_samples,
    palette={sid: "skyblue" for sid in saline_samples}| {sid: "salmon" for sid in lpc_samples}
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
i = 0
### visualize the factor scores on spatial maps for each sample
adata.obs[f'Factor_{i+1}_score'] = adata.obsm['X_nmf'][:, i]
adata_by_sample = {
    sid: adata[adata.obs['sample_id'] == sid].copy()
    for sid in adata.obs['sample_id'].unique()
}
plot.plot_grid_upgrade(adata_by_sample, sample_ids, key=f'Factor_{i+1}_score', 
        title_prefix=f" Factor {i+1} Scores", 
        from_obsm=False, figsize=(43, 30), fontsize=45,
        dot_size=60, palette_continuous='viridis_r') #figsize=(42, 30), fontsize=45

### plot beta_i*f_i spatial maps for each sample
adata.obs[f'Factor_{i+1}_contribution'] = coef[i+1] * adata.obsm['X_nmf'][:, i]
adata_by_sample = {
    sid: adata[adata.obs['sample_id'] == sid].copy()
    for sid in adata.obs['sample_id'].unique()
}
plot.plot_grid_upgrade(adata_by_sample, sample_ids, key=f'Factor_{i+1}_contribution', 
        title_prefix=f" Factor {i+1} * coefficient", 
        from_obsm=False, figsize=(43, 30), fontsize=45,
        dot_size=60, palette_continuous='viridis_r') #figsize=(42, 30), fontsize=45

### scatter plot of factor-1 scores vs p-hat values
plt.figure(figsize=(6, 6))
sns.scatterplot(
    x=f'Factor_{i+1}_score',
    y='p_hat',
    hue='Condition',
    data=adata.obs,
    alpha=0.6,
    palette={"Saline": "skyblue", "LPC": "salmon"}
)
plt.title(f"Factor {i+1} Scores vs p-hat")
plt.xlabel(f"Factor {i+1} Score")
plt.ylabel("p-hat")
plt.legend(title="Condition")
plt.show()

### scatter plot of factor-1 scores vs factor-1*beta contribution values
plt.figure(figsize=(6, 6))
sns.scatterplot(
    x=f'Factor_{i+1}_score',
    y=f'Factor_{i+1}_contribution',
    hue='Condition',
    data=adata.obs,
    alpha=0.6,
    palette={"Saline": "skyblue", "LPC": "salmon"}
)
plt.title(f"Factor {i+1} Scores vs Factor {i+1} Contribution")
plt.xlabel(f"Factor {i+1} Score")
plt.ylabel(f"Factor {i+1} Contribution")



##################################################################
def get_num_sig_de(de_results, fdr_threshold=0.05, logfc_threshold=0.1):
        sig_de = de_results[
            (de_results['pvals_adj'] < fdr_threshold) &
            (np.abs(de_results['logfoldchanges']) > logfc_threshold)
        ]
        return sig_de.shape[0]


CASE_COND_NAME = 'LPC'
p_hat_case = p_hat[adata.obs['Condition'] == CASE_COND_NAME]
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
mask_case = (adata.obs['Condition'].values == CASE_COND_NAME)

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


    
num_sig_DE = {}
###############################################################################
### perform DE between case 0 and 1
case_1_mask = (adata.obs[obs_col] == 'case_1').values
case_0_mask = (adata.obs[obs_col] == 'case_0').values
# Subset to the two clusters
keep = case_1_mask | case_0_mask
ad = adata[keep].copy()
# Temporary 2-level group label
grp_col = "_tmp_de_group"
ad.obs[grp_col] = pd.Categorical(
    np.where(case_1_mask[keep], "case1", "case0"),
    categories=["case0", "case1"]
)
# Wilcoxon DE: case1 vs case0
sc.tl.rank_genes_groups(
    ad,
    groupby=grp_col,
    groups=["case1"],
    reference="case0",
    method="wilcoxon",
    corr_method="benjamini-hochberg",
    use_raw=False,   
    layer=None,      
    n_genes=ad.n_vars
)
de_results = sc.get.rank_genes_groups_df(ad, group="case1").rename(columns={"names": "gene"})
# score column: Z-statistic (standardized Wilcoxon rank-sum score) computed for each gene after comparing expression ranks between groups.    
gene_names = hlps.map_ensembl_to_symbol(de_results.gene.tolist(), species='mouse')
de_results['gene_name'] = de_results['gene'].map(gene_names)
print(de_results.head(20))
num_sig_DE['case1_vs_case0'] = get_num_sig_de(de_results, fdr_threshold=0.05, logfc_threshold=0.1)
print(f"Number of significant DE genes (case1 vs case0): {num_sig_DE['case1_vs_case0']}")

###############################################################################
### perform DE between cluster-1 and control+cluster-0 
case_1_mask = (adata.obs[obs_col] == 'case_1').values
case_0_mask = (adata.obs[obs_col] == 'case_0').values
control_mask = (adata.obs['Condition'].values != CASE_COND_NAME)
# Subset to the two clusters + control
keep = case_1_mask | case_0_mask | control_mask
ad = adata[keep].copy()
# Temporary 2-level group label
grp_col = "_tmp_de_group"
ad.obs[grp_col] = pd.Categorical(
    np.where(case_1_mask[keep], "case1", "other"),
    categories=["other", "case1"]
)
# Wilcoxon DE: case1 vs other
sc.tl.rank_genes_groups(
    ad,
    groupby=grp_col,
    groups=["case1"],
    reference="other",
    method="wilcoxon",
    corr_method="benjamini-hochberg",
    use_raw=False,   
    layer=None,      
    n_genes=ad.n_vars
)
de_results = sc.get.rank_genes_groups_df(ad, group="case1").rename(columns={"names": "gene"})
# score column: Z-statistic (standardized Wilcoxon rank-sum score) computed for each gene after comparing expression ranks between groups.    
gene_names = hlps.map_ensembl_to_symbol(de_results.gene.tolist(), species='mouse')
de_results['gene_name'] = de_results['gene'].map(gene_names)
print(de_results.head(20))
num_sig_DE['case1_vs_other'] = get_num_sig_de(de_results, fdr_threshold=0.05, logfc_threshold=0.1)
print(f"Number of significant DE genes (case1 vs other): {num_sig_DE['case1_vs_other']}")

###############################################################################
### perform DE between cluster-1 and control_1
case_1_mask = (adata.obs[obs_col] == 'case_1').values
control_1_mask = (adata.obs[obs_col] == 'control_1').values
# Subset to the two clusters
keep = case_1_mask | control_1_mask
ad = adata[keep].copy()
# Temporary 2-level group label
grp_col = "_tmp_de_group"
ad.obs[grp_col] = pd.Categorical(
    np.where(case_1_mask[keep], "case1", "control1"),
    categories=["control1", "case1"]
)
# Wilcoxon DE: case1 vs control1
sc.tl.rank_genes_groups(
    ad,
    groupby=grp_col,
    groups=["case1"],
    reference="control1",
    method="wilcoxon",
    corr_method="benjamini-hochberg",
    use_raw=False,
    layer=None,
    n_genes=ad.n_vars
)
de_results = sc.get.rank_genes_groups_df(ad, group="case1").rename(columns={"names": "gene"})
# score column: Z-statistic (standardized Wilcoxon rank-sum score) computed for each gene after comparing expression ranks between groups.    
gene_names = hlps.map_ensembl_to_symbol(de_results.gene.tolist(), species='mouse')
de_results['gene_name'] = de_results['gene'].map(gene_names)
print(de_results.head(20))  
num_sig_DE['case1_vs_control1'] = get_num_sig_de(de_results, fdr_threshold=0.05, logfc_threshold=0.1)
print(f"Number of significant DE genes (case1 vs control1): {num_sig_DE['case1_vs_control1']}")


###############################################################################
### perform DE between case-0 and all control
case_0_mask = (adata.obs[obs_col] == 'case_0').values
control_mask = (adata.obs['Condition'].values != CASE_COND_NAME)
# Subset to the two clusters + control
keep = case_0_mask | control_mask
ad = adata[keep].copy()
# Temporary 2-level group label
grp_col = "_tmp_de_group"
ad.obs[grp_col] = pd.Categorical(
    np.where(case_0_mask[keep], "case0", "control"),
    categories=["control", "case0"]
)
# Wilcoxon DE: case0 vs control
sc.tl.rank_genes_groups(
    ad,
    groupby=grp_col,
    groups=["case0"],
    reference="control",
    method="wilcoxon",
    corr_method="benjamini-hochberg",
    use_raw=False,   
    layer=None,      
    n_genes=ad.n_vars
)
de_results = sc.get.rank_genes_groups_df(ad, group="case0").rename(columns={"names": "gene"})
# score column: Z-statistic (standardized Wilcoxon rank-sum score) computed for each gene after comparing expression ranks between groups.    
gene_names = hlps.map_ensembl_to_symbol(de_results.gene.tolist(), species='mouse')
de_results['gene_name'] = de_results['gene'].map(gene_names)
print(de_results.head(20))
num_sig_DE['case0_vs_control'] = get_num_sig_de(de_results, fdr_threshold=0.05, logfc_threshold=0.1)
print(f"Number of significant DE genes (case0 vs control): {num_sig_DE['case0_vs_control']}")


###############################################################################
### perform DE between case-0 and all control_0
case_0_mask = (adata.obs[obs_col] == 'case_0').values
control_0_mask = (adata.obs[obs_col] == 'control_0').values
# Subset to the two clusters + control
keep = case_0_mask | control_0_mask
ad = adata[keep].copy()
# Temporary 2-level group label
grp_col = "_tmp_de_group"
ad.obs[grp_col] = pd.Categorical(
    np.where(case_0_mask[keep], "case0", "control0"),
    categories=["control0", "case0"]
)
# Wilcoxon DE: case0 vs control0
sc.tl.rank_genes_groups(
    ad,
    groupby=grp_col,
    groups=["case0"],
    reference="control0",
    method="wilcoxon",
    corr_method="benjamini-hochberg",
    use_raw=False,   
    layer=None,      
    n_genes=ad.n_vars
)
de_results = sc.get.rank_genes_groups_df(ad, group="case0").rename(columns={"names": "gene"})
# score column: Z-statistic (standardized Wilcoxon rank-sum score) computed for each gene after comparing expression ranks between groups.    
gene_names = hlps.map_ensembl_to_symbol(de_results.gene.tolist(), species='mouse')
de_results['gene_name'] = de_results['gene'].map(gene_names)
print(de_results.head(20))
num_sig_DE['case0_vs_control0'] = get_num_sig_de(de_results, fdr_threshold=0.05, logfc_threshold=0.1)
print(f"Number of significant DE genes (case0 vs control0): {num_sig_DE['case0_vs_control0']}")



###############################################################################
### perform DE between control-1 and all control_0
control_1_mask = (adata.obs[obs_col] == 'control_1').values
control_0_mask = (adata.obs[obs_col] == 'control_0').values
# Subset to the two clusters + control
keep = control_1_mask | control_0_mask
ad = adata[keep].copy()
# Temporary 2-level group label
grp_col = "_tmp_de_group"
ad.obs[grp_col] = pd.Categorical(
    np.where(control_1_mask[keep], "control1", "control0"),
    categories=["control0", "control1"]
)
# Wilcoxon DE: control1 vs control0
sc.tl.rank_genes_groups(
    ad,
    groupby=grp_col,
    groups=["control1"],
    reference="control0",
    method="wilcoxon",
    corr_method="benjamini-hochberg",
    use_raw=False,   
    layer=None,      
    n_genes=ad.n_vars
)
de_results = sc.get.rank_genes_groups_df(ad, group="control1").rename(columns={"names": "gene"})
# score column: Z-statistic (standardized Wilcoxon rank-sum score) computed for each gene after comparing expression ranks between groups.    
gene_names = hlps.map_ensembl_to_symbol(de_results.gene.tolist(), species='mouse')
de_results['gene_name'] = de_results['gene'].map(gene_names)
print(de_results.head(20))
num_sig_DE['control1_vs_control0'] = get_num_sig_de(de_results, fdr_threshold=0.05, logfc_threshold=0.1)
print(f"Number of significant DE genes (control1 vs control0): {num_sig_DE['control1_vs_control0']}")



### calculate correlation between each NMF factor loading and the num_sig_DE['control1_vs_control0']
correlations = {}
for factor_idx in range(adata.obsm['X_nmf'].shape[1]):
    
    NMF_df = pd.DataFrame({'loadings': adata.uns['nmf']['H'][factor_idx,:],
                           'genes': adata.var_names
                          })
    

    DE_df = pd.DataFrame({
        'gene': de_results['gene'],
        'score': de_results['scores']
    })

    ### merge the two dataframes on gene names and calculate correlation
    merged_df = pd.merge(NMF_df, DE_df, left_on='genes', right_on='gene')
    corr = merged_df['loadings'].corr(merged_df['score'])
    correlations[f'Factor_{factor_idx+1}'] = corr

# plot the correlations
plt.figure(figsize=(15, 5))
sns.barplot(x=list(correlations.keys()), y=list(correlations.values()), palette="magma")
plt.title("Correlation between NMF factor loadings and DE scores (control1 vs control0)")
plt.xlabel("NMF Factors")
plt.ylabel("Pearson Correlation")
plt.xticks(rotation=45)
plt.show()


###############################################################################
### visualize number of significant DE genes in different comparisons
comparisons = list(num_sig_DE.keys())
sig_de_counts = [num_sig_DE[comp] for comp in comparisons]  
plt.figure(figsize=(8, 5))
sns.barplot(x=comparisons, y=sig_de_counts, palette="viridis")
plt.title(f"Number of significant Genes")
plt.ylabel("#sig DE genes")
plt.xlabel("Comparisons")
plt.xticks(rotation=45)
plt.show()


###############################################################################

### loop through each factor and visualize the factor scores over control-0 and control-1 clusters as violin plots
num_factors = adata.obsm['X_nmf'].shape[1]
for factor_idx in range(num_factors):
    factor_scores = adata.obsm['X_nmf'][:, factor_idx]
    adata.obs[f'Factor_{factor_idx+1}_score'] = factor_scores

    plt.figure(figsize=(8, 5))
    sns.violinplot(
        x=obs_col_v2,
        y=f'Factor_{factor_idx+1}_score',
        data=adata.obs,
        density_norm="count",
        order=x_order,
        inner="box"   # optional but consistent with your other plot
    )
    plt.title(f"Violin plot of NMF Factor {factor_idx+1} scores")
    plt.xlabel("Cluster")
    plt.ylabel(f"NMF Factor {factor_idx+1} Score")
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


###############################################################################

sample_ids = adata.obs['sample_id'].unique().tolist()
adata_by_sample = {
    sample_id: adata[adata.obs['sample_id'] == sample_id].copy()
    for sample_id in sample_ids
}

plot.plot_grid_upgrade(
    adata_by_sample, sample_ids, key=obs_col_num,
    title_prefix=f"Clusters",
    from_obsm=False, discrete=True,
    dot_size=2, figsize=(25, 10),
    palette=palette_num
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
                        dot_size=2, figsize=(25, 10))



