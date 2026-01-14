import os
import pandas as pd
import anndata
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
from sklearn.decomposition import NMF
import functools

from SpatialPeeler import importing as imp
from SpatialPeeler import helpers as hlps
from SpatialPeeler import plotting as plot
#from scipy.stats import mannwhitneyu
from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.simplefilter("ignore", category=ConvergenceWarning)

from scipy import io
from scipy.sparse import csr_matrix


RAND_SEED = 28
CASE_COND = 1
np.random.seed(RAND_SEED)
utils.set_random_seed(utils.RANDOM_SEED)
utils.print_module_versions([sc, anndata, scvi, hiddensc])
vis.visual_settings()
root_dir = '/home/delaram/SpatialPeeler/Data/Remyelin_Slide-seq/'

#######################################
#### importing the uncropped data  ####
base = "/home/delaram/SpatialPeeler/Data/Remyelin_Slide-seq/all_final_cropped_pucks_standardpipeline"
obs = pd.read_csv(f"{base}/all_final_cropped_pucks_standardpipeline_metadata.csv", index_col=0)

metadata = {'sample_id': obs['orig.ident'].values.tolist(),
            'Timepoint': obs['Timepoint'].values.tolist(),
            'Animal': obs['Animal'].values.tolist(),
            'Condition': obs['Condition'].values.tolist()}

#metadata['puck_id'] =  [f"2023-08-25_{name}" for name in metadata['sample_id']]   
metadata['puck_id'] = metadata['sample_id']
metadata_df = pd.DataFrame(metadata)
metadata_df = metadata_df.reset_index(drop=True)
duplicate_rows_mask = metadata_df.duplicated()
metadata_df = metadata_df[~duplicate_rows_mask]

LPC_t18_mask = (metadata_df['Timepoint'] == 18) & (metadata_df['Condition'] == 'LPC')
sample_id_LPC_t18 = metadata_df[LPC_t18_mask]['sample_id'].values.tolist()

LPC_t3_mask = (metadata_df['Timepoint'] == 3) & (metadata_df['Condition'] == 'LPC')
sample_id_LPC_t3 = metadata_df[LPC_t3_mask]['sample_id'].values.tolist()

LPC_t7_mask = (metadata_df['Timepoint'] == 7) & (metadata_df['Condition'] == 'LPC')
sample_id_LPC_t7 = metadata_df[LPC_t7_mask]['sample_id'].values.tolist()

Saline_t3_7_mask = (metadata_df['Timepoint'].isin([3,7])) & (metadata_df['Condition'] == 'Saline')
sample_id_Saline_t3_7 = metadata_df[Saline_t3_7_mask]['sample_id'].values.tolist()

LPC_mask = LPC_t7_mask
print(LPC_mask)
#LPC_mask = metadata_df['Condition'] == 'LPC'
sample_id_LPC = metadata_df[LPC_mask]['sample_id'].values.tolist()

########################################
### for the control samples, we use timepoints 12 and 18, as timepoints 3 and 7 are sparse, leading to technical variation
Saline_t12_18_mask = (metadata_df['Timepoint'].isin([12,18])) & (metadata_df['Condition'] == 'Saline')
sample_id_Saline_t12_18 = metadata_df[Saline_t12_18_mask]['sample_id'].values.tolist()

#Saline_t18_12_7_mask = (metadata_df['Timepoint'].isin([18,12,7])) & (metadata_df['Condition'] == 'Saline')
#sample_id_Saline_t18_12_7 = metadata_df[Saline_t18_12_7_mask]['sample_id'].values.tolist()

sample_id_merged = sample_id_LPC + sample_id_Saline_t12_18


print(metadata_df[metadata_df['sample_id'].isin(sample_id_merged)])
print(metadata_df.head())
print(metadata_df.shape)

adata_merged = imp.load_all_slide_seq_data(root_dir, normalize_log1p=False, scale_features=True)
adata_merged = adata_merged[adata_merged.obs['puck_id'].isin(sample_id_merged)]
adata_merged.obs['puck_id'].value_counts()
#adata_merged.X = adata_merged.layers['lognorm'].copy() # Use preprocessed layer (no additional log1p needed)
#sc.pp.highly_variable_genes(adata_merged, n_top_genes=2000, subset=True)


### add medata to adata object based on the puck_id column
adata_merged.obs = adata_merged.obs.merge(metadata_df, on='puck_id', how='left')
adata_merged.obs['puck_id'].value_counts()
print(adata_merged.obs.head())

########################################



##################################################################
################# Importing the cropped data  ####################
import_cropped = False
if import_cropped:
    base = "/home/delaram/SpatialPeeler/Data/Remyelin_Slide-seq/all_final_cropped_pucks_standardpipeline"
    #scaled_counts = io.mmread(f"{base}/all_final_cropped_pucks_standardpipeline_scaled_counts.mtx").tocsr()
    merged_counts = io.mmread(f"{base}/all_final_cropped_pucks_standardpipeline_counts_merged.mtx").tocsr()
    obs_names = pd.read_csv(f"{base}/all_final_cropped_pucks_standardpipeline_barcodes.tsv", header=None)[0].values
    var_names = pd.read_csv(f"{base}/all_final_cropped_pucks_standardpipeline_features.tsv", header=None)[0].values
    obs = pd.read_csv(f"{base}/all_final_cropped_pucks_standardpipeline_metadata.csv", index_col=0)

    all(obs.index == obs_names) ### Check if obs.index matches obs_names
    adata_merged = sc.AnnData(X=merged_counts.T, ### X should be(cells x genes)
                    obs= obs)
    adata_merged.obs_names = obs_names
    adata_merged.var_names = var_names
##################################################################



############################################
# cNMF-style preprocessing (pooled over pucks) - each sample is unit-variance scaled while importing (load_all_slide_seq_data with scale_features=True)
############################################
# ---- 0) Ensure we have raw counts to use for cNMF ----
# cNMF is intended to run on (non-log) counts with gene-wise scaling, not log1p-normalized X.
# Prefer: counts stored in a layer (common in many pipelines).
# If your loader stores raw counts somewhere else, point to it here.

if "counts" in adata_merged.layers:
    X_counts = adata_merged.layers["counts"]
elif "raw_counts" in adata_merged.layers:
    X_counts = adata_merged.layers["raw_counts"]
elif hasattr(adata_merged, "raw") and adata_merged.raw is not None:
    # Sometimes raw stores log-normalized; only use if you're sure it is counts.
    X_counts = adata_merged.raw.X
else:
    raise ValueError(
        "Could not find raw counts. cNMF-style preprocessing should use raw counts.\n"
        "Reload with normalize_log1p=False, or store counts in adata.layers['counts']."
    )

# Make a working copy that uses counts in .X
adata_cnmf = adata_merged.copy()
adata_cnmf.X = X_counts.copy()

# 2) basic gene filtering 
# Paper: remove genes detected in fewer than ~1/500 cells: min_cells = max(1, n_cells//500)
min_cells = max(1, adata_cnmf.n_obs // 500)
sc.pp.filter_genes(adata_cnmf, min_cells=min_cells)

# (remove extremely low-depth spots if needed.
# Paper used UMI threshold for cells
### check how many spots have UMI total less than 200
min_counts = 100
adata_cnmf.obs['total_counts'] = np.array(adata_cnmf.X.sum(axis=1)).flatten()
print("Number of spots with total counts < "+str(min_counts)+":", np.sum(adata_cnmf.obs['total_counts'] < min_counts))
### filter out spots with less than 1000 UMI counts
sc.pp.filter_cells(adata_cnmf, min_counts=min_counts)  # adjust if you want a hard cutoff

USE_ALL_GENES = False
if USE_ALL_GENES:
    pass  # keep all genes after min_cells filtering
else:
    # 3) HVG selection across pooled data, but batch-aware to avoid puck dominance ----
    sc.pp.highly_variable_genes(
        adata_cnmf,
        n_top_genes=2000,
        batch_key="puck_id",     # use puck_id as the "sample" grouping
        flavor="seurat_v3",
        subset=True
    )

print(adata_cnmf.shape)

# 4) sanity checks for NMF readiness ----
# Must be non-negative (or at least not heavily negative). If you see negatives, something upstream centered.
if sp.issparse(adata_cnmf.X):
    min_val = adata_cnmf.X.min()
else:
    min_val = np.min(adata_cnmf.X)

print("cNMF matrix shape (cells/spots x genes):", adata_cnmf.shape)
print("Min value in cNMF matrix:", float(min_val))


### Apply NMF
'''
# Initial preprocessing method -  Normalize and log-transform the merged dataset only - didn't work well
sc.pp.normalize_total(adata_merged, target_sum=1e4)
sc.pp.log1p(adata_merged)
adata_merged.layers["lognorm"] = adata_merged.X.copy()
sc.pp.highly_variable_genes(adata_merged, n_top_genes=2000, flavor='seurat_v3', subset=True, layer='lognorm')
sc.pp.scale(adata_merged, max_value=10, zero_center=True, layer='lognorm')
'''
# Run NMF
# Note: NMF requires dense input, so we convert the sparse matrix to dense.
# You can also use the `init='nndsvda'` option for better initialization
# and set `max_iter` to a higher value for convergence.

n_factors = 30
nmf_model = NMF(
    n_components=n_factors,
    init="nndsvda",
    random_state=RAND_SEED,
    max_iter=1000,
    solver="cd",      # default; works well
)

X = adata_cnmf.X  # use the preprocessed matrix (counts -> HVG -> scale no-center)
W = nmf_model.fit_transform(X)      # (cells/spots x factors)
H = nmf_model.components_           # (factors x genes)

adata_cnmf.obsm["X_nmf"] = W
adata_cnmf.uns["nmf"] = {
    "H": H,
    "genes": adata_cnmf.var_names.to_list(),
    "n_factors": n_factors,
    "model_params": nmf_model.get_params(),
}

### save adata object with NMF results  
# file_name = '/home/delaram/SpatialPeeler/Data/Remyelin_Slide-seq/Remyelin_NMF_30.h5ad'
# file_name = '/home/delaram/SpatialPeeler/Data/Remyelin_Slide-seq/Remyelin_NMF_30_uncropped.h5ad'
#file_name = '/home/delaram/SpatialPeeler/Data/Remyelin_Slide-seq/Remyelin_NMF_30_uncropped_SampleWiseNorm.h5ad'
#file_name = '/home/delaram/SpatialPeeler/Data/Remyelin_Slide-seq/Remyelin_NMF_30_uncropped_t3_7.h5ad'
#file_name = '/home/delaram/SpatialPeeler/Data/Remyelin_Slide-seq/Remyelin_NMF_30_uncropped_t18.h5ad'
#file_name = '/home/delaram/SpatialPeeler/Data/Remyelin_Slide-seq/Remyelin_NMF_30_uncropped_t18_K10.h5ad'
#file_name = '/home/delaram/SpatialPeeler/Data/Remyelin_Slide-seq/Remyelin_NMF_30_uncropped_t7.h5ad'
#file_name = '/home/delaram/SpatialPeeler/Data/Remyelin_Slide-seq/Remyelin_NMF_30_uncropped_t3_7_PreprocV2.h5ad'
#file_name = '/home/delaram/SpatialPeeler/Data/Remyelin_Slide-seq/Remyelin_NMF_30_uncropped_t3_7_PreprocV2_samplewise.h5ad'
#file_name = '/home/delaram/SpatialPeeler/Data/Remyelin_Slide-seq/Remyelin_NMF_30_uncropped_t3_PreprocV2_samplewise_ALLGENES.h5ad'
#file_name = '/home/delaram/SpatialPeeler/Data/Remyelin_Slide-seq/Remyelin_NMF_30_uncropped_allLPC_PreprocV2_samplewise.h5ad'
file_name = '/home/delaram/SpatialPeeler/Data/Remyelin_Slide-seq/Remyelin_NMF_30_uncropped_t7_PreprocV2_samplewise.h5ad'
adata_cnmf.write_h5ad(file_name)




#adata_merged.obs['sample_id'] = adata_merged.obs['orig.ident']
sample_ids = adata_cnmf.obs['sample_id'].unique().tolist()
adata_by_sample = {
    sample_id: adata_cnmf[adata_cnmf.obs['sample_id'] == sample_id].copy()
    for sample_id in sample_ids
}


#gene_symbol = 'Snap25'#'Mbp' #'Ttr'#'Snap25'# 
### convert gene sumbol to ensemble id - mouse
for gene_symbol in ['Snap25', 'Mbp', 'Ttr']:
    gene_ensembl = hlps.map_symbol_to_ensembl([gene_symbol], species='mouse')
    print(gene_ensembl)
    for sample_id_to_check in range(len(sample_ids)):
        an_adata_sample = adata_by_sample[sample_ids[sample_id_to_check]]
        print(f"Sample {sample_ids[sample_id_to_check]}:")
        plot.plot_gene_spatial(an_adata_sample, gene_ensembl[gene_symbol], 
                                title=f"{gene_symbol} - {sample_ids[sample_id_to_check]}", 
                                cmap="viridis")


