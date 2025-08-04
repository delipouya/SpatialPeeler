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
root_dir = '/home/delaram/SpatialPeeler/Data/Remyelin_Slide-seq'

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
Saline_t18_12_7_mask = (metadata_df['Timepoint'].isin([18,12,7])) & (metadata_df['Condition'] == 'Saline')
sample_id_Saline_t18_12_7 = metadata_df[Saline_t18_12_7_mask]['sample_id'].values.tolist()
sample_id_merged = sample_id_LPC_t18 + sample_id_Saline_t18_12_7

#LPC_t3_mask = (metadata_df['Timepoint'] == 3) & (metadata_df['Condition'] == 'LPC')
#sample_id_LPC_t3 = metadata_df[LPC_t3_mask]['sample_id'].values.tolist()
#Saline_t3_7_mask = (metadata_df['Timepoint'].isin([3,7])) & (metadata_df['Condition'] == 'Saline')
#sample_id_Saline_t3_7 = metadata_df[Saline_t3_7_mask]['sample_id'].values.tolist()
#sample_id_merged = sample_id_LPC_t3 + sample_id_Saline_t3_7

print(metadata_df[metadata_df['sample_id'].isin(sample_id_merged)])
print(metadata_df.head())
print(metadata_df.shape)

adata_merged = imp.load_all_slide_seq_data(root_dir, normalize_log1p=True) 
adata_merged = adata_merged[adata_merged.obs['puck_id'].isin(sample_id_merged)]
adata_merged.obs['puck_id'].value_counts()
#adata_merged.X = adata_merged.layers['lognorm'].copy() # Use preprocessed layer (no additional log1p needed)
sc.pp.highly_variable_genes(adata_merged, n_top_genes=2000, subset=True)


### add medata to adata object based on the puck_id column
adata_merged.obs = adata_merged.obs.merge(metadata_df, on='puck_id', how='left')
adata_merged.obs['puck_id'].describe()
print(adata_merged.obs.head())

########################################



##################################################################
################# Importing the cropped data  ####################
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

n_factors = 10#30  
nmf_model = NMF(n_components=n_factors, init='nndsvda', 
                random_state=RAND_SEED, max_iter=1000)
# X must be dense; convert if sparse
X = adata_merged.X
if sp.issparse(X): 
    X = X.toarray()
W = nmf_model.fit_transform(X)  # cell × factor matrix
H = nmf_model.components_        # factor × gene matrix

adata_merged.obsm["X_nmf"] = W
adata_merged.uns["nmf_components"] = H

### save adata object with NMF results  
# file_name = '/home/delaram/SpatialPeeler/Data/Remyelin_Slide-seq/Remyelin_NMF_30.h5ad'
# file_name = '/home/delaram/SpatialPeeler/Data/Remyelin_Slide-seq/Remyelin_NMF_30_uncropped.h5ad'
#file_name = '/home/delaram/SpatialPeeler/Data/Remyelin_Slide-seq/Remyelin_NMF_30_uncropped_SampleWiseNorm.h5ad'
#file_name = '/home/delaram/SpatialPeeler/Data/Remyelin_Slide-seq/Remyelin_NMF_30_uncropped_t3_7.h5ad'
#file_name = '/home/delaram/SpatialPeeler/Data/Remyelin_Slide-seq/Remyelin_NMF_30_uncropped_t18.h5ad'
file_name = '/home/delaram/SpatialPeeler/Data/Remyelin_Slide-seq/Remyelin_NMF_30_uncropped_t18_K10.h5ad'

adata_merged.write_h5ad(file_name)