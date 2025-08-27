from SpatialPeeler import weightedcorr as wc
import pandas as pd
import numpy as np
# Example data
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 6])
w = np.array([0.1, 0.2, 0.3, 0.2, 0.2]) # Weights
w = np.array([0.1, 0.2, 0.3, 0.2, 0.2])*10 # Weights
print(w)

df = pd.DataFrame({'x': x, 'y': y, 'w': w})
pearson = wc.WeightedCorr(xyw=df)('pearson')
print(pearson)




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
from matplotlib_venn import venn3

from SpatialPeeler import helpers as hlps
from SpatialPeeler import case_prediction as cpred
from SpatialPeeler import plotting as plot
import pickle

RAND_SEED = 28
CASE_COND = 1
np.random.seed(RAND_SEED)
utils.set_random_seed(utils.RANDOM_SEED)
utils.print_module_versions([sc, anndata, scvi, hiddensc])
vis.visual_settings()


adata_merged = sc.read_h5ad('/home/delaram/SpatialPeeler/Data/PSC_liver/PSC_NMF_30.h5ad')
sample_ids = adata_merged.obs['sample_id'].unique().tolist()

with open('/home/delaram/SpatialPeeler/Data/PSC_liver/results.pkl', 'rb') as f:
    results = pickle.load(f)

factor_idx = 22#8 #12

result = results[factor_idx] 
adata_merged.obs['p_hat'] = result['p_hat']
adata_merged.obs['p_hat'] = adata_merged.obs['p_hat'].astype('float32')
adata_merged.obs['1_p_hat'] = 1 - adata_merged.obs['p_hat']

# Create a dictionary splitting the merged data by sample
adata_by_sample = {
    sid: adata_merged[adata_merged.obs['sample_id'] == sid].copy()
    for sid in adata_merged.obs['sample_id'].unique()
}
sample_ids = list(adata_by_sample.keys())




#4:7 are PSC samples, 0:3 are normal samples - initial analysis on sample #5, 4
#sample_id_to_check = 5
sample_id_to_check = 1
an_adata_sample = adata_by_sample[sample_ids[sample_id_to_check]]
# Compute spatial weights using squidpy
# Get expression matrix and coordinates

#### calculating spatial weights using squidpy


PATTERN_COND = 'LOF'#'GOF'#'LOF'
expr_matrix = an_adata_sample.X.toarray() if issparse(an_adata_sample.X) else an_adata_sample.X  # shape: (n_spots, n_genes)

p_hat_vector = an_adata_sample.obs['p_hat']  # shape: (n_spots,)
neg_p_hat_vector = an_adata_sample.obs['1_p_hat']  # shape: (n_spots,)
pattern_vector = p_hat_vector if PATTERN_COND == 'GOF' else neg_p_hat_vector

g = 0
exp_g = expr_matrix[:, g]
exp_g_centered = exp_g - exp_g.mean()
NMF_idx_values = an_adata_sample.obsm["X_nmf"][:,factor_idx]

df = pd.DataFrame({'x': exp_g, 'y': pattern_vector, 'w': NMF_idx_values})
pearson = wc.WeightedCorr(xyw=df)('pearson')
print(pearson)

pearson_vector = np.zeros(expr_matrix.shape[1])
for g in range(expr_matrix.shape[1]):
    exp_g = expr_matrix[:, g]
    exp_g_centered = exp_g - exp_g.mean()
    NMF_idx_values = an_adata_sample.obsm["X_nmf"][:,factor_idx]
    
    df = pd.DataFrame({'x': exp_g, 'y': pattern_vector, 'w': NMF_idx_values})
    pearson = wc.WeightedCorr(xyw=df)('pearson')
    pearson_vector[g] = pearson
    if g % 100 == 0:
        print(f"Processed gene {g+1}/{expr_matrix.shape[1]}: Pearson correlation = {pearson:.4f}")
    

### make a dataframe with gene names and pearson values and gene symbols using hlps.map_ensembl_to_symbol
pearson_df = pd.DataFrame({
    'gene': an_adata_sample.var_names,
    'pearson_correlation': pearson_vector
})
pearson_df['symbol'] = hlps.map_ensembl_to_symbol(pearson_df['gene'])
pearson_df = pearson_df.sort_values(by='pearson_correlation', ascending=False)




for g in range(expr_matrix.shape[1]):
        x = expr_matrix[:, g]
        x_centered = x - x.mean()


spatial_corr_1 = gid.spatial_weighted_correlation_matrix(expr_matrix, pattern_vector, 
                                                        W1, an_adata_sample.var_names)








####################################################
#### calculating spatial weights using squidpy
import squidpy as sq
sq.gr.spatial_neighbors(an_adata_sample, coord_type="generic", n_neighs=10) #### how many neighbors to use?
W1 = an_adata_sample.obsp["spatial_connectivities"]
W1_array = np.array(W1.todense())  # if W1 is sparse, convert to dense matrix
coords = an_adata_sample.obsm['spatial']  # shape: (n_spots, 2)
W2_geary = gid.compute_spatial_weights(coords, k=10, mode="geary")
W2_moran = gid.compute_spatial_weights(coords, k=10, mode="moran")
spatial_corr = gid.spatial_weighted_correlation_matrix(expr_matrix, pattern_vector,
                                                        W1, an_adata_sample.var_names)

################################################################



