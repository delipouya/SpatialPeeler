import os
import sys
root_path = os.path.abspath('./..')
sys.path.insert(0, root_path )

import itertools
import functools
from tqdm import tqdm

import pandas as pd
import numpy as np
import sklearn
import sklearn.linear_model
import sklearn.cluster
import sklearn.metrics
import matplotlib.pyplot as plt
import seaborn as sns

import hiddensc
from hiddensc import utils, files, vis

import scanpy as sc
import scvi
import anndata
import functools


import scipy as sp
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from scipy.stats import describe as describe_stats
#from scipy.stats import mannwhitneyu
from scipy.special import logit
RAND_SEED = 28
np.random.seed(RAND_SEED)

utils.set_random_seed(utils.RANDOM_SEED)
utils.print_module_versions([sc, anndata, scvi, hiddensc])
vis.visual_settings()

from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.simplefilter("ignore", category=ConvergenceWarning)

from hiddensc import models


CASE_COND = 1
def standalone_logistic(X, y):
    #clf = LogisticRegression(random_state=RAND_SEED, penalty='none').fit(X, y)
    clf = LogisticRegression(random_state=RAND_SEED, penalty=None).fit(X, y)
    predicted_label = clf.predict(X)
    predicted_prob = clf.predict_proba(X)
    return predicted_prob[:,1]

def PCA_logistic_kmeans(adata, num_pcs):
    
    p_hat = standalone_logistic(adata.obsm['X_pca'][:, 0:num_pcs], adata.obs['status'].values)
    df_p_hat = pd.DataFrame()
    df_p_hat['p_hat'] = p_hat
    df_p_hat['status'] = adata.obs['status'].values
    df_p_hat['sample_id'] = adata.obs['sample_id'].values
    adata.obs['p_hat'] = p_hat
    logit_p_hat = logit(p_hat)
    df_p_hat['logit_p_hat'] = logit_p_hat
    adata.obs['logit_p_hat'] = logit_p_hat

    kmeans_case = KMeans(n_clusters=2, random_state=0).fit(pd.DataFrame(logit(p_hat))[(adata.obs['status']==CASE_COND).values])
    mean_p_hat_kmeans_label0 = np.mean(p_hat[(adata.obs['status']==CASE_COND).values][kmeans_case.labels_==0]) 
    mean_p_hat_kmeans_label1 = np.mean(p_hat[(adata.obs['status']==CASE_COND).values][kmeans_case.labels_==1])
    zero_lab_has_lower_mean = mean_p_hat_kmeans_label0 < mean_p_hat_kmeans_label1

    df_p_hat_clust_case = df_p_hat.copy()
    df_p_hat_clust_case['kmeans'] = 0
    df_p_hat_clust_case['kmeans'][(adata.obs['status']==CASE_COND).values] = [1 if x==int(zero_lab_has_lower_mean) else 0 for x in kmeans_case.labels_]
    
       
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    sns.histplot(x=logit_p_hat, hue='status',
                 #palette={0:"#1B9E77", CASE_COND:"#D95F02"}, 
                 data=df_p_hat, ax=ax1)
    ax1.set_title(f'distr. of logit(p_hat), orig labels, all cells')
    sns.histplot(x=logit_p_hat, hue='kmeans', data=df_p_hat_clust_case, ax=ax2)
    ax2.set_title(f'distr. of logit(p_hat), new labels, all cells')
    #plt.savefig(f'figures/logit_p_hat_clust_case_pcs{num_pcs}_{CASE_COND}.png')
    plt.show()
    
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    sns.histplot(x=p_hat, hue='status', 
                 #palette={0:"#1B9E77", CASE_COND:"#D95F02"}, 
                 data=df_p_hat, ax=ax1)
    ax1.set_title(f'distr. of p_hat, orig labels, all cells')
    sns.histplot(x=p_hat, hue='kmeans', data=df_p_hat_clust_case, ax=ax2)
    ax2.set_title(f'distr. of p_hat, new labels, all cells')
    #plt.savefig(f'figures/p_hat_clust_case_pcs{num_pcs}_{CASE_COND}.png')
    plt.show()
    
    return p_hat, df_p_hat_clust_case['kmeans']




# Files to load
file_names = [
    "normal_A", "normal_B", "normal_C", "normal_D",
    "PSC_A", "PSC_B", "PSC_C", "PSC_D"
]


adata_dict = {}
cell_type_sets = {}
# Load and extract unique cell types
for fname in file_names:
    
    PREFIX = fname.split('.')[0]  # Use the first file name as prefix
    at_data_dir = functools.partial(os.path.join, root_path, 'data_PSC')
    adata = sc.read(at_data_dir(f'{PREFIX}.h5ad'))
    adata_dict[fname] = adata

    adata.obs['binary_label'] = adata.obs['disease']!='normal' #'primary sclerosing cholangitis'

    hiddensc.datasets.preprocess_data(adata)
    hiddensc.datasets.normalize_and_log(adata)

    cell_types = adata.obs['cell_type'].unique()
    cell_type_sets[fname] = set(cell_types)

cell_type_sets
#['periportal region hepatocyte', 'centrilobular region hepatocyte',
#  'endothelial cell', 'hepatocyte', 'blood cell']

'''
## EXPERIMNT FOR SANITY CHECKING
### switch two of the samples - one normal and one PSC - to see if the model can detect the switch
# Swap disease labels for one normal and one PSC sample
adata_dict["normal_A"].obs['disease'] = "PSC"
adata_dict["normal_A"].obs['binary_label'] = True

adata_dict["PSC_A"].obs['disease'] = "normal"
adata_dict["PSC_A"].obs['binary_label'] = False
adata_dict["normal_A"].obs['is_swapped'] = True
adata_dict["PSC_A"].obs['is_swapped'] = True
for k in adata_dict:
    if 'is_swapped' not in adata_dict[k].obs:
        adata_dict[k].obs['is_swapped'] = False
'''


# Make a DataFrame to compare
all_cell_types = sorted(set.union(*cell_type_sets.values()))
df = pd.DataFrame(
    {fname: [ct in cell_type_sets[fname] for ct in all_cell_types] for fname in file_names},
    index=all_cell_types
)
## 'periportal region hepatocyte', 'centrilobular region hepatocyte' are present in all samples


# Add batch info before merging
for fname, ad in adata_dict.items():
    ad.obs["batch"] = fname  # This will let HiDDEN know the origin

# Concatenate all datasets
adata_merged = anndata.concat(adata_dict.values(), 
                              label="sample_id", keys=adata_dict.keys(), merge="same")


adata_merged.obs['batch'] = adata_merged.obs['batch'].astype(str)
adata_merged.obs['sample_id'] = adata_merged.obs['sample_id'].astype(str)

hiddensc.datasets.normalize_and_log(adata_merged)

# Generate features (optional: PCA, neighbors, etc. depending on what HiDDEN expects)
hiddensc.datasets.augment_for_analysis(adata_merged)

sc.pl.umap(adata_merged, color=['batch'], s=120)
sc.pl.pca(adata_merged, color=['batch'], s=120)
sc.pl.umap(adata_merged, color=['cell_type'], s=120)
sc.pl.pca(adata_merged, color=['cell_type'], s=120)


num_pcs, ks, ks_pval = hiddensc.models.determine_pcs_heuristic_ks(adata=adata_merged, 
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

# (21553, 4997)
#optimal_num_pcs_ks = #5


NUM_PCS = optimal_num_pcs_ks
NUM_PCS = 5
adata_merged.obs['status'] = adata_merged.obs['binary_label'].astype('int').values
## 0=False=normal=control
## 1=True=PSC=case
p_hat, new_labels = PCA_logistic_kmeans(adata_merged, num_pcs=NUM_PCS)


## (21553, 4997)
# len(p_hat) 21553

# Make a copy of the obs DataFrame
df_combined = adata_merged.obs.copy()

# Add p_hat and new_labels as new columns
df_combined['p_hat'] = p_hat
df_combined['new_label'] = new_labels.values  # make sure it's aligned and has the right length

# Show shape and a preview
print(df_combined.shape)
df_combined.head()

adata_merged.obs['new_label'] = new_labels.values
adata_merged.obs['new_label'] = adata_merged.obs['new_label'].astype('category')
adata_merged.obs['p_hat'] = p_hat
adata_merged.obs['p_hat'] = adata_merged.obs['p_hat'].astype('float32')

### save adata_merged to h5ad to be imported in R
at_data_dir = functools.partial(os.path.join, root_path, 'data_PSC')
#adata_merged.write(at_data_dir('PSC_merged.h5ad'), compression='gzip')





######################## Plotting the results
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from matplotlib.patches import Patch

## Load the merged data
adata = sc.read_h5ad("/home/delaram/LabelCorrection/data_PSC/PSC_merged.h5ad")

adata = adata_merged.copy()
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



def plot_spatial_pc(adata, pc_idx, sample_id=None):
    spatial = adata.obsm["spatial"]
    pc_values = adata.obsm["X_pca"][:, pc_idx]
    
    # Optionally cap extremes for better visualization
    vmax = np.quantile(pc_values, 0.99)
    vmin = np.quantile(pc_values, 0.01)
    pc_values = np.clip(pc_values, vmin, vmax)

    plt.figure(figsize=(5, 5))
    sc = plt.scatter(spatial[:, 0], spatial[:, 1], c=pc_values, cmap="coolwarm", s=10)
    plt.axis("equal")
    title = f"PC{pc_idx + 1}"
    if sample_id:
        title += f" – {sample_id}"
    plt.title(title)
    plt.colorbar(sc, label=f"PC{pc_idx + 1} value")
    plt.xticks([]); plt.yticks([])
    plt.tight_layout()
    plt.show()


# For individual samples
for sid in sample_ids:
    print(f"Plotting PCs for sample {sid}")
    for pc in range(4):
        plot_spatial_pc(adata_by_sample[sid], pc, sample_id=sid)


### plot the distribution of PC1-4 over all samples in a voilin plot

# Extract first 4 PCs
pcs = adata.obsm['X_pca'][:, :4]
pc_df = pd.DataFrame(pcs, columns=[f'PC{i+1}' for i in range(4)])

# Add sample ID
pc_df['sample_id'] = adata.obs['sample_id'].values

# Melt to long format for seaborn
pc_long = pc_df.melt(id_vars='sample_id', var_name='PC', value_name='value')

# Plot
plt.figure(figsize=(12, 6))
sns.violinplot(
    x="PC", y="value", hue="sample_id",
    data=pc_long, split=False, inner="box", palette="Set2"
)
plt.title("Distribution of PC1–PC4 across samples")
plt.legend(title="Sample ID", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


# 1. Extract features (PCs) and target (disease label)
X = adata.obsm['X_pca'][:, :50]
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

# Optional: print all for inspection
# display(importance_df_sorted)

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



# Parameters
pc_idx = 1  # PC2
pc_idx = 2  # PC4

for pc_idx in range(12):
    pc_name = f"PC{pc_idx + 1}"

    # Global scale
    pc2_all = adata.obsm["X_pca"][:, pc_idx]
    vmin, vmax = np.quantile(pc2_all, [0.01, 0.99])

    # Grid for 8 samples
    n_cols = 4
    n_rows = int(np.ceil(len(sample_ids) / n_cols))
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(16, 8))
    axs = axs.flatten()

    # Plot PC2 for each sample
    for i, sid in enumerate(sample_ids):
        ad = adata_by_sample[sid]
        spatial = ad.obsm["spatial"]
        pc2 = ad.obsm["X_pca"][:, pc_idx]

        im = axs[i].scatter(
            spatial[:, 0], spatial[:, 1],
            c=np.clip(pc2, vmin, vmax),
            cmap="RdBu_r", s=10, vmin=vmin, vmax=vmax
        )
        axs[i].set_title(sid, fontsize=10)
        axs[i].axis("off")

    # Add a colorbar on the side (not overlapping)
    cbar_ax = fig.add_axes([0.92, 0.25, 0.015, 0.5])  # [left, bottom, width, height]
    cb = fig.colorbar(im, cax=cbar_ax)
    cb.set_label(f"{pc_name} value", fontsize=12)

    plt.suptitle(f"{pc_name} projection over spatial coordinates (per sample)", fontsize=14)
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    plt.show()

# Define a consistent color palette for cell types
cell_types = adata.obs['cell_type'].unique().tolist()
palette = sns.color_palette("tab20", n_colors=len(cell_types))
color_map = dict(zip(cell_types, palette))

# Plot setup
n_cols = 4
n_rows = int(np.ceil(len(sample_ids) / n_cols))
fig, axs = plt.subplots(n_rows, n_cols, figsize=(16, 8))
axs = axs.flatten()

# Plot each sample
for i, sid in enumerate(sample_ids):
    ad = adata_by_sample[sid]
    spatial = ad.obsm["spatial"]
    cell_types_series = ad.obs['cell_type'].astype(str)
    colors = cell_types_series.map(color_map)

    axs[i].scatter(
        spatial[:, 0], spatial[:, 1],
        c=colors, s=10
    )
    axs[i].set_title(sid, fontsize=10)
    axs[i].axis("off")

# Add a single legend
# Use dummy points to generate legend from color_map
legend_elements = [Patch(facecolor=color_map[ct], label=ct) for ct in cell_types]

fig.legend(handles=legend_elements, title="Cell Type", loc='center right', bbox_to_anchor=(1.02, 0.5), fontsize=8, title_fontsize=9)
plt.suptitle("Cell Type Distribution over Spatial Coordinates (per sample)", fontsize=14)
plt.tight_layout(rect=[0, 0, 0.9, 0.95])
plt.show()






######################## using spatial_factorization embeddings instead of PCA
import anndata as ad
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from matplotlib.patches import Patch

## Load the merged data
adata = sc.read_h5ad("/home/delaram/LabelCorrection/data_PSC/PSC_merged.h5ad")
import spatial_factorization
from spatial_factorization import SpatialFactorization, ModelTrainer
