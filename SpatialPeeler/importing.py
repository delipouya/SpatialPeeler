import os
import pandas as pd
import anndata
from scipy import io
import os
import sys
root_path = os.path.abspath('./..')
sys.path.insert(0, root_path)
import pandas as pd
import numpy as np
import hiddensc
from hiddensc import utils, files, vis
import scanpy as sc
import scvi
import anndata
from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.simplefilter("ignore", category=ConvergenceWarning)
import re

RAND_SEED = 28
CASE_COND = 1
np.random.seed(RAND_SEED)
utils.set_random_seed(utils.RANDOM_SEED)
utils.print_module_versions([sc, anndata, scvi, hiddensc])
vis.visual_settings()



def load_slide_seq_puck(puck_dir, puck_id):
    """
    Load a single Slide-seq puck (from root of each puck folder).
    """
    matrix_path = None
    features_path = None
    barcodes_path = None

    for file in os.listdir(puck_dir):
        if file.endswith(".mtx"):
            matrix_path = os.path.join(puck_dir, file)
        elif "features" in file and file.endswith(".tsv"):
            features_path = os.path.join(puck_dir, file)
        elif "barcodes" in file and file.endswith(".tsv"):
            barcodes_path = os.path.join(puck_dir, file)

    if not all([matrix_path, features_path, barcodes_path]):
        print(f"Skipping {puck_id}: Missing .mtx, features, or barcodes.")
        return None

    print(f"Loading puck: {puck_id}")

    X = io.mmread(matrix_path).tocsr()

    genes = pd.read_csv(features_path, header=None, sep='\t')
    gene_names = genes.iloc[:, 0].values

    barcodes = pd.read_csv(barcodes_path, header=None, sep='\t')
    cell_barcodes = barcodes.iloc[:, 0].values

    adata = anndata.AnnData(X=X.T)
    adata.var_names = gene_names
    adata.obs_names = [f"{puck_id}_{bc}" for bc in cell_barcodes]
    adata.obs['puck_id'] = puck_id

    return adata



def load_all_slide_seq_data(root_dir):
    """
    Load all puck folders (assuming .mtx and .tsv files live directly inside each puck folder root).
    """
    ## three samples were excluded from the analysis in the original study
    valid_puck_names = [
        "Puck_230117_01",
        "Puck_230117_39",
        "Puck_230130_19",
        "Puck_230130_31",
        "Puck_230130_32",
        "Puck_230130_33",
        "Puck_230130_34",
        "Puck_230130_37",
        "Puck_230130_38",
        "Puck_230130_39",
        "Puck_230321_08",
        "Puck_230321_19",
        "Puck_230321_20",
        "Puck_230403_21",
        "Puck_230403_23"
    ]

    full_puck_names = [f"2023-08-25_{name}" for name in valid_puck_names]   

    pattern = r"2023-08-25_Puck_.*"  # Match all folders starting with "2023-08-25_Puck_"
    path = os.listdir(root_dir)
    sorted_path = [d for d in path if os.path.isdir(os.path.join(root_dir, d)) and re.match(pattern, d)]
    valid_sorted_path = [d for d in sorted_path if d in full_puck_names] 

    adata_list = []

    for subdir in valid_sorted_path:
        puck_folder = os.path.join(root_dir, subdir)
        if not os.path.isdir(puck_folder):
            continue

        adata = load_slide_seq_puck(puck_folder, subdir)
        if adata is not None:
            adata_list.append(adata)

    if len(adata_list) == 0:
        raise ValueError("No valid puck datasets found.")

    print(f"Merging {len(adata_list)} pucks...")
    adata_merged = anndata.concat(
        adata_list, 
        label='puck_id', 
        keys=[adata.obs['puck_id'][0] for adata in adata_list]
    )

    return adata_merged
