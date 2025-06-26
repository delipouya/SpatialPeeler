import os
import pandas as pd
import anndata
from scipy import io


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

    adata = anndata.AnnData(X=X)
    adata.var_names = gene_names
    adata.obs_names = [f"{puck_id}_{bc}" for bc in cell_barcodes]
    adata.obs['puck_id'] = puck_id

    return adata

def load_all_slide_seq_data(root_dir):
    """
    Load all puck folders (assuming .mtx and .tsv files live directly inside each puck folder root).
    """
    adata_list = []

    for subdir in sorted(os.listdir(root_dir)):
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

root_dir = '/home/delaram/SpatialPeeler/Data/Remyelin_Slide-seq'
adata_merged = load_all_slide_seq_data(root_dir)