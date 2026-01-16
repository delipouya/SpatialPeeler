import re
import gzip
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.io import mmread


SAMPLE_DIRS_PCS = [
    "~/SpatialPeeler/Data/PSC_liver/PCS_GSE245620/GSM7845914_PSC011_A1",
    "~/SpatialPeeler/Data/PSC_liver/PCS_GSE245620/GSM7845915_PSC011_B1",
    "~/SpatialPeeler/Data/PSC_liver/PCS_GSE245620/GSM7845916_PSC011_C1",
    "~/SpatialPeeler/Data/PSC_liver/PCS_GSE245620/GSM7845917_PSC011_D1",
]

SAMPLE_DIRS_CONTROL = [
    "~/SpatialPeeler/Data/PSC_liver/CONTROL_GSE240429/GSM7697868_C73A1",
    "~/SpatialPeeler/Data/PSC_liver/CONTROL_GSE240429/GSM7697869_C73B1",
    "~/SpatialPeeler/Data/PSC_liver/CONTROL_GSE240429/GSM7697870_C73C1",
    "~/SpatialPeeler/Data/PSC_liver/CONTROL_GSE240429/GSM7697871_C73D1",
]

SAMPLE_DIRS = SAMPLE_DIRS_PCS + SAMPLE_DIRS_CONTROL


def _find_prefix(sample_dir: Path) -> str:
    """
    Find the matrix file and define the prefix as filename without '_matrix.mtx.gz'.
    Works for both:
      GSM..._VISIUM_matrix.mtx.gz
      GSM..._matrix.mtx.gz
    """
    mtx_file = next(sample_dir.glob("*_matrix.mtx.gz"), None)
    if mtx_file is None:
        raise FileNotFoundError(f"No '*_matrix.mtx.gz' found in {sample_dir}")
    return mtx_file.name.replace("_matrix.mtx.gz", "")


def _sample_name_from_folder(sample_dir: Path) -> str:
    """
    GSM7845914_PSC011_A1 -> PSC011_A1
    GSM7697868_C73A1     -> C73A1
    """
    parts = sample_dir.name.split("_", 1)
    return parts[1] if len(parts) == 2 else sample_dir.name


def load_one_geo_visium(sample_dir: str) -> sc.AnnData:
    sample_dir = Path(sample_dir).expanduser().resolve()

    prefix = _find_prefix(sample_dir)
    sample_name = _sample_name_from_folder(sample_dir)

    # --- counts (transpose: genes x barcodes -> barcodes x genes)
    with gzip.open(sample_dir / f"{prefix}_matrix.mtx.gz", "rb") as f:
        X = mmread(f).tocsr()
    X = X.T.tocsr()

    # --- barcodes
    barcodes = pd.read_csv(
        sample_dir / f"{prefix}_barcodes.tsv.gz",
        header=None, sep="\t"
    )[0].astype(str).values

    # --- features
    feats = pd.read_csv(
        sample_dir / f"{prefix}_features.tsv.gz",
        header=None, sep="\t"
    )
    gene_ids = feats.iloc[:, 0].astype(str).values
    gene_names = feats.iloc[:, 1].astype(str).values

    # --- build AnnData
    adata = sc.AnnData(X=X)
    adata.obs_names = barcodes
    adata.var_names = gene_names
    adata.var["gene_ids"] = gene_ids
    adata.var_names_make_unique()

    # --- tissue positions
    tp = pd.read_csv(sample_dir / f"{prefix}_tissue_positions_list.csv.gz", header=None)
    tp.columns = [
        "barcode", "in_tissue", "array_row", "array_col",
        "pxl_row_in_fullres", "pxl_col_in_fullres"
    ]
    tp["barcode"] = tp["barcode"].astype(str)
    tp = tp.set_index("barcode").reindex(adata.obs_names)

    # if any missing barcodes, restrict to intersection
    missing = tp["in_tissue"].isna()
    if missing.any():
        keep = ~missing
        adata = adata[keep].copy()
        tp = tp.loc[adata.obs_names]

    adata.obs["in_tissue"] = tp["in_tissue"].astype(int).values
    adata.obs["array_row"] = tp["array_row"].astype(int).values
    adata.obs["array_col"] = tp["array_col"].astype(int).values

    adata.obsm["spatial"] = np.c_[
        tp["pxl_col_in_fullres"].values,
        tp["pxl_row_in_fullres"].values
    ].astype(float)

    # --- metadata
    adata.obs["sample"] = sample_name
    adata.obs["condition"] = "PSC" if sample_name.startswith("PSC") else "CONTROL"
    adata.uns["source"] = "GEO"
    adata.uns["geo_folder"] = str(sample_dir)
    adata.uns["geo_prefix"] = prefix

    return adata


for d in SAMPLE_DIRS:
    adata = load_one_geo_visium(d)

    sample = adata.obs["sample"].iloc[0]
    out_path = Path(d).expanduser().resolve() / f"{sample}_raw.h5ad"
    adata.write_h5ad(out_path)

    print(f"Saved {out_path} | shape={adata.shape} | condition={adata.obs['condition'].iloc[0]}")