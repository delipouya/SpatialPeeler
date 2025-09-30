import os
import sys
root_path = os.path.abspath('./..')
sys.path.insert(0, root_path)
import hiddensc
from hiddensc import utils, vis
import scanpy as sc
import scvi
import anndata
import numpy as np
import pandas as pd
import mygene

#from scipy.stats import mannwhitneyu
RAND_SEED = 28
CASE_COND = 1
np.random.seed(RAND_SEED)
utils.set_random_seed(utils.RANDOM_SEED)
utils.print_module_versions([sc, anndata, scvi, hiddensc])
vis.visual_settings()
from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.simplefilter("ignore", category=ConvergenceWarning)


def map_ensembl_to_symbol(ensembl_ids, species='human'):
    mg = mygene.MyGeneInfo()
    result = mg.querymany(ensembl_ids, scopes='ensembl.gene', fields='symbol',species=species)
    
    # Build mapping dict
    id_to_symbol = {r['query']: r.get('symbol', None) for r in result}
    return id_to_symbol

### map symbol to ensembl
def map_symbol_to_ensembl(gene_symbols, species='human'):
    mg = mygene.MyGeneInfo()
    result = mg.querymany(gene_symbols, scopes='symbol', fields='ensembl.gene',species=species)
    
    # Build mapping dict
    symbol_to_id = {r['query']: r.get('ensembl', None) for r in result}
    # Handle cases where 'ensembl' is a list of dicts
    for key, value in symbol_to_id.items():
        if isinstance(value, list):
            symbol_to_id[key] = value[0]['gene']  # Take the first ensembl ID
        elif isinstance(value, dict):
            symbol_to_id[key] = value['gene']  # Directly take the gene ID
    return symbol_to_id