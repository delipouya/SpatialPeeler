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


def map_ensembl_to_symbol(ensembl_ids):
    mg = mygene.MyGeneInfo()
    result = mg.querymany(ensembl_ids, scopes='ensembl.gene', fields='symbol', species='human')
    
    # Build mapping dict
    id_to_symbol = {r['query']: r.get('symbol', None) for r in result}
    return id_to_symbol

