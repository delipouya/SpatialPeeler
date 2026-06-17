"""
B.6 — SpatialPeeler benchmark evaluation: 4×4×4 parameter grid (64 conditions).

Runs NMF + SpatialPeeler on all 64 case h5ad files in generated_benchmark_data_final/,
evaluates in-circle AUROC and gene-recovery Jaccard (correlation- and DE-based),
and saves results to benchmark/benchmark_results_grid_v5_64parameters.csv.

Runtime estimate: ~4–6 hours on a single farm node.
"""

import sys
import time
import warnings
from itertools import product
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scanpy as sc
import statsmodels.api as sm
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score

warnings.simplefilter('ignore', category=ConvergenceWarning)
sc.settings.verbosity = 0

ROOT = Path('/lustre/scratch126/gengen/teams_v2/marks/dp31/SpatialPeeler')
sys.path.insert(0, str(ROOT))

RAND_SEED  = 28
N_FACTORS  = 15
np.random.seed(RAND_SEED)

DATA_DIR  = ROOT / 'benchmark' / 'generated_benchmark_data'
FINAL_DIR = ROOT / 'benchmark' / 'generated_benchmark_data_final'
SAVE_PATH = ROOT / 'benchmark' / 'benchmark_results_grid_v5_64parameters.csv'

# ── Parameter grid ────────────────────────────────────────────────────────────
PERTURB_FRACS  = [0.30, 0.50, 0.70, 1.0]
FIXED_LAMS     = [3, 0.5, 0.3, 0.15]
TOP_GENES_LIST = [1, 2, 5, 10]

grid = list(product(PERTURB_FRACS, FIXED_LAMS, TOP_GENES_LIST))
print(f'{len(grid)} conditions', flush=True)

# ── Load shared control ───────────────────────────────────────────────────────
print('Loading control...', flush=True)
adata_ctrl = ad.read_h5ad(DATA_DIR / 'adata06_top.h5ad')
adata_ctrl.obs['sample_id'] = 'ctrl'
adata_ctrl.obs['status']    = 0
adata_ctrl.obs['Condition'] = 'Control'
if 'in_circle' not in adata_ctrl.obs.columns:
    adata_ctrl.obs['in_circle'] = False
adata_ctrl.obs['in_circle'] = adata_ctrl.obs['in_circle'].astype(bool)
print(f'Control: {adata_ctrl.shape}', flush=True)

# ── Ground-truth gene sets per top_genes value ────────────────────────────────
nmf_genes_by_tg = {}
for tg in TOP_GENES_LIST:
    csv_path = FINAL_DIR / f'nmf_genes_top{tg}_per_factor.csv'
    nmf_genes_by_tg[tg] = set(pd.read_csv(csv_path)['gene'].unique())
    print(f'top_genes={tg:>2}: {len(nmf_genes_by_tg[tg])} unique perturbed genes', flush=True)

# ── Helper functions ──────────────────────────────────────────────────────────
def preprocess(adata_ctrl, adata_case, force_genes=None):
    adata = ad.concat(
        [adata_ctrl, adata_case],
        join='inner', merge='first',
        label='sample_id', keys=['ctrl', 'case'],
        index_unique='-'
    )
    for col in ['status', 'Condition', 'in_circle']:
        adata.obs[col] = np.concatenate([
            adata_ctrl.obs[col].values,
            adata_case.obs[col].values
        ])
    adata.obs['status']    = adata.obs['status'].astype(int)
    adata.obs['in_circle'] = adata.obs['in_circle'].astype(bool)

    adata.layers['counts'] = adata.X.copy()

    min_cells = max(1, adata.n_obs // 500)
    n_expr    = np.array((adata.X > 0).sum(axis=0)).flatten()
    adata     = adata[:, n_expr >= min_cells].copy()

    adata.layers['lognorm'] = adata.layers['counts'].copy()
    sc.pp.normalize_total(adata, target_sum=1e4, layer='lognorm')
    sc.pp.log1p(adata, layer='lognorm')
    sc.pp.highly_variable_genes(
        adata, n_top_genes=2000,
        batch_key='sample_id', flavor='seurat',
        layer='lognorm', subset=False
    )

    if force_genes is not None:
        symbols    = (adata.var['features'].values
                      if 'features' in adata.var.columns
                      else adata.var_names.values)
        is_forced  = np.isin(symbols, list(force_genes))
        adata.var['highly_variable'] = adata.var['highly_variable'] | is_forced
        print(f'  forced {is_forced.sum()}/{len(force_genes)} genes into HVG set', end=' ', flush=True)

    adata = adata[:, adata.var['highly_variable']].copy()
    sc.pp.scale(adata, zero_center=False)
    return adata


def run_nmf(adata, n_factors=N_FACTORS, rand_seed=RAND_SEED):
    X = adata.X
    if sp.issparse(X):
        X = X.toarray()
    X = X.astype(np.float32)
    model = NMF(n_components=n_factors, init='nndsvda',
                random_state=rand_seed, max_iter=1000, solver='cd')
    W = model.fit_transform(X)
    adata.obsm['X_nmf'] = W
    return W


def run_spatialpeeler(adata, n_factors=N_FACTORS):
    X  = adata.obsm['X_nmf']
    y  = adata.obs['status'].values
    results = []
    for i in range(n_factors):
        Xi    = X[:, i].reshape(-1, 1)
        X_int = sm.add_constant(Xi)
        try:
            fit   = sm.Logit(y, X_int).fit(disp=False)
            p_hat = fit.predict(X_int)
            coef  = float(fit.params[1])
            pval  = float(fit.pvalues[1])
        except Exception:
            p_hat = np.full(len(y), np.nan)
            coef  = 0.0
            pval  = 1.0
        results.append({'factor_index': i, 'coef': coef, 'pval': pval, 'p_hat': p_hat})
    return results


def incircle_auroc(results, adata):
    case_mask = adata.obs['status'].values == 1
    gt        = adata.obs['in_circle'].values[case_mask].astype(int)
    aucs = {}
    for r in results:
        ph_case = r['p_hat'][case_mask]
        if np.isnan(ph_case).all() or len(np.unique(gt)) < 2:
            aucs[r['factor_index']] = np.nan
        else:
            try:
                aucs[r['factor_index']] = roc_auc_score(gt, ph_case)
            except Exception:
                aucs[r['factor_index']] = np.nan
    return aucs


def phat_gene_corr(adata, p_hat, case_only=True):
    mask = (adata.obs['status'].values == 1) if case_only else np.ones(adata.n_obs, bool)
    X    = adata.layers['lognorm'][mask, :]
    if sp.issparse(X):
        X = X.toarray()
    pv  = p_hat[mask]
    Xc  = X - X.mean(axis=0)
    pvc = pv - pv.mean()
    dX  = np.sqrt((Xc ** 2).sum(axis=0))
    dp  = np.sqrt((pvc ** 2).sum())
    with np.errstate(divide='ignore', invalid='ignore'):
        corr = np.where(
            (dX > 0) & (dp > 0),
            (Xc.T @ pvc) / (dX * dp),
            np.nan
        )
    genes = (adata.var['features'].values
             if 'features' in adata.var.columns
             else adata.var_names.values)
    return pd.DataFrame({'gene': genes, 'correlation': corr}).sort_values(
        'correlation', ascending=False, ignore_index=True
    )


def jaccard(set_a, set_b):
    a, b = set(set_a), set(set_b)
    return len(a & b) / len(a | b) if (a | b) else 0.0


# ── Main loop ─────────────────────────────────────────────────────────────────
N_CONDITIONS = len(grid)
rows         = []
t_start      = time.time()

for i, (pf, lam, tg) in enumerate(grid):
    tag  = f'frac{pf:.0%}_lam{lam}_top{tg}genes'
    path = FINAL_DIR / f'adata06_bot_case_nmfGenes_{tag}.h5ad'
    t0   = time.time()

    print(f'[{i+1:02d}/{N_CONDITIONS}] {tag} ...', end=' ', flush=True)

    adata_case = ad.read_h5ad(path)
    adata_case.obs['sample_id'] = 'case'
    adata_case.obs['status']    = 1
    adata_case.obs['Condition'] = 'Case'
    adata_case.obs['in_circle'] = adata_case.obs['in_circle'].astype(bool)

    n_in_circle = int(adata_case.obs['in_circle'].sum())
    gt_genes    = nmf_genes_by_tg[tg]
    n_gt        = len(gt_genes)

    adata   = preprocess(adata_ctrl, adata_case, force_genes=gt_genes)
    run_nmf(adata)
    results = run_spatialpeeler(adata)
    aucs    = incircle_auroc(results, adata)

    top_r    = max(results, key=lambda r: r['coef'])
    top_fi   = top_r['factor_index']
    top_auc  = aucs[top_fi]
    best_auc = max(aucs.values())

    # correlation-based gene recovery
    corr_df       = phat_gene_corr(adata, top_r['p_hat'], case_only=True)
    gt_in_hvg     = gt_genes & set(corr_df['gene'].dropna())
    recovered_all = set(corr_df.head(n_gt)['gene'].values)
    recovered_hvg = set(corr_df.head(len(gt_in_hvg))['gene'].values)
    j_all = jaccard(recovered_all, gt_genes)
    j_hvg = jaccard(recovered_hvg, gt_in_hvg)

    # DE-based gene recovery
    case_mask  = adata.obs['status'].values == 1
    ctrl_mask  = ~case_mask
    p_hat_case = top_r['p_hat'][case_mask]

    km      = KMeans(n_clusters=2, random_state=RAND_SEED, n_init='auto')
    km.fit(p_hat_case.reshape(-1, 1))
    centers = km.cluster_centers_.ravel()
    order   = np.argsort(centers)
    remap   = {order[0]: 0, order[1]: 1}
    case_km = np.vectorize(remap.get)(km.labels_)

    threshold      = centers.mean()
    cluster_labels = np.empty(adata.n_obs, dtype=object)
    cluster_labels[case_mask] = np.where(case_km == 1, 'case_1', 'case_0')
    cluster_labels[ctrl_mask] = np.where(
        top_r['p_hat'][ctrl_mask] >= threshold, 'control_1', 'control_0'
    )

    grp_mask       = cluster_labels == 'case_1'
    ref_mask       = cluster_labels == 'control_0'
    j_de           = np.nan
    n_de_recovered = 0

    if grp_mask.sum() >= 5 and ref_mask.sum() >= 5:
        adata_de = adata[grp_mask | ref_mask].copy()
        adata_de.obs['_grp'] = pd.Categorical(
            np.where(cluster_labels[grp_mask | ref_mask] == 'case_1', 'case_1', 'control_0'),
            categories=['control_0', 'case_1']
        )
        sc.tl.rank_genes_groups(
            adata_de, groupby='_grp', groups=['case_1'], reference='control_0',
            method='wilcoxon', corr_method='benjamini-hochberg',
            use_raw=False, layer='lognorm', n_genes=adata_de.n_vars
        )
        de_result = sc.get.rank_genes_groups_df(adata_de, group='case_1')
        var_to_symbol = dict(zip(
            adata.var_names,
            adata.var['features'].values if 'features' in adata.var.columns
            else adata.var_names.values
        ))
        de_result['gene_symbol'] = de_result['names'].map(var_to_symbol)
        top_de = set(
            de_result.sort_values('logfoldchanges', ascending=False)
            .head(n_gt)['gene_symbol'].values
        )
        j_de           = jaccard(top_de, gt_genes)
        n_de_recovered = len(top_de & gt_genes)

    elapsed = time.time() - t0
    print(f'F{top_fi+1}  top_auc={top_auc:.3f}  j_corr={j_all:.3f}  '
          f'j_de={j_de:.3f}  ({elapsed:.0f}s)', flush=True)

    row = {
        'perturb_frac':     pf,
        'fixed_lam':        lam,
        'top_genes':        tg,
        'tag':              tag,
        'n_in_circle':      n_in_circle,
        'top_factor':       top_fi + 1,
        'top_coef':         top_r['coef'],
        'top_pval':         top_r['pval'],
        'top_auc':          top_auc,
        'best_auc':         best_auc,
        'gene_jaccard_all': j_all,
        'gene_jaccard_hvg': j_hvg,
        'gene_jaccard_de':  j_de,
        'n_gt_genes':       n_gt,
        'n_gt_in_hvg':      len(gt_in_hvg),
        'n_recovered_all':  len(recovered_all & gt_genes),
        'n_recovered_hvg':  len(recovered_hvg & gt_in_hvg),
        'n_de_recovered':   n_de_recovered,
        'n_case_1':         int(grp_mask.sum()),
        'n_control_0':      int(ref_mask.sum()),
    }
    for fi, auc in aucs.items():
        row[f'auc_f{fi+1}'] = auc
    rows.append(row)

results_df = pd.DataFrame(rows)

results_df.to_csv(SAVE_PATH, index=False)
print(f'\nSaved {len(results_df)} rows → {SAVE_PATH}')
print(f'Total elapsed: {(time.time() - t_start)/60:.1f} min')
