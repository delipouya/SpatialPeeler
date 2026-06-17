"""
Spatial null testing on benchmark data (frac70%, lam0.5, top10genes).

Runs BrainSMASH-style surrogate correction on 5 k subsampled beads,
saves results CSV and all figures to benchmark/spatial_nulls_results/.

Estimated runtime: ~3 hours on a single node.
"""

import sys
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scanpy as sc
import statsmodels.api as sm
from sklearn.decomposition import NMF
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score

warnings.simplefilter('ignore', category=ConvergenceWarning)
sc.settings.verbosity = 0

ROOT = Path('/lustre/scratch126/gengen/teams_v2/marks/dp31/SpatialPeeler')
sys.path.insert(0, str(ROOT))

from SpatialPeeler.spatial_nulls import (
    spatial_factor_pvalues_all,
    spatial_factor_pvalue,
    coords_from_adata,
)

# ── Parameters ────────────────────────────────────────────────────────────────
RAND_SEED    = 28
N_FACTORS    = 15
PERTURB_FRAC = 0.70
FIXED_LAM    = 0.5
TOP_GENES    = 10
N_SURROGATES = 200
N_SUBSAMPLE  = 5000

np.random.seed(RAND_SEED)

DATA_DIR  = ROOT / 'benchmark' / 'generated_benchmark_data'
OUT_DIR   = ROOT / 'benchmark' / 'spatial_nulls_results'
OUT_DIR.mkdir(parents=True, exist_ok=True)

TAG       = f'frac{PERTURB_FRAC:.0%}_lam{FIXED_LAM}_top{TOP_GENES}genes'
CASE_PATH = DATA_DIR / f'adata06_bot_case_nmfGenes_{TAG}.h5ad'

t_start = time.time()
print(f'[{time.strftime("%H:%M:%S")}] Starting spatial null benchmark — {TAG}')
print(f'  N_SURROGATES={N_SURROGATES}  N_SUBSAMPLE={N_SUBSAMPLE}')
print(f'  Output dir: {OUT_DIR}')


# ── Helpers (from B.6) ────────────────────────────────────────────────────────
def preprocess(adata_ctrl, adata_case, force_genes=None):
    adata = ad.concat(
        [adata_ctrl, adata_case],
        join='inner', merge='first',
        label='sample_id', keys=['ctrl', 'case'],
        index_unique='-',
    )
    for col in ['status', 'Condition', 'in_circle']:
        adata.obs[col] = np.concatenate([
            adata_ctrl.obs[col].values,
            adata_case.obs[col].values,
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
        adata, n_top_genes=2000, batch_key='sample_id',
        flavor='seurat', layer='lognorm', subset=False,
    )
    if force_genes is not None:
        symbols = (adata.var['features'].values
                   if 'features' in adata.var.columns else adata.var_names.values)
        adata.var['highly_variable'] |= np.isin(symbols, list(force_genes))

    adata = adata[:, adata.var['highly_variable']].copy()
    sc.pp.scale(adata, zero_center=False)
    return adata


def run_nmf(adata):
    X = adata.X
    if sp.issparse(X):
        X = X.toarray()
    X = X.astype(np.float32)
    model = NMF(n_components=N_FACTORS, init='nndsvda',
                random_state=RAND_SEED, max_iter=1000, solver='cd')
    W = model.fit_transform(X)
    adata.obsm['X_nmf'] = W
    return W


def run_spatialpeeler(adata):
    X = adata.obsm['X_nmf']
    y = adata.obs['status'].values
    results = []
    for i in range(N_FACTORS):
        Xi    = X[:, i].reshape(-1, 1)
        X_int = sm.add_constant(Xi)
        try:
            fit   = sm.Logit(y, X_int).fit(disp=False)
            coef  = float(fit.params[1])
            pval  = float(fit.pvalues[1])
            p_hat = fit.predict(X_int)
        except Exception:
            coef, pval, p_hat = 0.0, 1.0, np.full(len(y), np.nan)
        results.append({'factor_index': i, 'coef': coef, 'pval': pval, 'p_hat': p_hat})
    return results


# ── 1. Load data ──────────────────────────────────────────────────────────────
print(f'[{time.strftime("%H:%M:%S")}] Loading data...')

adata_ctrl = ad.read_h5ad(DATA_DIR / 'adata06_top.h5ad')
adata_ctrl.obs['sample_id'] = 'ctrl'
adata_ctrl.obs['status']    = 0
adata_ctrl.obs['Condition'] = 'Control'
if 'in_circle' not in adata_ctrl.obs.columns:
    adata_ctrl.obs['in_circle'] = False
adata_ctrl.obs['in_circle'] = adata_ctrl.obs['in_circle'].astype(bool)

adata_case = ad.read_h5ad(CASE_PATH)
adata_case.obs['sample_id'] = 'case'
adata_case.obs['status']    = 1
adata_case.obs['Condition'] = 'Case'
adata_case.obs['in_circle'] = adata_case.obs['in_circle'].astype(bool)

gt_genes = set(
    pd.read_csv(DATA_DIR / f'nmf_genes_top{TOP_GENES}_per_factor.csv')['gene'].unique()
)
print(f'  Control: {adata_ctrl.shape}  |  Case: {adata_case.shape}')
print(f'  In-circle beads: {adata_case.obs["in_circle"].sum():,}  |  '
      f'GT genes: {len(gt_genes)}')


# ── 2. Preprocess + NMF ───────────────────────────────────────────────────────
print(f'[{time.strftime("%H:%M:%S")}] Preprocessing...')
adata = preprocess(adata_ctrl, adata_case, force_genes=gt_genes)
print(f'[{time.strftime("%H:%M:%S")}] Running NMF (k={N_FACTORS})...')
run_nmf(adata)
print(f'  Combined adata: {adata.shape}')


# ── 3. Standard SpatialPeeler ─────────────────────────────────────────────────
print(f'[{time.strftime("%H:%M:%S")}] Standard SpatialPeeler...')
sp_results = run_spatialpeeler(adata)

sp_df = pd.DataFrame([{
    'factor':        f"F{r['factor_index']+1}",
    'factor_index':  r['factor_index'],
    'coef':          r['coef'],
    'standard_pval': r['pval'],
} for r in sp_results])

case_mask = adata.obs['status'].values == 1
gt_vec    = adata.obs['in_circle'].values[case_mask].astype(int)
for r in sp_results:
    ph = r['p_hat'][case_mask]
    try:
        sp_df.loc[sp_df['factor_index'] == r['factor_index'], 'auroc'] = roc_auc_score(gt_vec, ph)
    except Exception:
        sp_df.loc[sp_df['factor_index'] == r['factor_index'], 'auroc'] = np.nan

top_factor = sp_df.sort_values('coef', ascending=False).iloc[0]
top_fi     = int(top_factor['factor_index'])
print(f"  Top factor: {top_factor['factor']}  "
      f"coef={top_factor['coef']:.3f}  "
      f"pval={top_factor['standard_pval']:.2e}  "
      f"AUROC={top_factor['auroc']:.3f}")
print(f"  Standard p<0.05: {(sp_df['standard_pval'] < 0.05).sum()} / {len(sp_df)} factors")


# ── 4. Subsample for spatial null ─────────────────────────────────────────────
print(f'[{time.strftime("%H:%M:%S")}] Building coordinates and subsampling...')

coords = coords_from_adata(adata, obsm_key='spatial',
                            x_col='Raw_Slideseq_X', y_col='Raw_Slideseq_Y')
adata.obsm['spatial'] = coords

rng_sub  = np.random.default_rng(RAND_SEED)
ctrl_idx = np.where(adata.obs['status'].values == 0)[0]
case_idx = np.where(adata.obs['status'].values == 1)[0]
n_ctrl_sub = round(N_SUBSAMPLE * len(ctrl_idx) / len(adata))
n_case_sub = N_SUBSAMPLE - n_ctrl_sub
sub_idx    = np.sort(np.concatenate([
    rng_sub.choice(ctrl_idx, n_ctrl_sub, replace=False),
    rng_sub.choice(case_idx, n_case_sub, replace=False),
]))
adata_sub = adata[sub_idx].copy()
print(f'  Subsampled: {adata_sub.n_obs:,} beads  '
      f'(ctrl={n_ctrl_sub:,}, case={n_case_sub:,})')


# ── 5. Spatial null test ──────────────────────────────────────────────────────
print(f'[{time.strftime("%H:%M:%S")}] Running spatial null test '
      f'({N_SURROGATES} surrogates × {N_FACTORS} factors)...')

spatial_df = spatial_factor_pvalues_all(
    adata_sub,
    factor_key='X_nmf',
    status_key='status',
    coords_obsm='spatial',
    n_surrogates=N_SURROGATES,
    seed=RAND_SEED,
    verbose=True,
)
spatial_df['factor'] = [f'F{i+1}' for i in spatial_df['factor_index']]
print(f'[{time.strftime("%H:%M:%S")}] Spatial null test complete.')


# ── 6. Null distribution for top factor (saved for Panel D) ──────────────────
print(f'[{time.strftime("%H:%M:%S")}] Generating null distribution for '
      f'{top_factor["factor"]}...')
coords_sub   = adata_sub.obsm['spatial']
loadings_sub = adata_sub.obsm['X_nmf'][:, top_fi]
status_sub   = adata_sub.obs['status'].values

p_val_top, obs_coef_top, null_coefs_top = spatial_factor_pvalue(
    loadings_sub, coords_sub, status_sub,
    n_surrogates=N_SURROGATES, seed=RAND_SEED,
)
np.save(OUT_DIR / f'null_coefs_{top_factor["factor"]}.npy', null_coefs_top)
print(f'  Observed coef={obs_coef_top:.3f}  spatial_p={p_val_top:.4f}')


# ── 7. Merge and save CSV ─────────────────────────────────────────────────────
compare_df = sp_df.merge(spatial_df[['factor_index', 'spatial_pval']], on='factor_index')
compare_df['neg_log10_standard'] = -np.log10(compare_df['standard_pval'].clip(1e-300))
compare_df['neg_log10_spatial']  = -np.log10(compare_df['spatial_pval'].clip(1 / (N_SURROGATES + 1)))
compare_df['sig_standard'] = compare_df['standard_pval'] < 0.05
compare_df['sig_spatial']  = compare_df['spatial_pval']  < 0.05

def assign_group(row):
    s, sp = row['sig_standard'], row['sig_spatial']
    if s and sp:      return 'sig (both)'
    if s and not sp:  return 'standard only'
    if not s and sp:  return 'spatial only'
    return 'n.s.'

compare_df['sig_group'] = compare_df.apply(assign_group, axis=1)

compare_df.to_csv(OUT_DIR / 'compare_df.csv', index=False)
print(f'\n=== Significance summary (p < 0.05) ===')
print(f"  Standard : {compare_df['sig_standard'].sum()} factors")
print(f"  Spatial  : {compare_df['sig_spatial'].sum()} factors")
print(compare_df[['factor', 'coef', 'standard_pval', 'spatial_pval', 'auroc', 'sig_group']]
      .sort_values('coef', ascending=False).to_string(index=False))


# ── 8. Figures ────────────────────────────────────────────────────────────────

# Panel A — spatial map of top factor loadings
print(f'[{time.strftime("%H:%M:%S")}] Saving figures...')
loadings = adata.obsm['X_nmf'][:, top_fi]
is_case  = adata.obs['status'].values == 1
in_circ  = adata.obs['in_circle'].values

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, mask, title in [
    (axes[0], ~is_case, 'Control'),
    (axes[1],  is_case, 'Case'),
]:
    sc_ = ax.scatter(
        coords[mask, 0], coords[mask, 1],
        c=loadings[mask], cmap='viridis', s=1, alpha=0.6, rasterized=True,
    )
    ax.set_aspect('equal')
    ax.set_title(f'{title} — {top_factor["factor"]} loading', fontsize=11)
    ax.set_xlabel('X', fontsize=9)
    ax.set_ylabel('Y', fontsize=9)
    plt.colorbar(sc_, ax=ax, label='NMF loading', fraction=0.046, pad=0.04)

axes[1].scatter(
    coords[is_case & in_circ, 0], coords[is_case & in_circ, 1],
    s=3, facecolors='none', edgecolors='red', linewidths=0.4,
    alpha=0.4, label='in-circle (GT)', rasterized=True,
)
axes[1].legend(fontsize=8, frameon=False, markerscale=3)
fig.suptitle(f'Spatial distribution of {top_factor["factor"]} ({TAG})', fontsize=12)
plt.tight_layout()
fig.savefig(OUT_DIR / 'fig_A_spatial_map.png', dpi=150, bbox_inches='tight')
plt.close(fig)


# Panel B — coefficients coloured by significance
def bar_colors(df, pval_col, thresh=0.05):
    colors = []
    for _, row in df.sort_values('factor_index').iterrows():
        if row[pval_col] < thresh and row['coef'] > 0:
            colors.append('#d62728')
        elif row[pval_col] < thresh and row['coef'] < 0:
            colors.append('#1f77b4')
        else:
            colors.append('#aec7e8')
    return colors

x_ticks = np.arange(N_FACTORS)
labels  = [f'F{i+1}' for i in range(N_FACTORS)]

fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), sharey=True)
for ax, pval_col, title in [
    (axes[0], 'standard_pval', 'Standard logistic regression p-value'),
    (axes[1], 'spatial_pval',  f'Spatial null p-value ({N_SURROGATES} surrogates, {N_SUBSAMPLE:,} beads)'),
]:
    sub    = compare_df.sort_values('factor_index')
    coefs  = sub['coef'].values
    colors = bar_colors(sub, pval_col)
    ax.bar(x_ticks, coefs, color=colors, edgecolor='black', linewidth=0.6)
    ax.bar(top_fi, coefs[top_fi], color=colors[top_fi], edgecolor='gold', linewidth=2.0)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(labels, fontsize=9, rotation=45, ha='right')
    ax.set_xlabel('NMF factor', fontsize=10)
    ax.set_title(title, fontsize=10)
    handles = [
        mpatches.Patch(facecolor='#d62728', edgecolor='black', label='sig. positive (p<0.05)'),
        mpatches.Patch(facecolor='#1f77b4', edgecolor='black', label='sig. negative (p<0.05)'),
        mpatches.Patch(facecolor='#aec7e8', edgecolor='black', label='n.s.'),
        mpatches.Patch(facecolor='gray',    edgecolor='gold',   linewidth=2, label='top factor'),
    ]
    ax.legend(handles=handles, fontsize=8, frameon=False)
axes[0].set_ylabel('Logistic regression coefficient', fontsize=10)
fig.suptitle(f'SpatialPeeler factor significance — {TAG}', fontsize=12)
plt.tight_layout()
fig.savefig(OUT_DIR / 'fig_B_coefficients.png', dpi=150, bbox_inches='tight')
plt.close(fig)


# Panel C — standard vs spatial −log10(p) scatter
fig, ax = plt.subplots(figsize=(5, 5))
x = compare_df['neg_log10_standard'].values
y = compare_df['neg_log10_spatial'].values
sc_ = ax.scatter(x, y, c=compare_df['auroc'].values,
                 cmap='RdYlGn', vmin=0.5, vmax=1.0, s=60,
                 edgecolors='black', linewidths=0.6, zorder=3)
plt.colorbar(sc_, ax=ax, label='AUROC (in-circle)', fraction=0.046, pad=0.04)
lim = max(x.max(), y.max()) * 1.05
ax.plot([0, lim], [0, lim], 'k--', linewidth=0.8, label='y = x')
ax.axvline(-np.log10(0.05), color='steelblue', linestyle=':', linewidth=1, label='standard p=0.05')
ax.axhline(-np.log10(0.05), color='tomato',    linestyle=':', linewidth=1, label='spatial p=0.05')
for _, row in compare_df.iterrows():
    ax.annotate(row['factor'],
                (row['neg_log10_standard'], row['neg_log10_spatial']),
                fontsize=7, ha='left', va='bottom',
                xytext=(3, 2), textcoords='offset points')
ax.set_xlabel('Standard  −log₁₀(p)', fontsize=10)
ax.set_ylabel(f'Spatial null  −log₁₀(p)  ({N_SURROGATES} surrogates)', fontsize=10)
ax.set_title('Standard vs spatial significance per factor', fontsize=11)
ax.legend(fontsize=8, frameon=False)
ax.set_xlim(left=0)
ax.set_ylim(bottom=0)
plt.tight_layout()
fig.savefig(OUT_DIR / 'fig_C_scatter.png', dpi=150, bbox_inches='tight')
plt.close(fig)


# Panel D — null distribution for top factor
fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(null_coefs_top, bins=30, color='steelblue', alpha=0.7,
        edgecolor='white', linewidth=0.5, label=f'Null ({N_SURROGATES} surrogates)')
ax.axvline(obs_coef_top,  color='red',    linewidth=2,   label=f'Observed = {obs_coef_top:.2f}')
ax.axvline(-obs_coef_top, color='red',    linewidth=1.2, linestyle='--', label='−|observed|')
ax.axvline(np.percentile(np.abs(null_coefs_top), 95),
           color='orange', linewidth=1.2, linestyle=':', label='95th pct |null|')
ax.set_xlabel('Logistic regression coefficient', fontsize=10)
ax.set_ylabel('Count', fontsize=10)
ax.set_title(f'Null distribution — {top_factor["factor"]}  '
             f'(spatial p = {p_val_top:.3f})', fontsize=11)
ax.legend(fontsize=8, frameon=False)
plt.tight_layout()
fig.savefig(OUT_DIR / 'fig_D_null_distribution.png', dpi=150, bbox_inches='tight')
plt.close(fig)


# Panel E — AUROC by significance group
fig, ax = plt.subplots(figsize=(5, 4))
groups  = ['sig (both)', 'standard only', 'spatial only', 'n.s.']
palette = {'sig (both)': '#2ca02c', 'standard only': '#ff7f0e',
           'spatial only': '#9467bd', 'n.s.': '#aec7e8'}
rng_jit = np.random.default_rng(RAND_SEED)
present = [g for g in groups if g in compare_df['sig_group'].values]
for xi, grp in enumerate(present):
    vals   = compare_df.loc[compare_df['sig_group'] == grp, 'auroc'].dropna().values
    jitter = rng_jit.uniform(-0.15, 0.15, size=len(vals))
    ax.scatter(np.full_like(vals, xi) + jitter, vals,
               color=palette[grp], s=50, edgecolors='black', linewidths=0.5, zorder=3)
    if len(vals):
        ax.hlines(vals.mean(), xi - 0.3, xi + 0.3, colors='black', linewidth=1.5, zorder=4)
ax.axhline(0.5, color='gray', linestyle='--', linewidth=0.8)
ax.set_xticks(range(len(present)))
ax.set_xticklabels(present, fontsize=9)
ax.set_ylabel('AUROC (in-circle, case beads)', fontsize=10)
ax.set_title('Factor AUROC by significance category', fontsize=11)
plt.tight_layout()
fig.savefig(OUT_DIR / 'fig_E_auroc_by_group.png', dpi=150, bbox_inches='tight')
plt.close(fig)

elapsed = time.time() - t_start
print(f'[{time.strftime("%H:%M:%S")}] All done in {elapsed/3600:.2f} h')
print(f'  Results → {OUT_DIR}')
