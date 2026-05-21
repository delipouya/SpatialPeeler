# SpatialPeeler

** Discovery of disease-associated cell states and spatial regions in spatial transcriptomics**

The method takes spatial transcriptomics data, decomposes it into NMF factors, and iteratively identifies spots / regions enriched in disease-associated signal — "peeling" them away to reveal underlying cell-state changes.

---

## Repository layout

```
SpatialPeeler/
├── SpatialPeeler/          # importable Python package (core logic)
├── Data/                   # raw and preprocessed datasets (not in git)
├── Results/                # fitted model objects (.pkl) and DE results
├── Plots/                  # saved figures, organised by dataset/timepoint
├── GeneticAlg/             # genetic-algorithm mask optimisation scripts & outputs
├── simulation/             # synthetic-data generation scripts
├── hiddensc/               # local copy of the HiDDEN package (case/control prediction)
├── 0–6.*.py / .ipynb       # numbered analysis scripts (see pipeline below)
└── contrastive_FA_benchmark.py  # benchmark against contrastive factor analysis
```

---

## Datasets

| Name | Technology | Directory |
|------|-----------|-----------|
| Remyelin (LPC-injection demyelination model) | Slide-seq | `Data/Remyelin_Slide-seq/` |
| PSC (primary sclerosing cholangitis, liver) | Visium | `Data/PSC_liver/` |
| MERFISH polyomavirus | MERFISH | `Data/MERFISH_polyomavirus/` |

Each dataset has its own set of numbered scripts (suffix `remyelin`, `psc`, or `merfish`).
The Remyelin dataset has multiple versions of the h5ad, distinguished by filename suffixes:

- `_uncropped` — full slide, not spatially cropped
- `_t3`, `_t7`, `_t18` — timepoint subset (days post-injection)
- `_t3_7`, `_t12.18control` — multi-timepoint combinations
- `_PreprocV2` — revised normalisation pipeline
- `_samplewise` — samplewise (per-puck) normalisation
- `_NMF_30` — NMF embedding (K=30) already stored in `obsm["X_nmf"]`

---

## Analysis pipeline

Scripts are numbered to reflect pipeline order. New work is written as Jupyter notebooks (`.ipynb`) for better compatibility with Sanger farm interactive sessions; older steps remain as plain `.py` scripts.

### Step 0 — Data loading / format conversion

| Script | Dataset | What it does |
|--------|---------|-------------|
| `0.remyelin_readData.R` | Remyelin | Reads raw Slide-seq pucks from GEO, assembles metadata, exports to a format readable by Python |
| `0.GEO_visium_to_h5ad.py` | PSC | Downloads Visium data from GEO and converts to `.h5ad` |

### Step 1 — NMF factor decomposition

| Script | Dataset | What it does |
|--------|---------|-------------|
| `1.remyelin_applyNMF.py` | Remyelin | Loads Slide-seq pucks via `SpatialPeeler.importing`, filters/normalises counts, runs sklearn NMF (K=30), stores factor scores in `adata.obsm["X_nmf"]`, saves to `Data/…_NMF_30.h5ad` |
| `1.psc_applyNMF.py` | PSC | Same for Visium PSC data |
| `1.merfish_applyNMF.py` | MERFISH | Same for MERFISH data |

**Key output:** an `.h5ad` file where `adata.obsm["X_nmf"]` holds the (spots × K) factor matrix.

### Step 2 — SpatialPeeler core (single selected factor)

| Script | Dataset | What it does |
|--------|---------|-------------|
| `2.remyelin_spatialpeeler.py` | Remyelin | Loads the NMF h5ad, selects one factor of interest, runs HiDDEN case/control logistic regression (`SpatialPeeler.case_prediction`) to assign each spot a disease-probability score `p_hat`, saves per-factor results to a `.pkl` dict |
| `2.psc_spatialpeeler.py` | PSC | Same |
| `2.merfish_spatialpeeler.py` | MERFISH | Same |

**Key output:** `Results/…_results_factorwise.pkl` — a dict keyed by factor index, each value containing the fitted logistic model and `p_hat` scores per spot.

### Step 2.3 — SpatialPeeler over all factors

| Script | Dataset | What it does |
|--------|---------|-------------|
| `2.3.remyelin_spatialpeeler_allFactors.py` | Remyelin | Loops over every NMF factor automatically and runs the Step 2 logic for each; classifies factors as gain-of-function (GOF) or loss-of-function (LOF) based on their association with the disease condition |
| `2.3.psc_spatialpeeler_allFactors.py` | PSC | Same |

### Step 2.5 — SpatialPeeler with factor clustering (Fclust)

| Script | Dataset | What it does |
|--------|---------|-------------|
| `2.5.remyelin_spatialpeeler_Fclust.py` | Remyelin | Groups NMF factors into clusters (KMeans on the factor loading matrix), then runs SpatialPeeler on each cluster-representative factor. Reduces redundancy when many factors capture similar spatial patterns. Saves to `Results/remyelin_nmf30_hidden_logistic_zeroThr_Fclust_*.pkl` |
| `2.5.psc_spatialpeeler_Fclust.py` | PSC | Same |

### Step 3 — Gene identification

| Script | Dataset | What it does |
|--------|---------|-------------|
| `3.remyelin_geneIdent.py` | Remyelin | Loads SpatialPeeler results and h5ad; uses `SpatialPeeler.gene_identification` to rank genes by correlation / weighted correlation / regression with each factor's `p_hat` pattern; runs g:Profiler pathway enrichment on top-ranked genes; produces Venn diagrams and ranked gene lists |
| `3.psc_geneIdent.py` | PSC | Same |
| `3.merfish_geneIdent.py` | MERFISH | Same |

**Key output:** ranked gene tables and pathway enrichment results, saved under `Results/DE_results/`.

### Step 4 — Spatial masking

| Script | Dataset | What it does |
|--------|---------|-------------|
| `4.remyelin_masking.py` | Remyelin | Thresholds `p_hat` scores to create a binary spatial mask (disease vs. background spots), evaluates mask quality, and generates spatial visualisations |
| `4.remyelin_masking_Fclust.py` / `.ipynb` | Remyelin | Same but applied to the Fclust-based results |
| `4.merfish_masking.py` | MERFISH | Same |

### Step 5 — Genetic algorithm mask optimisation

| Script | Dataset | What it does |
|--------|---------|-------------|
| `5.remyelin_geneticAlg_mask.py` | Remyelin | Uses `pygad` to evolve per-sample binary spatial masks that maximise a DE-gene fitness score (Welch t-test p < 0.05). Initialises each sample's population from binarised `p_hat` masks of GOF factors. Saves the best solution and progress plots to `Results/` and `GeneticAlg/` |

### Step 6 — Visualisation (notebook)

| Notebook | Dataset | What it does |
|----------|---------|-------------|
| `6.remyelin_DE_visualize.ipynb` | Remyelin | Loads `de_results_dict_factorwise.pkl` and the NMF h5ad, log-normalises expression, maps Ensembl IDs to gene symbols via MyGene.info, then plots spatial expression of the top-3 DE genes for each factor × comparison, saving figures into per-comparison subfolders under `Results/DE_results/` |

### Utility notebooks / scripts

| File | What it does |
|------|-------------|
| `remyelin_genecheck.ipynb` | Ad-hoc spatial expression checks for individual genes of interest (e.g. `Mbp`, `Snap25`) |
| `contrastive_FA_benchmark.py` | Benchmarks SpatialPeeler against contrastive factor analysis on memory B-cell datasets |

---

## Core package: `SpatialPeeler/`

This is the importable module shared by all analysis scripts. It is the natural starting point for refactoring into a clean, installable package.

### `importing.py`
Loads raw Slide-seq pucks from disk into `AnnData` objects.

| Function | Description |
|----------|-------------|
| `load_slide_seq_puck(puck_dir, puck_id, ...)` | Loads a single puck: reads count matrix + bead coordinates, stores coordinates in `adata.obsm["spatial"]`, optionally normalises |
| `load_all_slide_seq_data(root_dir, ...)` | Loops over all pucks in a directory and concatenates into one `AnnData` |
| `load_slide_seq_puck_nospatial(...)` | Loads a puck without spatial coordinates (for non-spatial analyses) |

### `case_prediction.py`
Fits logistic regression models to assign per-spot disease probability scores.

| Function | Description |
|----------|-------------|
| `standalone_logistic(X, y)` | Main entry point: fits a logistic regression and returns a results object |
| `standalone_logistic_v1/v2(X, y)` | Earlier variants kept for reproducibility |
| `single_factor_logistic_evaluation(adata, ...)` | Iterates over all NMF factors, fits one logistic model per factor using that factor's scores as the predictor and case/control labels as the outcome; returns per-factor AUC and `p_hat` arrays |

### `gene_identification.py`
Ranks genes by their spatial correlation with a disease-probability pattern.

| Function | Description |
|----------|-------------|
| `pearson_correlation_with_pattern(...)` | Pearson correlation of each gene's expression with a pattern vector |
| `weighted_pearson_correlation_with_pattern(...)` | Weighted version (weights = spot-level confidence / `p_hat`) |
| `regression_with_pattern(...)` | Linear regression of expression on pattern; returns coefficients and p-values |
| `fit_gp_similarity_scores(...)` | Gaussian process-based spatial similarity scoring (slow, high quality) |
| `fit_gp_similarity_scores_fastmode(...)` | Faster approximate GP version |
| `compute_spatial_weights(coords, k, mode)` | Builds a spatial weight matrix (k-NN graph, Geary/Moran mode) |
| `spatial_weighted_correlation_matrix(...)` | Spatially-weighted correlation between each gene's expression and a pattern |

### `plotting.py`
Visualisation utilities for spatial data and model outputs.

| Function | Description |
|----------|-------------|
| `plot_gene_spatial(adata, gene_id, ...)` | Single-gene spatial scatter plot |
| `plot_spatial_nmf(adata, factor_idx, ...)` | Spatial scatter coloured by NMF factor score |
| `plot_spatial_p_hat(adata, sample_id)` | Spatial scatter coloured by HiDDEN `p_hat` |
| `plot_p_hat_vs_nmf_by_sample(...)` | Scatter of `p_hat` vs. NMF score, one panel per sample |
| `plot_logit_p_hat_vs_nmf_by_sample(...)` | Same with logit-transformed `p_hat` |
| `plot_grid(...)` / `plot_grid_upgrade(...)` | Multi-sample grid of spatial maps (main workhorse for figure generation) |
| `plot_grid_naive(...)` / `plot_grid_cliped(...)` | Earlier / value-clipped variants |

### `helpers.py`
Shared constants and utility functions.

| Symbol | Description |
|--------|-------------|
| `RAND_SEED = 28` | Global random seed used throughout |
| `CASE_COND = 1` | Integer label for the case/disease condition |
| `map_ensembl_to_symbol(ids, species)` | Batch Ensembl ID → gene symbol lookup via MyGene.info |
| `map_symbol_to_ensembl(symbols, species)` | Batch gene symbol → Ensembl ID lookup |

### `weightedcorr.py`
Contains the `WeightedCorr` class: implements weighted Pearson and Spearman correlation, used internally by `gene_identification.py`.

---

## Supporting directories

### `GeneticAlg/`
Three script variants for evolving binary spatial masks with `pygad`:

| Script | What it optimises |
|--------|------------------|
| `GeneticAlg_Circles.py` | Circular mask shapes |
| `GeneticAlg_Rectangle.py` | Rectangular mask shapes |
| `GeneticAlg_scCircles.py` | Circles on single-cell-level simulated data |

Outputs (`.npy` best solutions, `.pkl` GA instances, `.png` progress plots) are saved in this directory and in `Results/`.

### `simulation/`
Scripts for generating synthetic spatial transcriptomics datasets to benchmark the method:

| Script | What it simulates |
|--------|-----------------|
| `simulate_spatial_circles.py` | Places cell types (e.g. CD4 T cells) inside a circle and another type (B cells) outside, using real PBMC3k single-cell data as expression templates. Ground truth stored in `Data/ground_truth_cd4_inside_b_outside.h5ad` |
| `simulate_LR.py` | Simulates ligand–receptor signalling patterns across space |

### `hiddensc/`
Local copy of the **HiDDEN** package (case/control prediction for single-cell/spatial data). Imported via `sys.path` manipulation because it is not yet available on PyPI. When packaging SpatialPeeler, this should become a proper pip dependency with a pinned version.

### `Results/`
- `*_results_factorwise.pkl` — dicts of per-factor SpatialPeeler outputs (fitted models, `p_hat` arrays)
- `DE_results/` — differential expression results, organised by dataset/run with per-comparison subfolders
- `GeneticAlg_*.pkl` — saved GA instances for warm-restart of optimisation runs

### `Plots/`
Saved figures, organised by dataset and timepoint (e.g. `remyelin_t3_7_phat/`, `remyelin_t18_NMF/`).

---

## Key conventions

| Convention | Value / rule |
|-----------|-------------|
| Random seed | `RAND_SEED = 28` in every script |
| Case label | `CASE_COND = 1` (disease = integer 1) |
| Spatial coordinates | always stored in `adata.obsm["spatial"]` |
| NMF factors | stored in `adata.obsm["X_nmf"]` (shape: spots × K) |
| Results format | `pickle` dicts keyed by factor index; values are dicts with model object, `p_hat` array, and metadata |
| File naming | `{Dataset}_NMF_{K}_{variant}.h5ad` for data; `{dataset}_nmf{K}_hidden_logistic_{variant}.pkl` for results |
| Paths | currently hardcoded to `/home/delaram/SpatialPeeler/` — needs parameterisation |

---

## Notes for packaging / refactoring

1. **Hardcoded paths** are the main friction point. Every script sets `root_dir`, `file_name`, `outp`, and `folder_name` as string literals. These should become config arguments (CLI flags, a config YAML, or a `paths.py` module).

2. **Repeated boilerplate** — the same ~15-line import block and `RAND_SEED` / `CASE_COND` setup appears in every script. Extract to a `SpatialPeeler.config` module.

3. **`hiddensc` dependency** — the local copy in `hiddensc/` should be replaced with a versioned pip install once HiDDEN is published, or vendored with a clear version pin and licence acknowledgement.

4. **`sys.path` manipulation** — `sys.path.insert(0, root_path)` is used to import `SpatialPeeler` and `hiddensc`. Installing the package with `pip install -e .` would eliminate this.

5. **No `pyproject.toml` yet** — a minimal `pyproject.toml` listing the dependencies in `requirements.txt` is the first step toward making this installable.

6. **`.py` → `.ipynb` transition** — steps 4 and 6 already exist as notebooks. The intention is to convert all numbered scripts to notebooks for reproducibility on the Sanger farm (interactive LSF jobs via Jupyter).
