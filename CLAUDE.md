# SpatialPeeler

**Iterative discovery of disease-associated cell states and interactions in spatial transcriptomics**

A method development project at the Wellcome Sanger Institute. The goal is to identify disease-associated spatial regions and cell states from spatial transcriptomics data.

## Datasets

| Name | Technology | Directory |
|------|-----------|-----------|
| Remyelin | Slide-seq | `Data/Remyelin_Slide-seq/` |
| PSC (liver) | Visium | `Data/PSC_liver/` |
| MERFISH polyomavirus | MERFISH | `Data/MERFISH_polyomavirus/` |

## Analysis pipeline

Scripts are numbered to reflect pipeline ordering. Separate scripts exist per dataset. The project is transitioning from numbered `.py` scripts to Jupyter notebooks (better compatibility with Sanger farm interactive sessions).

| Step | Description |
|------|-------------|
| `0.*` | Data loading / format conversion (GEO → h5ad) |
| `1.*` | NMF factor decomposition |
| `2.*` | SpatialPeeler core (factor-based spatial peeling) |
| `2.3.*` | SpatialPeeler — all factors |
| `2.5.*` | SpatialPeeler — factor clustering (Fclust) |
| `3.*` | Gene identification |
| `4.*` | Masking |
| `5.*` | Genetic algorithm / optimisation |

Dataset suffixes: `remyelin`, `psc`, `merfish`.

## Core module: `SpatialPeeler/`

The `SpatialPeeler/` subdirectory is the importable Python package:

- `case_prediction.py` — case/control prediction logic
- `gene_identification.py` — disease-associated gene identification
- `helpers.py` — shared utilities, constants (`RAND_SEED=28`, `CASE_COND=1`)
- `importing.py` — data import helpers
- `plotting.py` — visualisation functions
- `weightedcorr.py` — weighted correlation utilities

Key dependencies: `scanpy`, `anndata`, `scvi`, `hiddensc`, `numpy`, `pandas`, `sklearn`, `mygene`.

## Key outputs

Results are stored in `Results/` (`.pkl` files for fitted models) and `Plots/` for figures. The `GeneticAlg/` directory holds genetic algorithm runs.

## Conventions

- Follow the existing script numbering when adding new analysis steps.
- Prefer notebooks over plain `.py` scripts for new work.
- Random seed: `RAND_SEED = 28`.
- The `hiddensc` package (in `hiddensc/`) is a local dependency — import it via `sys.path` manipulation as done in `helpers.py`.
