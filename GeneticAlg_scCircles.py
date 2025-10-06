#!/usr/bin/env python3
import os

from hiddensc.types import AnnData
os.environ.setdefault("OMP_NUM_THREADS","1")
os.environ.setdefault("MKL_NUM_THREADS","1")
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
os.environ.setdefault("NUMEXPR_NUM_THREADS","1")
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import anndata as ad
import pygad
import time
import scanpy as sc


rng = np.random.default_rng(42)

# sanity images
def show_marker(adata_like, gene, title_suffix="", H=200, W=300):

    if gene not in adata_like.var_names:
        print(f"[warn] {gene} not in var_names")
        return
    gi = adata_like.var_names.get_loc(gene)
    col = adata_like.X[:, gi]
    if sp.issparse(col):
        col = col.toarray().ravel()
    else:
        col = np.asarray(col).ravel()
    #H, W = 200, 300        # grid height × width
    img = col.reshape(H, W)

    plt.figure(figsize=(6, 4))
    plt.imshow(img, origin="lower", interpolation="nearest")
    plt.title(f"{gene} expression {title_suffix}".strip())
    plt.axis("off")
    plt.colorbar(label="expression")
    plt.show()



# GA over full HxW mask; fitness = DE count (expected direction) on both seeds.
# No union individual, no reindexing tricks.

# ----------------------------
# defining the grid Geometry: circle + seeds as masks
H, W = 40, 60
y, x  = np.ogrid[:H, :W]
cy, cx = H/2, W/2
r = min(H, W) * 0.30

circle = (x - cx)**2 + (y - cy)**2 <= r**2
top    = circle & (y < cy)
bottom = circle & (y >= cy)

gt_mask    = np.zeros((H, W), np.uint8)
gt_mask[circle]  = 1 ## sum(circle.ravel())=441
seed1_mask = np.zeros((H, W), np.uint8)
seed1_mask[bottom] = 1  # bottom half
seed2_mask = np.zeros((H, W), np.uint8)
seed2_mask[top]    = 1  # top half

gt_vec_mask    = gt_mask.ravel() #gt_mask.ravel().sum()=441
seed1_vec_mask = seed1_mask.ravel()
seed2_vec_mask = seed2_mask.ravel()
n_genes = H * W  # genome length (binary mask over all spots)

### get_vec should have 2400 elements -> 1959 zeros and 441 ones


# ----------------------------
#  Load seed expression fields
# Each h5ad should be a grid of H*W observations (spots), genes in var.

# T-cell markers expected higher INSIDE; B-cell markers expected higher OUTSIDE
t_markers_all = ["IL32","CD2","LTB","TRAC","CD27","CD28"]
b_markers_all = ["MS4A1","CD79A","CD79B","CD19"]
marker_list = ['IL32','CD2','MS4A1','CD19','PRF1','GZMA','GZMK',
              'NKG7','PDCD1','TIGIT','CD79A','CD79B']

seed_t = ad.read_h5ad("/home/delaram/SpatialPeeler/Data/scCircles/seed_top_cd4_inside_b_outside_H40_W60.h5ad")
seed_b = ad.read_h5ad("/home/delaram/SpatialPeeler/Data/scCircles/seed_bottom_cd4_inside_b_outside_H40_W60.h5ad")
ground_truth = ad.read_h5ad("/home/delaram/SpatialPeeler/Data/scCircles/ground_truth_cd4_inside_b_outside_H40_W60.h5ad")

for marker in marker_list:
    show_marker(seed_t, marker, "(seed top)", H=40, W=60)
    show_marker(seed_b, marker, "(seed bottom)", H=40, W=60)
    show_marker(ground_truth, marker,"(ground truth)", H=40, W=60)

assert seed_t.n_obs == H*W and seed_b.n_obs == H*W, "Obs must be H*W."
assert np.all(seed_t.var_names == seed_b.var_names), "Gene order must match."

adata_top    = seed_t
adata_bottom = seed_b

# ----------------------------
# Initial MASK population (two seeds + noisy variants)
# ----------------------------
rng = np.random.default_rng(42)
sol_per_pop = 20 #60 120


init = [seed1_vec_mask.copy(), seed2_vec_mask.copy()]
while len(init) < sol_per_pop:
    base = seed1_vec_mask if (len(init) % 2 == 0) else seed2_vec_mask
    indiv = base.copy()

    # Flip a handful of bits (~5% of genome) to avoid duplicates
    m = max(1, int(0.10 * n_genes))              # flip ~10% bits for diversity
    flip = rng.choice(n_genes, size=m, replace=False) ### flipping indices
    indiv[flip] = 1 - indiv[flip]
    init.append(indiv)
initial_population = np.stack(init, axis=0).astype(np.uint8, copy=False)

### remove the first two (seeds) to visualize only the random ones
#initial_population = initial_population[2:]
counter=1
for indiv in initial_population:
    ### visualize the chromosome
    plt.figure(figsize=(6, 4))
    plt.imshow(indiv.reshape(H, W), origin='lower', interpolation='nearest', 
        cmap="gray_r", vmin=0, vmax=1)
    plt.title(f"Initial Population Chromosome {counter}")
    counter += 1
    plt.axis("off")
    plt.show()


############################################################################################
# ----------------------------
# fitness function
# ----------------------------

# TODO: include polarity into obj function?
# +1 for T markers (want inside>outside), -1 for B markers (want inside<outside)

def de_score_for_mask(mask_bool: np.ndarray, anndata, min_effect=0.1, 
                         lognorm=True, scale=False, 
                         DE_criterion='t_test', verbose=False) -> int:
    """
    DE score for a binary mask on the ground truth AnnData.
    mask_bool: 1D boolean array of length n_obs 
    anndata: AnnData with raw counts
    min_effect: minimum absolute mean difference to count as DE (for 'mean_diff' criterion)
    lognorm: whether to log-normalize the data
    scale: whether to z-score the data
    DE_criterion: 'mean_diff' or 't_test'

    Returns the count of DE genes with abs(mean_in - mean_out) > min_effect
    """
    # Coerce to 1D boolean mask
    inside  = np.asarray(mask_bool, dtype=bool).ravel()
    outside = np.logical_not(inside)

    # Sanity checks
    assert inside.shape == (anndata.n_obs,), \
        f"Mask length {inside.size} != n_obs {anndata.n_obs}"
    if inside.sum() == 0 or outside.sum() == 0:
        return 0

    ### normalize using scanpy
    if lognorm and 'log1p' not in anndata.uns_keys():
            #sc.pp.normalize_total(anndata, target_sum=1e4)
            sc.pp.log1p(anndata)

    if scale:
        sc.pp.scale(anndata)

    ### end normalize
    # Extract inside/outside expression matrices
    gt_in  = anndata[inside, :].X
    gt_out = anndata[outside, :].X

    # Denseify if sparse
    if sp.issparse(gt_in):  
        gt_in = gt_in.toarray()
    if sp.issparse(gt_out):
        gt_out = gt_out.toarray()

    # Quick sanity
    if verbose:
        print(gt_in.shape, gt_out.shape)  # expect (inside_sum, n_genes) and (outside_sum, n_genes)

    num_DE = None
    if DE_criterion == 'mean_diff':
        # Mean difference criterion
        mean_in  = gt_in.mean(axis=0)
        mean_out = gt_out.mean(axis=0)
        diff = mean_in - mean_out
        num_DE = int(np.sum(np.abs(diff) > min_effect))
        
        top_DE_genes = np.argsort(-np.abs(diff))[:10]
        if verbose:
            print("Top DE genes (by mean diff):", anndata.var_names[top_DE_genes].tolist())

    elif DE_criterion == 't_test':
        # t-test criterion
        from scipy.stats import ttest_ind
        _, pvals = ttest_ind(gt_in, gt_out, axis=0, equal_var=False)
        num_DE = int(np.sum(pvals < 0.05))

        if verbose:
            if num_DE > 0:
                print(f"Found {num_DE} DE genes (p<0.05)")
            else:
                print("No DE genes found (p<0.05)")

        top_DE_genes = np.argsort(pvals)[:10]
        if verbose:
            print("Top DE genes (by p-value):", anndata.var_names[top_DE_genes].tolist())


    return num_DE

##############################################
### test the DE function
##############################################
print("DE count for seed1:", de_score_for_mask(seed1_vec_mask.astype(bool), 
                                                  ground_truth.copy(), verbose=True,
                                                  min_effect=0.2, lognorm=True, scale=False,
                                                  DE_criterion='mean_diff'))
print("DE count for seed2:", de_score_for_mask(seed2_vec_mask.astype(bool), 
                                                  ground_truth.copy(), verbose=True,
                                                  min_effect=0.2, lognorm=True, scale=False,
                                                  DE_criterion='mean_diff'))
print("DE count for GT:   ", de_score_for_mask(gt_vec_mask.astype(bool), 
                                                  ground_truth.copy(), verbose=True,
                                                  min_effect=0.2, lognorm=True, scale=False,
                                                  DE_criterion='mean_diff'))


### test the function
print("DE count for seed1:", de_score_for_mask(seed1_vec_mask.astype(bool), 
                                                  ground_truth.copy(), verbose=True,
                                                  min_effect=0.2, lognorm=True, scale=False, 
                                                  DE_criterion='t_test'))
print("DE count for seed2:", de_score_for_mask(seed2_vec_mask.astype(bool), 
                                                  ground_truth.copy(), verbose=True,
                                                  min_effect=0.2, lognorm=True, scale=False,
                                                  DE_criterion='t_test'))
print("DE count for GT:   ", de_score_for_mask(gt_vec_mask.astype(bool), 
                                                  ground_truth.copy(), verbose=True,
                                                  min_effect=0.2, lognorm=True, scale=False,
                                                  DE_criterion='t_test'))

# sanity check
print("Initial population DE scores:")
init_pop_scores = []
for i, indiv in enumerate(initial_population):
    score = de_score_for_mask(indiv.astype(bool), ground_truth.copy(), verbose=True,
                              lognorm=True, scale=False, DE_criterion='t_test')
    init_pop_scores.append(score)
    print(f"Indiv {i+1:2d}: DE count = {score}")

### score for a random mask
random_mask = rng.integers(0, 2, size=n_genes, dtype=np.uint8)
random_score = de_score_for_mask(random_mask.astype(bool), ground_truth.copy(), verbose=True,
                                    lognorm=True, scale=False, DE_criterion='t_test')

### histogram of initial population scores
plt.figure(figsize=(6,4))
plt.hist(init_pop_scores, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
plt.title("Initial Population DE Scores")
plt.xlabel("DE Count")
plt.ylabel("Frequency")
plt.grid(axis='y', alpha=0.75)
plt.show()
##############################################


def fitness_de(ga, sol, _):
    # GA fitness: number of DE genes on ground truth
    # sol is 0/1 vector of length H*W
    mask = sol.astype(bool, copy=False)
    num_DEs = de_score_for_mask(mask, ground_truth.copy(), verbose=False,
                                lognorm=True, scale=False, DE_criterion='t_test')
    return float(num_DEs)

def total_variation(flat):
    img = flat.reshape(H, W)
    edges = np.sum(img[:,1:] != img[:,:-1]) + np.sum(img[1:,:] != img[:-1,:])
    return edges / (H*(W-1) + (H-1)*W)

# fitness with TV penalty
lam_tv = 0.006  # tiny; tune 0.004–0.012
def fitness_de_v2(ga, sol, _):
    # GA fitness: sum DE counts across both seeds - λ·TV

    mask = sol.astype(bool, copy=False)
    num_DEs = de_score_for_mask(mask, ground_truth.copy(), 
                                lognorm=True, scale=False, DE_criterion='t_test')
    tv = total_variation(sol)
    n_markers = ground_truth.n_vars
    
    return float(num_DEs - lam_tv * tv * n_markers)


# ----------------------------
# GA config (strong pressure + annealed mutation)
# ----------------------------
ga = pygad.GA(
    fitness_func=fitness_de,              # change to fitness_de_v2 to include TV penalty
    num_generations=2000,
    num_parents_mating=round(len(initial_population)/2),
    #sol_per_pop=sol_per_pop,
    num_genes=n_genes,
    gene_space=[0, 1],
    gene_type=np.uint8,
    initial_population=initial_population,

    parent_selection_type="tournament",
    K_tournament=9,
    keep_parents=4,

    crossover_type="two_points",
    crossover_probability=1.0,

    mutation_type="random",
    mutation_probability=0.60,
    mutation_num_genes=max(1, int(0.08*n_genes)),

    stop_criteria=["saturate_600"],
    random_seed=17,
)

def on_generation(g):
    if g.generations_completed > 0 and g.generations_completed % 100 == 0:
        g.mutation_probability = max(0.05, g.mutation_probability * 0.80)
        g.mutation_num_genes   = max(1, int(g.mutation_num_genes * 0.80))

    if g.generations_completed in (1,5,10) or g.generations_completed % 100 == 0:
        best, fit, _ = g.best_solution()
        #tv = total_variation(best)
        #print(f"gen {g.generations_completed:4d} | DE_fit={fit:.2f} | TV={tv:.4f} "
        #      f"| mp={g.mutation_probability:.3f} mk={g.mutation_num_genes}")
        print(f"gen {g.generations_completed:4d} | DE_fit={fit:.2f}"
              f"| mp={g.mutation_probability:.3f} mk={g.mutation_num_genes}")
ga.on_generation = on_generation

# ----------------------------
# Run & visualize
# ----------------------------
t0 = time.time(); 
ga.run(); 
print(f"Elapsed: {time.time()-t0:.2f}s")
### save the GA instance
import pickle
with open("GeneticAlg_scCircles_ga_instance_numGen_2000.pkl", "wb") as f:
    pickle.dump(ga, f)  

best_vec, best_fit, _ = ga.best_solution()
best_img = best_vec.reshape(H, W)

### save the best solution
np.save("GeneticAlg_scCircles_best_solution_numGen_2000.npy", best_vec)
def show(ax, img, title):
    ax.imshow(img, origin="lower", cmap="gray_r", vmin=0, vmax=1)
    ax.set_title(title)
    ax.axis("off")

fig, axs = plt.subplots(1, 4, figsize=(12,3))
show(axs[0], seed1_mask, "Seed1 (bottom half)")
show(axs[1], seed2_mask, "Seed2 (top half)")
show(axs[2], gt_mask,    "GT (full circle)")
show(axs[3], best_img,   f"Best (DE fit={best_fit:.1f})")
plt.tight_layout(); 
plt.savefig("GeneticAlg_scCircles_result_numGen_2000.png", dpi=150)
plt.show()

plt.figure(figsize=(6,3))
plt.plot(ga.best_solutions_fitness, marker="o", ms=3)
plt.xlabel("Generation"); 
plt.ylabel("DE fitness")
plt.grid(alpha=0.3); 
plt.title("GA Progress")
plt.savefig("GeneticAlg_scCircles_progress_numGen_2000.png", dpi=150)
plt.show()