#!/usr/bin/env python3
import os
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
from scipy.stats import ttest_ind
from hiddensc.types import AnnData
import pickle

rng = np.random.default_rng(42)

##################################################################################

def de_score_for_mask(mask_bool: np.ndarray,
                      anndata,
                      min_effect: float = 0.1,
                      DE_criterion: str = 't_test',
                      verbose: bool = False):
    """
    Compute a DE-based score for a binary mask over cells (spots).

    Parameters
    ----------
    mask_bool : 1D array-like of length n_obs
        Boolean mask for the 'inside' region.
    anndata : AnnData
        Expression matrix in .X (dense or sparse). n_obs × n_vars.
    min_effect : float
        Threshold for 'mean_diff' mode only: count genes with |mean_in - mean_out| > min_effect.
    DE_criterion : {'mean_diff','t_test','t_sum','cohen_d_sum'}
        - 'mean_diff'  -> integer count of genes with |delta mu| > min_effect
        - 't_test'     -> integer count of genes with Welch test p < 0.05
        - 'cohen_d_sum'-> float, mean(|Cohen's d|) across genes (smooth, size-robust??)
    verbose : bool
        Print a few diagnostics and top genes.

    Returns
    -------
    score : int or float
        Depends on DE_criterion (see above).
    """
    # --- mask & sanity ---
    inside  = np.asarray(mask_bool, dtype=bool).ravel()
    outside = ~inside
    n_in, n_out = int(inside.sum()), int(outside.sum())

    assert inside.shape == (anndata.n_obs,), \
        f"Mask length {inside.size} != n_obs {anndata.n_obs}"
    if n_in == 0 or n_out == 0:
        # degenerate mask
        return 0 if DE_criterion in ('mean_diff', 't_test') else -1e9

    # --- slice matrices ---
    Xin = anndata[inside, :].X
    Xout = anndata[outside, :].X
    if sp.issparse(Xin):  Xin  = Xin.toarray()
    if sp.issparse(Xout): Xout = Xout.toarray()

    if verbose:
        print("Xin/Xout shapes:", Xin.shape, Xout.shape)

    # --- compute stats we’ll reuse ---
    mu_in  = Xin.mean(axis=0)
    mu_out = Xout.mean(axis=0)
    ## sample variance along the columns - ddof=1: divided by N-1
    var_in  = Xin.var(axis=0, ddof=1) 
    var_out = Xout.var(axis=0, ddof=1)
    diff = mu_in - mu_out
    eps = 1e-8

    if DE_criterion == 'mean_diff':
        score = int(np.sum(np.abs(diff) > min_effect))
        if verbose:
            top = np.argsort(-np.abs(diff))[:10]
            print("Top by |delta mu|:", anndata.var_names[top].tolist())
        return score

    elif DE_criterion == 't_test':
        from scipy.stats import ttest_ind
        _, pvals = ttest_ind(Xin, Xout, axis=0, equal_var=False)
        score = int(np.sum(pvals < 0.05))
        if verbose:
            top = np.argsort(pvals)[:10]
            print(f"DE (p<0.05): {score}; top by p:", anndata.var_names[top].tolist())
        return score

    elif DE_criterion == 't_sum':
        # Welch t per gene, then average absolute value (smooth)
        t = diff / np.sqrt(var_in / n_in + var_out / n_out + eps)
        ### TODO: t includes some nans if var_in or var_out is zero?? fix this
        mask = ~np.isnan(t)
        # Use the boolean mask to select only the non-NaN values from the original array
        clean_t = t[mask]
        ### TODO: does it even make sense to average t values??
        score = float(np.mean(np.abs(clean_t)))
        if verbose:
            top = np.argsort(-np.abs(t))[:10]
            print("Top by |t|:", anndata.var_names[top].tolist())
            print('Number of invalid t-statistics (NaN):', np.sum(np.isnan(t)))
        return score

    elif DE_criterion == 'cohen_d_sum':
        # Cohen's d with pooled variance (more size-robust than t)
        s2_pooled = ((n_in - 1) * var_in + (n_out - 1) * var_out) / (n_in + n_out - 2 + eps)
        ### TODO: check for nans in s2_pooled ?? why?? fix this
        mask = ~np.isnan(s2_pooled)
        clean_s2_pooled = s2_pooled[mask]
        clean_diff = diff[mask]
        # Use the boolean mask to select only the non-NaN values from the original array
        d_clean = clean_diff / np.sqrt(clean_s2_pooled + eps)
        d = diff / np.sqrt(s2_pooled + eps) ### need this for verbose top genes
        score = float(np.mean(np.abs(d_clean)))

        if verbose:
            top = np.argsort(-np.abs(d))[:10]
            print("Top by |d|:", anndata.var_names[top].tolist())
        return score

    else:
        raise ValueError("DE_criterion must be one of "
                         "{'mean_diff','t_test','t_sum','cohen_d_sum'}")



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


def preprocess(anndata, lognorm=True, scale=False):
### normalize using scanpy
    if lognorm and 'log1p' not in anndata.uns_keys():
            #sc.pp.normalize_total(anndata, target_sum=1e4)
            sc.pp.log1p(anndata)

    if scale:
        sc.pp.scale(anndata)
    return anndata


def show(ax, img, title):
    ax.imshow(img, origin="lower", cmap="gray_r", vmin=0, vmax=1)
    ax.set_title(title)
    ax.axis("off")

############################################################################################
# Main script: GA to find a spatial mask that maximizes DE genes on ground truth.

# GA over full HxW mask; fitness = DE count (expected direction) on both seeds.
# No union individual, no reindexing tricks.

# ----------------------------
# defining the grid Geometry: circle + seeds as masks
H, W = 200, 300#40, 60
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

#seed_t = ad.read_h5ad("/home/delaram/SpatialPeeler/Data/scCircles/seed_top_cd4_inside_b_outside_H40_W60.h5ad")
#seed_b = ad.read_h5ad("/home/delaram/SpatialPeeler/Data/scCircles/seed_bottom_cd4_inside_b_outside_H40_W60.h5ad")
#ground_truth = ad.read_h5ad("/home/delaram/SpatialPeeler/Data/scCircles/ground_truth_cd4_inside_b_outside_H40_W60.h5ad")


seed_t = ad.read_h5ad("/home/delaram/SpatialPeeler/Data/scCircles/seed_top_cd4_inside_b_outside_H200_W300.h5ad")
seed_b = ad.read_h5ad("/home/delaram/SpatialPeeler/Data/scCircles/seed_bottom_cd4_inside_b_outside_H200_W300.h5ad")
ground_truth = ad.read_h5ad("/home/delaram/SpatialPeeler/Data/scCircles/ground_truth_cd4_inside_b_outside_H200_W300.h5ad")

ground_truth = preprocess(ground_truth, lognorm=True, scale=False)

for marker in marker_list:
    show_marker(seed_t, marker, "(seed top)", H=200, W=300)
    show_marker(seed_b, marker, "(seed bottom)", H=200, W=300)
    show_marker(ground_truth, marker,"(ground truth)", H=200, W=300)

assert seed_t.n_obs == H*W and seed_b.n_obs == H*W, "Obs must be H*W."
assert np.all(seed_t.var_names == seed_b.var_names), "Gene order must match."

adata_top    = seed_t
adata_bottom = seed_b


############################################################################################
############################################################################################

sol_per_pop = 10 #60 120
NUM_GENS = 20#2000
# ----------------------------
# Initial MASK population (two seeds + noisy variants)
# ----------------------------

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

### test/debug the DE function
mask_bool = seed1_vec_mask.astype(bool)
anndata = ground_truth.copy()
min_effect = 0.2
DE_criterion = 't_test'
verbose = True

##############################################
### test the DE function
##############################################

for DE_CRIT in ['mean_diff', 't_test', 't_sum', 'cohen_d_sum']:
    print(f"\nTesting DE_criterion = '{DE_CRIT}'")
    print("DE count/score for seed1:", de_score_for_mask(seed1_vec_mask.astype(bool), 
                                                  ground_truth.copy(),
                                                    min_effect=0.2,
                                                    DE_criterion=DE_CRIT,
                                                    verbose=True))
    print("DE count/score for seed2:", de_score_for_mask(seed2_vec_mask.astype(bool),
                                                    ground_truth.copy(),
                                                        min_effect=0.2,
                                                        DE_criterion=DE_CRIT,
                                                        verbose=True))

    print("DE count/score for GT:   ", de_score_for_mask(gt_vec_mask.astype(bool),
                                                    ground_truth.copy(),
                                                        min_effect=0.2,
                                                        DE_criterion=DE_CRIT,
                                                        verbose=True))
    
# sanity check
print("Initial population DE scores:")
init_pop_scores = {}
for i, indiv in enumerate(initial_population):
    for DE_CRIT in ['mean_diff', 't_test', 't_sum', 'cohen_d_sum']:
        print(f"\nTesting DE_criterion = '{DE_CRIT}'")
        score = de_score_for_mask(indiv.astype(bool), 
                                  ground_truth.copy(),
                                  verbose=True,
                                  DE_criterion=DE_CRIT)
        init_pop_scores[DE_CRIT] = init_pop_scores.get(DE_CRIT, [])
        init_pop_scores[DE_CRIT].append(score)
        print(f"Indiv {i+1:2d}: DE count = {score}")

random_score_dict = {}
### score for a random mask
random_mask = rng.integers(0, 2, size=n_genes, dtype=np.uint8)
for DE_CRIT in ['mean_diff', 't_test', 't_sum', 'cohen_d_sum']:
    print(f"\nTesting DE_criterion = '{DE_CRIT}'")
    random_score = de_score_for_mask(random_mask.astype(bool), 
                                     ground_truth.copy(), 
                                     verbose=True,
                                     DE_criterion=DE_CRIT)
    random_score_dict[DE_CRIT] = random_score
    print(f"Random mask DE count/score = {random_score}")

gt_score_dict = {}
### score for the ground truth mask
for DE_CRIT in ['mean_diff', 't_test', 't_sum', 'cohen_d_sum']:
    print(f"\nTesting DE_criterion = '{DE_CRIT}'")
    gt_score = de_score_for_mask(gt_vec_mask.astype(bool), 
                                ground_truth.copy(), 
                                verbose=True,
                                DE_criterion=DE_CRIT)
    gt_score_dict[DE_CRIT] = gt_score
    print(f"GT mask DE count/score = {gt_score}")                

for DE_CRIT in ['mean_diff', 't_test', 't_sum', 'cohen_d_sum']:
    plt.figure(figsize=(8,4))
    plt.hist(init_pop_scores[DE_CRIT], bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(random_score_dict[DE_CRIT], color='red', linestyle='dashed', linewidth=2, label='Random Mask Score')
    plt.axvline(gt_score_dict[DE_CRIT], color='green', linestyle='dashed', linewidth=2, label='GT Mask Score')
    plt.title(f"Initial Population DE Scores ({DE_CRIT})")
    plt.xlabel("DE Count/Score")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(axis='y', alpha=0.75)
    plt.show()


############################################################################################
# ----------------------------
# fitness function
# ----------------------------

verbose = False
de_mode='t_test'
alpha=0.15     # weight of the size bonus (try 0.05–0.20)
gamma=1
# fitness with TV penalty
lam_tv = 0.006  # tiny; tune 0.004–0.012


def total_variation(flat):
    img = flat.reshape(H, W)
    edges = np.sum(img[:,1:] != img[:,:-1]) + np.sum(img[1:,:] != img[:-1,:])
    return edges / (H*(W-1) + (H-1)*W)


def isolated_ones_fraction(flat, neigh_thresh=2):
    """Fraction of 1s with <2 neighbors in 8-neighborhood (suppresses speckles).
    For a pixel at (i,j), we want the sum of its 8 neighbors:
    (i-1,j-1)  (i-1,j)  (i-1,j+1)
    (i,  j-1)      *     (i,  j+1)
    (i+1,j-1)  (i+1,j)  (i+1,j+1)

    Naive double loop solution (slower):
    for i in range(1, H-1):
        for j in range(1, W-1):
            neigh[i,j] = (
                z[i-1,j-1] + z[i-1,j] + z[i-1,j+1] +
                z[i,  j-1]            + z[i,  j+1] +
                z[i+1,j-1] + z[i+1,j] + z[i+1,j+1]
            )
    vectorized implementation of the loop is provided in this function:
    """

    img = flat.reshape(H, W)
    z = np.pad(img, 1, mode="constant")
    ## sum of neighbors for each pixel 
    neigh = (
        z[0:-2,0:-2] + z[0:-2,1:-1] + z[0:-2,2:] +
        z[1:-1,0:-2]                + z[1:-1,2:] +
        z[2:  ,0:-2] + z[2:  ,1:-1] + z[2:  ,2:]
    )
    
    ### boolean array indicating which pixels are 1s
    ones = (img == 1)

    if ones.sum() == 0:
        return 0.0
    
    ## binary array indicating which 1s are isolated (have <neigh_thresh neighbors)
    isolated_ones = neigh[ones] < neigh_thresh

    return float(np.mean(isolated_ones)) ## fraction of isolated 1s


def fitness_de_v0(ga, sol, _):
    # GA fitness: number of DE genes on ground truth
    # sol is 0/1 vector of length H*W
    mask = sol.astype(bool, copy=False)
    num_DEs = de_score_for_mask(mask, ground_truth.copy(), verbose=False,DE_criterion='t_test')
    
    ## multipy by the number of 1s to incentivize larger circles
    num_ones = int(mask.sum())
    num_DEs = num_DEs * num_ones / n_genes  # normalize by genome length

    return float(num_DEs)


def fitness_de_v1(ga, sol, _,):     
    mask = sol.astype(bool, copy=False)

    # 1) DE signal
    de = de_score_for_mask(mask, ground_truth, DE_criterion=de_mode, verbose=False)

    # 2) Small size incentive 
    frac_ones = float(mask.mean())               # in [0,1]
    size_bonus = frac_ones ** gamma              # e.g., sqrt(frac) if gamma=0.5

    # 3) Combine (keep DE dominant)
    if verbose:
        print(f"DE={de:.1f}, size_bonus={size_bonus:.3f}")
        print(alpha * size_bonus * ground_truth.n_vars)
    score = de + alpha * size_bonus * ground_truth.n_vars  # scale size bonus to gene count
    return float(score)


def fitness_de_v2(ga, sol, _):
    # GA fitness: sum DE counts across both seeds - gamma·TV

    mask = sol.astype(bool, copy=False)
    num_DEs = de_score_for_mask(mask, ground_truth.copy(), 
                                lognorm=True, scale=False, DE_criterion='t_test')
    tv = total_variation(sol)
    n_markers = ground_truth.n_vars
    
    return float(num_DEs - lam_tv * tv * n_markers)



lam_tv = 0.006  # tiny; tune 0.004–0.012



lam_iso = 0.35  # weight of isolated 1s penalty
gamma = 1
def fitness_de(ga, sol, _,):     
    mask = sol.astype(bool, copy=False)

    # 1) DE signal
    de = de_score_for_mask(mask, ground_truth, DE_criterion=de_mode, verbose=False)

    # 2) Small size incentive 
    frac_ones = float(mask.mean())               # in [0,1]
    size_bonus = frac_ones ** gamma              # e.g., sqrt(frac) if gamma=0.5

    # 3) Combine (keep DE dominant)
    if verbose:
        print(f"DE={de:.1f}, size_bonus={size_bonus:.3f}")
    
    #tv = total_variation(sol)
    isolated_frac = isolated_ones_fraction(sol, neigh_thresh=2)
    #score = de * size_bonus 
    score = (de * size_bonus) * (1 - lam_iso * isolated_frac)
    
    return float(score)


'''
### test isolated_ones_fraction values for all initial population
for i, indiv in enumerate(initial_population):
    isolated_frac = isolated_ones_fraction(indiv, neigh_thresh=2)
    print(f"Indiv {i+1:2d}: isolated fraction = {isolated_frac:.2f}")

    fit = fitness_de(None, indiv, None)
    print(f"Indiv {i+1:2d}: fitness = {fit:.2f}")

### test for a random mask
isolated_frac = isolated_ones_fraction(random_mask, neigh_thresh=2)
print(f"Random mask isolated fraction = {isolated_frac:.2f}")
fit = fitness_de(None, random_mask, None)
print(f"Random mask fitness = {fit:.2f}")



### test the fitness function
for i, indiv in enumerate(initial_population):
    fit = fitness_de(None, indiv, None)
    print(f"Indiv {i+1:2d}: fitness = {fit:.2f}")
### score for a random mask
fit = fitness_de(None, random_mask, None)
print(f"Random mask fitness = {fit:.2f}")
### score for the ground truth mask
fit = fitness_de(None, gt_vec_mask, None)
print(f"GT mask fitness = {fit:.2f}")
'''


# ----------------------------
# GA config (strong pressure + annealed mutation)
# ----------------------------
ga = pygad.GA(
    fitness_func=fitness_de,              # change to fitness_de_v2 to include TV penalty
    num_generations=NUM_GENS,
    num_parents_mating=round(len(initial_population)/2),
    #sol_per_pop=sol_per_pop,
    num_genes=n_genes,
    gene_space=[0, 1],
    gene_type=np.uint8,
    initial_population=initial_population,

    parent_selection_type="tournament",
    K_tournament=3, ## change this from 3-10 for selection pressure - smaller means weaker, bigger could cause low parent diversity -> local optima
    keep_parents=round(len(initial_population)/2),

    crossover_type="two_points",
    crossover_probability=1.0,

    mutation_type="random",
    mutation_probability=0,#0.60,
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
#with open("GeneticAlg_scCircles_ga_instance_numGen_"+str(NUM_GENS)+"_sizescaled_DEscore.pkl", "wb") as f:
#    pickle.dump(ga, f)  

best_vec, best_fit, _ = ga.best_solution()
best_img = best_vec.reshape(H, W)

### save the best solution
#np.save("GeneticAlg_scCircles_best_solution_numGen_"+str(NUM_GENS)+"_sizescaled_DEscore.npy", best_vec)

fig, axs = plt.subplots(1, 4, figsize=(12,3))
show(axs[0], seed1_mask, "Seed1 (bottom half)")
show(axs[1], seed2_mask, "Seed2 (top half)")
show(axs[2], gt_mask,    "GT (full circle)")
show(axs[3], best_img,   f"Best (DE fit={best_fit:.1f})")
plt.tight_layout(); 
#plt.savefig("GeneticAlg_scCircles_result_numGen_"+str(NUM_GENS)+"_sizescaled_DEscore.png", dpi=150)
plt.show()

plt.figure(figsize=(6,3))
plt.plot(ga.best_solutions_fitness, marker="o", ms=3)
plt.xlabel("Generation"); 
plt.ylabel("DE fitness")
plt.grid(alpha=0.3); 
plt.title("GA Progress")
#plt.savefig("GeneticAlg_scCircles_progress_numGen_"+str(NUM_GENS)+"_sizescaled_DEscore.png", dpi=150)
plt.show()