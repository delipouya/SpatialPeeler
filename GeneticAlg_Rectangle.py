#!/usr/bin/env python3
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import matplotlib.pyplot as plt
import pygad


def plot_img(img, title=""):
    plt.figure(figsize=(6,4))
    plt.imshow(img, origin="lower", cmap="gray_r", vmin=0, vmax=1)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout(); 
    plt.show()

# ----------------------------
# 1) Build rectangle dataset
# ----------------------------
H, W = 20, 30
rect_h, rect_w = 10, 16
cy, cx = H // 2, W // 2
y0, y1 = cy - rect_h // 2, cy + (rect_h - rect_h // 2)
x0, x1 = cx - rect_w // 2, cx + (rect_w - rect_w // 2)

rect_mask = np.zeros((H, W), dtype=bool)
rect_mask[y0:y1, x0:x1] = True

seed1 = np.zeros((H, W), np.uint8) 
seed1[cy:y1, x0:x1] = 1   # bottom half
seed2 = np.zeros((H, W), np.uint8) 
seed2[y0:cy, x0:x1] = 1   # top half
gt    = np.zeros((H, W), np.uint8) 
gt[rect_mask]       = 1   # full rect

seed1_vec, seed2_vec, gt_vec = seed1.ravel(), seed2.ravel(), gt.ravel()
n_genes = gt_vec.size

### visualize seeds & GT separately (optional)
plot_img(seed1, "SEED1 (top half)")
plot_img(seed2, "SEED2 (bottom half)")
plot_img(gt, "GROUND TRUTH")

# ----------------------------
# 2) Fitness function
# ----------------------------
def fitness_func(ga, sol, _):
    tp = np.sum((sol == 1) & (gt_vec == 1))
    tn = np.sum((sol == 0) & (gt_vec == 0))
    fp = np.sum((sol == 1) & (gt_vec == 0))
    fn = np.sum((sol == 0) & (gt_vec == 1))

    # penalize false negatives strongly, false positives mildly
    return tp + 0.5*tn - 2*fn - 1*fp

beta = 9
pos  = (gt_vec == 1)
neg  = ~pos
def fitness_fbeta(ga, sol, idx):
    TP = np.sum((sol == 1) & pos, dtype=np.int32)
    FP = np.sum((sol == 1) & neg, dtype=np.int32)
    FN = np.sum((sol == 0) & pos, dtype=np.int32)
    return float((1+beta)*TP) / ((1+beta)*TP + beta*FN + FP + 1e-9)  # [0,1]

def fitness_fbeta1(ga, sol, idx):
    TP = np.sum((sol == 1) & pos, dtype=np.int32)
    TN = np.sum((sol == 0) & neg, dtype=np.int32)
    FP = np.sum((sol == 1) & neg, dtype=np.int32)
    FN = np.sum((sol == 0) & pos, dtype=np.int32)
    return float((1+beta)*TP + 0.5*(1+beta)*TN) / ((1+beta)*TP + 0.5*(1+beta)*TN + beta*FN + FP + 1e-9) 

# ----------------------------
# 3) Initial population
# ----------------------------
rng = np.random.default_rng(42)
sol_per_pop = 20
init = [seed1_vec.copy(), seed2_vec.copy()]
while len(init) < sol_per_pop:
    base = seed1_vec if (len(init) % 2 == 0) else seed2_vec
    indiv = base.copy()
    m = max(1, int(0.1 * n_genes))  # ~2% flips
    flip = rng.choice(n_genes, size=m, replace=False)
    indiv[flip] ^= 1
    init.append(indiv)
initial_population = np.stack(init, axis=0).astype(np.uint8, copy=False)

counter=1
for indiv in initial_population:
    ## convert indiv to a HxW image and visualize
    img = indiv.reshape(H, W)
    plot_img(img, 'indiv'+str(counter))
    counter += 1
# 4) GA config
# ----------------------------

## calculate fitness of initial population with both functions
for i, indiv in enumerate(initial_population):
    fit1 = fitness_func(None, indiv, i)
    fit2 = fitness_fbeta(None, indiv, i)
    fit3 = fitness_fbeta1(None, indiv, i)
    print(f"Indiv {i+1}: fitness_func={fit1:.2f}, fitness_fbeta={fit2:.4f}, fitness_fbeta1={fit3:.4f}")


ga = pygad.GA(
    fitness_func=fitness_fbeta1,
    num_generations=10000,
    num_parents_mating=20,
    sol_per_pop=sol_per_pop,
    num_genes=n_genes,
    gene_space=[0, 1],
    gene_type=np.uint8,
    initial_population=initial_population,

    parent_selection_type="tournament",
    K_tournament=5,
    keep_parents=2,

    crossover_type="single_point",
    crossover_probability=0.9,

    mutation_type="random",
    mutation_probability=0.2,
    mutation_num_genes=max(1, int(0.02 * n_genes)),

    stop_criteria=["saturate_400"],
)

ga.run()
best_vec, best_fit, _ = ga.best_solution()
best_img = best_vec.reshape(H, W)
plot_img(best_img, 'best image')

# ----------------------------
# Visualize fitness over generations
# ----------------------------

plt.figure(figsize=(6,3))
plt.plot(ga.best_solutions_fitness, marker="o", ms=3)
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.grid(alpha=0.3)
plt.show()











#!/usr/bin/env python3
# Full-grid GA for rectangle union (no union individual, no indexing tricks)

import os
os.environ.setdefault("OMP_NUM_THREADS","1")
os.environ.setdefault("MKL_NUM_THREADS","1")
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
os.environ.setdefault("NUMEXPR_NUM_THREADS","1")

import numpy as np
import matplotlib.pyplot as plt
import pygad
import time

# ----------------------------
# Problem setup: rectangle seeds & ground truth
# ----------------------------
H, W = 20, 30
rect_h, rect_w = 10, 16
cy, cx = H//2, W//2
y0, y1 = cy - rect_h//2, cy + (rect_h - rect_h//2)
x0, x1 = cx - rect_w//2, cx + (rect_w - rect_w//2)

seed1 = np.zeros((H,W), np.uint8); seed1[cy:y1, x0:x1] = 1   # bottom half
seed2 = np.zeros((H,W), np.uint8); seed2[y0:cy, x0:x1] = 1   # top half
gt    = np.zeros((H,W), np.uint8); gt[y0:y1, x0:x1]   = 1    # full rect

seed1_vec, seed2_vec, gt_vec = seed1.ravel(), seed2.ravel(), gt.ravel()
n_genes = H*W

# ----------------------------
# Fitness: Fβ (β² weighting) minus small smoothness penalties
# ----------------------------
beta = 10.0         # recall weight (penalize FN >> FP); try 6–12
b2   = beta*beta
pos  = (gt_vec == 1)
neg  = ~pos

lam_tv  = 0.010    # TV penalty weight (edges)   ~0.006–0.02
lam_iso = 0.020    # isolated-1 penalty weight   ~0.01–0.04 - 0.05

def total_variation(flat):
    """Normalized 4-neighborhood boundary length."""
    img = flat.reshape(H, W)
    edges = np.sum(img[:,1:] != img[:,:-1]) + np.sum(img[1:,:] != img[:-1,:])
    return edges / (H*(W-1) + (H-1)*W)

def isolated_ones_fraction(flat):
    """Fraction of 1s having <2 neighbors in 8-neighborhood (suppresses speckles)."""
    img = flat.reshape(H, W)
    z = np.pad(img, 1, mode="constant")
    neigh = (
        z[0:-2,0:-2] + z[0:-2,1:-1] + z[0:-2,2:] +
        z[1:-1,0:-2]                + z[1:-1,2:] +
        z[2:  ,0:-2] + z[2:  ,1:-1] + z[2:  ,2:]
    )
    ones = (img == 1)
    if ones.sum() == 0:
        return 0.0
    return float(np.mean(neigh[ones] < 2))

def fitness(ga, sol, _):
    TP = np.sum((sol == 1) & pos, dtype=np.int32)
    FP = np.sum((sol == 1) & neg, dtype=np.int32)
    FN = np.sum((sol == 0) & pos, dtype=np.int32)
    fbeta = (1+b2)*TP / ((1+b2)*TP + b2*FN + FP + 1e-9)   # [0,1]
    tv    = total_variation(sol)
    iso   = isolated_ones_fraction(sol)
    return float(fbeta - lam_tv*tv - lam_iso*iso)

# ----------------------------
# Initial population: ONLY noisy variants of the two seeds
# ----------------------------
rng = np.random.default_rng(42)
sol_per_pop = 120
init = [seed1_vec.copy(), seed2_vec.copy()]
while len(init) < sol_per_pop:
    base = seed1_vec if (len(init) % 2 == 0) else seed2_vec
    indiv = base.copy()
    m = max(1, int(0.10 * n_genes))                 # flip ~10% bits for diversity
    flip = rng.choice(n_genes, size=m, replace=False)
    indiv[flip] ^= 1
    init.append(indiv)
initial_population = np.stack(init, axis=0).astype(np.uint8, copy=False)

# ----------------------------
# GA configuration (strong pressure + annealed mutation)
# ----------------------------
ga = pygad.GA(
    fitness_func=fitness,
    num_generations=2000,
    num_parents_mating=60,
    sol_per_pop=sol_per_pop,
    num_genes=n_genes,
    gene_space=[0, 1],
    gene_type=np.uint8,
    initial_population=initial_population,

    parent_selection_type="tournament",
    K_tournament=9,
    keep_parents=4,

    crossover_type="two_points",          # good on row-major genomes
    crossover_probability=1.0,

    mutation_type="random",
    mutation_probability=0.60,            # start high, then anneal
    mutation_num_genes=max(1, int(0.08*n_genes)),

    stop_criteria=["reach_1.0", "saturate_600"],
    random_seed=17,
)

def on_generation(g):
    # anneal every 100 generations
    if g.generations_completed > 0 and g.generations_completed % 100 == 0:
        g.mutation_probability = max(0.05, g.mutation_probability * 0.80)
        g.mutation_num_genes   = max(1, int(g.mutation_num_genes * 0.80))

    if g.generations_completed in (1,5,10) or g.generations_completed % 100 == 0:
        best, fit, _ = g.best_solution()
        TP = np.sum((best == 1) & pos); FP = np.sum((best == 1) & neg)
        FN = np.sum((best == 0) & pos)
        fbeta = (1+b2)*TP / ((1+b2)*TP + b2*FN + FP + 1e-9)
        tv  = total_variation(best)
        iso = isolated_ones_fraction(best)
        print(f"gen {g.generations_completed:4d} | Fβ={fbeta:.4f} TV={tv:.4f} ISO={iso:.4f} "
              f"| fit={fit:.4f} | mp={g.mutation_probability:.3f} mk={g.mutation_num_genes}")
ga.on_generation = on_generation

# ----------------------------
# Run & report
# ----------------------------
t0 = time.time()
ga.run()
print(f"Elapsed: {time.time()-t0:.2f}s")

best_vec, best_fit, _ = ga.best_solution()
best_img = best_vec.reshape(H, W)

# ----------------------------
# Visualize
# ----------------------------
def show(ax, img, title):
    ax.imshow(img, origin="lower", cmap="gray_r", vmin=0, vmax=1)
    ax.set_title(title); ax.axis("off")

fig, axs = plt.subplots(1, 4, figsize=(12,3))
show(axs[0], seed1, "Seed1")
show(axs[1], seed2, "Seed2")
show(axs[2], gt,    "GT")
show(axs[3], best_img, "Best")
plt.tight_layout(); plt.show()

plt.figure(figsize=(6,3))
plt.plot(ga.best_solutions_fitness, marker="o", ms=3)
plt.xlabel("Generation"); plt.ylabel("Fitness (Fβ - λ·TV - λ·ISO)")
plt.grid(alpha=0.3); plt.show()