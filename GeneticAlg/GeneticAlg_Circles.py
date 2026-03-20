#!/usr/bin/env python3
# GA on full HxW binary grid to recover a full circle from top/bottom half-circle seeds.
# No union individual, no reindexing tricks.

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
# 1) Build circle GT and half-circle seeds
# ----------------------------
H, W = 40, 60                 # small & fast; feel free to increase
y, x  = np.ogrid[:H, :W]
cy, cx = H/2, W/2
r = min(H, W) * 0.30          # circle radius

circle  = (x - cx)**2 + (y - cy)**2 <= r**2
top     = circle & (y < cy)
bottom  = circle & (y >= cy)

gt    = np.zeros((H, W), np.uint8)
gt[circle]  = 1
seed1 = np.zeros((H, W), np.uint8)
seed1[bottom] = 1    # bottom half
seed2 = np.zeros((H, W), np.uint8)
seed2[top]    = 1    # top half

gt_vec, seed1_vec, seed2_vec = gt.ravel(), seed1.ravel(), seed2.ravel()
n_genes = H*W

# ----------------------------
# Initial population: ONLY noisy variants of seeds (no union)
# ----------------------------
rng = np.random.default_rng(42)
sol_per_pop = 120 #60
init = [seed1_vec.copy(), seed2_vec.copy()]
while len(init) < sol_per_pop:
    base = seed1_vec if (len(init) % 2 == 0) else seed2_vec
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
    plt.imshow(indiv.reshape(H, W), origin='lower', interpolation='nearest', cmap="gray_r", vmin=0, vmax=1)
    plt.title(f"Initial Population Chromosome {counter}")
    counter += 1
    plt.axis("off")
    plt.show()

# ----------------------------
# Fitness functions
# ---------------------------
# ----------------------------
beta = 8.0  # beta>1 weights recall (penalizes false negatives) more than precision (FN >> FP); try 6–12
b2   = beta*beta
pos  = (gt_vec == 1)
neg  = ~pos

lam_tv  = 0.010            # TV penalty weight ~0.006–0.02
lam_iso = 0.020            # isolated-1 penalty weight ~0.01–0.04


def total_variation(flat):
    """Normalized 4-neighborhood boundary length (discourages ragged edges or noisy boundaries)."""
    img = flat.reshape(H, W)

    # Count horizontal disagreements (between column c and c+1)
    h_edges = np.sum(img[:,1:] != img[:,:-1])

    # Count vertical disagreements (between row r and r+1)
    v_edges = np.sum(img[1:,:] != img[:-1,:])

    edges = h_edges + v_edges
    total_4_neighbor_edges = H*(W-1) + (H-1)*W #
    return edges / total_4_neighbor_edges


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


def fitness_v0(ga, sol, _):
    TP = np.sum((sol == 1) & pos, dtype=np.int32)
    FP = np.sum((sol == 1) & neg, dtype=np.int32)
    FN = np.sum((sol == 0) & pos, dtype=np.int32)
    fbeta = (1+b2)*TP / ((1+b2)*TP + b2*FN + FP + 1e-9)   # ∈ [0,1]
    tv    = total_variation(sol)
    iso   = isolated_ones_fraction(sol)
    return float(fbeta - lam_tv*tv - lam_iso*iso)


def fitness(ga, sol, _):
    TP = np.sum((sol == 1) & pos, dtype=np.int32)
    TN = np.sum((sol == 0) & neg, dtype=np.int32)
    FP = np.sum((sol == 1) & neg, dtype=np.int32)
    FN = np.sum((sol == 0) & pos, dtype=np.int32)
    fbeta = ((1+b2)*TP + TN) / ((1+b2)*TP + TN + b2*FN + FP + 1e-9)   # ∈ [0,1]
    #tv    = total_variation(sol)
    #iso   = isolated_ones_fraction(sol)
    return float(fbeta)

################################################################################
########## Naive fitness functions for comparison ##############################
## fitness functions -> maximize 1/(1 + distance)
def fitness_func_count(ga_instance, solution, solution_idx):
    distance = np.sum(solution != gt_vec) ## total different bits

    # Negating the fitness value to make it increasing rather than decreasing.
    fitness = 1.0 / (1 + distance)
    return fitness

def fitness_func_frac(ga_instance, solution, solution_idx):
    distance = np.sum(solution != gt_vec)/len(gt_vec) ## fraction of different bits over total bits

    # Negating the fitness value to make it increasing rather than decreasing.
    fitness = 1.0 / (1 + distance)
    return fitness


def fitness_func_linear(ga_instance, solution, solution_idx):
    # Hamming distance
    distance = np.sum(solution != gt_vec, dtype=np.int32)
    # Larger is better; perfect match gets n_genes
    return n_genes - distance
################################################################################

solution = seed1_vec.copy()
print("Fitness fitness_func_count of seed1:", fitness_func_count(None, solution, 0))
print("Fitness fitness_func_frac of seed1:", fitness_func_frac(None, solution, 0))
print("Fitness fitness_func_linear of seed1:", fitness_func_linear(None, solution, 0))


solution = seed2_vec.copy()
print("Fitness fitness_func_count of seed2:", fitness_func_count(None, solution, 0))
print("Fitness fitness_func_frac of seed2:", fitness_func_frac(None, solution, 0))
print("Fitness fitness_func_linear of seed2:", fitness_func_linear(None, solution, 0))

solution = initial_population[5]
print("Fitness fitness_func_count of initial_population[16]:", fitness_func_count(None, solution, 0))
print("Fitness fitness_func_frac of initial_population[16]:", fitness_func_frac(None, solution, 0))
print("Fitness fitness_func_linear of initial_population[16]:", fitness_func_linear(None, solution, 0))

solution = gt_vec.copy()
print("Fitness fitness_func_count of ground truth:", fitness_func_count(None, solution, 0))
print("Fitness fitness_func_frac of ground truth:", fitness_func_frac(None, solution, 0))
print("Fitness fitness_func_linear of ground truth:", fitness_func_linear(None, solution, 0))

# ----------------------------
# 4) GA configuration (strong pressure + annealed mutation)
# ----------------------------
ga = pygad.GA(
    fitness_func=fitness_v0,
    num_generations=2000,
    num_parents_mating=round(sol_per_pop/2),
    sol_per_pop=sol_per_pop,
    num_genes=n_genes,
    gene_space=[0, 1],
    gene_type=np.uint8,
    initial_population=initial_population,

    parent_selection_type="tournament",
    K_tournament=9,
    keep_parents=4,

    crossover_type="two_points",           # bit mixing along row-major layout
    crossover_probability=1.0,

    mutation_type="random",
    mutation_probability=0.60,             # start high, anneal over time
    mutation_num_genes=max(1, int(0.08*n_genes)),

    stop_criteria=["reach_1.0", "saturate_600"],
    random_seed=17,
)

def on_generation(g):
    # anneal mutation every 100 generations
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
# 5) Run & visualize
# ----------------------------
t0 = time.time()
ga.run()
elapsed = time.time() - t0
print(f"Elapsed time: {elapsed:.3f} seconds  (workers=1)")

print(f"Elapsed: {time.time()-t0:.2f}s")

best_vec, best_fit, _ = ga.best_solution()
best_img = best_vec.reshape(H, W)

def show(ax, img, title):
    ax.imshow(img, origin="lower", cmap="gray_r", vmin=0, vmax=1)
    ax.set_title(title); ax.axis("off")


#### visualize seeds, GT, best seperately
plt.figure(figsize=(4,3))
show(plt.gca(), seed1, "Seed 1 (bottom half)")
plt.show()

plt.figure(figsize=(4,3))
show(plt.gca(), seed2, "Seed 2 (top half)")
plt.show()

plt.figure(figsize=(4,3))
show(plt.gca(), gt, "Ground Truth (full circle)")
plt.show()

plt.figure(figsize=(4,3))
show(plt.gca(), best_img, f"GA Best Solution; fit={best_fit:.4f}")
plt.show()

### visualize the best solution
result = best_vec.reshape(H, W)
plt.figure(figsize=(6, 4))
plt.title(f"GA Best Solution; fitness={best_fit:.4f}", color='red', fontsize=10)
plt.imshow(result, origin='lower', interpolation='nearest', cmap="gray_r", vmin=0, vmax=1)
plt.axis("off")
plt.show()


plt.figure(figsize=(6,3))
plt.plot(ga.best_solutions_fitness, marker="o", ms=3)
plt.xlabel("Generation"); plt.ylabel("Fitness (F.beta - lambda·TV - lambda·ISO)")
plt.grid(alpha=0.3); plt.show()


### save the GA instance (optional)
#ga_instance.save("pygad_ga_instance.pkl")