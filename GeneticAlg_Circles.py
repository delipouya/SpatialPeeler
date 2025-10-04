#!/usr/bin/env python3
# --- optional: keep BLAS to 1 thread per proc (safer if you parallelize later) ---
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import matplotlib.pyplot as plt
import pygad
import time
# ----------------------------
# 1) Binary grids (0/1)
# ----------------------------
H, W = 10, 20#20, 30
y, x = np.ogrid[:H, :W]
cy, cx, r = H/2, W/2, min(H, W) * 0.25

circle = (x - cx)**2 + (y - cy)**2 <= r**2
top    = circle & (y < cy)
bottom = circle & (y >= cy)

seed1 = np.zeros((H, W), dtype=np.uint8) 
seed1[bottom] = 1  # bottom half
seed2 = np.zeros((H, W), dtype=np.uint8) 
seed2[top]    = 1  # top half
gt    = np.zeros((H, W), dtype=np.uint8) 
gt[circle]    = 1  # full circle


# Visualize seeds & GT (optional)
for title, img in [("SEED1 (bottom half)", seed1),
                   ("SEED2 (top half)",    seed2),
                   ("Ground Truth (full)", gt)]:
    plt.figure(figsize=(6, 4))
    plt.imshow(img, origin='lower', interpolation='nearest', cmap="gray_r", vmin=0, vmax=1)
    plt.title(title); plt.axis("off"); plt.show()



# Flatten to genome vectors
seed1_vec = seed1.ravel()
seed2_vec = seed2.ravel()
gt_vec    = gt.ravel()
n_genes   = gt_vec.size  # 600


# ----------------------------
# 2) Initial population: start from the two seeds + tiny diversifications
# ----------------------------
rng = np.random.default_rng(123)
sol_per_pop = 20
initial_population = [seed1_vec.copy(), seed2_vec.copy()]
while len(initial_population) < sol_per_pop+2:
    base = seed1_vec if (len(initial_population) % 2 == 0) else seed2_vec
    indiv = base.copy()
    # Flip a handful of bits (~5% of genome) to avoid duplicates
    m = max(1, int(0.1 * n_genes))  # ~60 flips #int(0.01 * n_genes)
    flip_idx = rng.choice(n_genes, size=m, replace=False)
    indiv[flip_idx] = 1 - indiv[flip_idx]
    initial_population.append(indiv)
initial_population = np.stack(initial_population, axis=0).astype(np.uint8, copy=False)

### remove the first two (seeds) to visualize only the random ones
initial_population = initial_population[2:]

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
# 3) Fitness: maximize 1/(1 + distance)
# ----------------------------

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

solution = seed1_vec.copy()
print("Fitness fitness_func_count of seed1:", fitness_func_count(None, solution, 0))
print("Fitness fitness_func_frac of seed1:", fitness_func_frac(None, solution, 0))
print("Fitness fitness_func_linear of seed1:", fitness_func_linear(None, solution, 0))


solution = seed2_vec.copy()
print("Fitness fitness_func_count of seed2:", fitness_func_count(None, solution, 0))
print("Fitness fitness_func_frac of seed2:", fitness_func_frac(None, solution, 0))
print("Fitness fitness_func_linear of seed2:", fitness_func_linear(None, solution, 0))

solution = initial_population[16]
print("Fitness fitness_func_count of initial_population[16]:", fitness_func_count(None, solution, 0))
print("Fitness fitness_func_frac of initial_population[16]:", fitness_func_frac(None, solution, 0))
print("Fitness fitness_func_linear of initial_population[16]:", fitness_func_linear(None, solution, 0))

solution = gt_vec.copy()
print("Fitness fitness_func_count of ground truth:", fitness_func_count(None, solution, 0))
print("Fitness fitness_func_frac of ground truth:", fitness_func_frac(None, solution, 0))
print("Fitness fitness_func_linear of ground truth:", fitness_func_linear(None, solution, 0))


# ----------------------------
# 4) GA configuration (defaults, but tuned to be quick)
# ----------------------------

# Sanity: ensure seeds are binary {0,1}
assert set(np.unique(initial_population)) <= {0,1}, "Init pop must be binary."


# --- GA config: stronger selection + more chances to cut near the row boundary ---
#sol_per_pop = 60               # small genome -> we can afford a larger pop
#num_parents = 30

ga_instance = pygad.GA(
    fitness_func=fitness_func_linear,
    num_generations=100000,
    num_parents_mating=10,
    sol_per_pop=len(initial_population),
    num_genes=n_genes,

    gene_space=[0, 1],
    gene_type=np.uint8,
    initial_population=initial_population,

    # Selection with pressure
    parent_selection_type='sss',#"tournament",
    #K_tournament=5,
    keep_parents=0, #2

    # One-point cut has a real chance to separate top vs bottom when pop is bigger
    crossover_type="single_point",
    crossover_probability=0.9,

    # Binary mutation: flip a concrete number of bits
    mutation_type="random",
    mutation_probability=0.4,                        # mutate 40% of offspring
    mutation_percent_genes=20,                     # ~10% genes per mutated offspring
    #mutation_num_genes=max(1, int(0.08 * n_genes)),  # flip ~8% of bits per mutated indiv

    stop_criteria=["reach_"+str(n_genes), "saturate_5000"], # reach_1.0 for fitness_func_frac
)
# Optional progress log
def on_generation(ga):
    if ga.generations_completed in (1, 5, 10) or ga.generations_completed % 50 == 0:
        best = ga.best_solution()[0]
        ham  = int(np.sum(best != gt_vec))
        print(f"gen {ga.generations_completed:4d} | hamming={ham} | fitness={ga.best_solution()[1]:.6f}")
ga_instance.on_generation = on_generation

# ----------------------------
# 6) Run GA
# ----------------------------

# Lightweight progress log so you can bail early if it's converged enough
def on_generation(ga):
    gc = ga.generations_completed
    if gc == 1 or gc % 500 == 0:
        best_fit = ga.best_solution()[1]
        # convert back to mean absolute error for interpretability
        current_best = ga.best_solution()[0]
        mae = float(np.abs(current_best - gt_vec).mean()) ## TODO: this needs to be changed
        print(f"Gen {gc:6d} | best fitness={best_fit:.4f} | MAE={mae:.6f}")
ga_instance.on_generation = on_generation


# ----------------------------
# Run & time
# ----------------------------
start_time = time.time()
ga_instance.run()
elapsed = time.time() - start_time
print(f"Elapsed time: {elapsed:.3f} seconds  (workers=1)")

### save the GA instance (optional)
#ga_instance.save("pygad_ga_instance.pkl")
# ----------------------------
# Plot & save fitness curve
# ----------------------------
# after ga.run()
plt.figure(figsize=(6,4))
plt.plot(ga_instance.best_solutions_fitness, marker="o", markersize=3)
plt.xlabel("Generation")
plt.ylabel("Best Fitness")
plt.title("Fitness over Generations")
plt.grid(alpha=0.3)
plt.show()
plt.close()

# ----------------------------
# Best solution & reconstruction
# ----------------------------
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print(f"Fitness value of the best solution = {solution_fitness}")
print(f"Index of the best solution        = {solution_idx}")

### visualize the best solution
result = solution.reshape(H, W)
plt.figure(figsize=(6, 4))
plt.title(f"GA Best Solution; fitness={solution_fitness:.4f}", color='red', fontsize=10)
plt.imshow(result, origin='lower', interpolation='nearest', cmap="gray_r", vmin=0, vmax=1)
plt.axis("off")
plt.show()



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

gt    = np.zeros((H, W), np.uint8); gt[circle]  = 1
seed1 = np.zeros((H, W), np.uint8); seed1[bottom] = 1    # bottom half
seed2 = np.zeros((H, W), np.uint8); seed2[top]    = 1    # top half

gt_vec, seed1_vec, seed2_vec = gt.ravel(), seed1.ravel(), seed2.ravel()
n_genes = H*W

# ----------------------------
# 2) Fitness: Fβ (β²)  – λ_tv * TV – λ_iso * ISO
# ----------------------------
beta = 8.0                 # ↑ recall (FN >> FP); try 6–12
b2   = beta*beta
pos  = (gt_vec == 1)
neg  = ~pos

lam_tv  = 0.010            # TV penalty weight ~0.006–0.02
lam_iso = 0.020            # isolated-1 penalty weight ~0.01–0.04

def total_variation(flat):
    """Normalized 4-neighborhood boundary length (discourages ragged edges)."""
    img = flat.reshape(H, W)
    edges = np.sum(img[:,1:] != img[:,:-1]) + np.sum(img[1:,:] != img[:-1,:])
    return edges / (H*(W-1) + (H-1)*W)

def isolated_ones_fraction(flat):
    """Fraction of 1s with <2 neighbors in 8-neighborhood (suppresses speckles)."""
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
    fbeta = (1+b2)*TP / ((1+b2)*TP + b2*FN + FP + 1e-9)   # ∈ [0,1]
    tv    = total_variation(sol)
    iso   = isolated_ones_fraction(sol)
    return float(fbeta - lam_tv*tv - lam_iso*iso)

# ----------------------------
# 3) Initial population: ONLY noisy variants of seeds (no union)
# ----------------------------
rng = np.random.default_rng(42)
sol_per_pop = 120
init = [seed1_vec.copy(), seed2_vec.copy()]
while len(init) < sol_per_pop:
    base = seed1_vec if (len(init) % 2 == 0) else seed2_vec
    indiv = base.copy()
    m = max(1, int(0.10 * n_genes))              # flip ~10% bits for diversity
    flip = rng.choice(n_genes, size=m, replace=False)
    indiv[flip] ^= 1
    init.append(indiv)
initial_population = np.stack(init, axis=0).astype(np.uint8, copy=False)

# ----------------------------
# 4) GA configuration (strong pressure + annealed mutation)
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
print(f"Elapsed: {time.time()-t0:.2f}s")

best_vec, best_fit, _ = ga.best_solution()
best_img = best_vec.reshape(H, W)

def show(ax, img, title):
    ax.imshow(img, origin="lower", cmap="gray_r", vmin=0, vmax=1)
    ax.set_title(title); ax.axis("off")

fig, axs = plt.subplots(1, 4, figsize=(12,3))
show(axs[0], seed1, "Seed1 (bottom half)")
show(axs[1], seed2, "Seed2 (top half)")
show(axs[2], gt,    "GT (full circle)")
show(axs[3], best_img, "Best")
plt.tight_layout(); plt.show()

plt.figure(figsize=(6,3))
plt.plot(ga.best_solutions_fitness, marker="o", ms=3)
plt.xlabel("Generation"); plt.ylabel("Fitness (Fβ - λ·TV - λ·ISO)")
plt.grid(alpha=0.3); plt.show()