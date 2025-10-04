#!/usr/bin/env python3
# --- keep BLAS to 1 thread per process (must be before numpy import) ---
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import time
import functools
import operator

import numpy as np
import imageio
import matplotlib.pyplot as plt
import pygad

print("pygad version:", pygad.__version__)

# ----------------------------
# Utils: image <-> chromosome
# ----------------------------
def img2chromosome(img_arr):
    # convert an image to a 1D vector
    return np.reshape(img_arr, newshape=(functools.reduce(operator.mul, img_arr.shape)))

# ----------------------------
# Load target image as float32 in [0,1]
# ----------------------------
target_im = imageio.imread('/home/delaram/SpatialPeeler/fruit.jpg')
target_im = np.asarray(target_im, dtype=np.float32) / 255.0
target_chromosome = img2chromosome(target_im)
n_genes = target_chromosome.size
print("target shape:", target_im.shape)
print("chromosome length:", n_genes)

# Cache constant term used in fitness
target_sum = np.sum(target_chromosome, dtype=np.float32)

# ----------------------------
# Fitness: maximize -(L1 error), implemented as target_sum - L1
# ----------------------------
def fitness_fun(ga_instance, solution, solution_idx):
    # solution is a 1D float vector in [0,1]
    diff = np.abs(target_chromosome - solution).sum(dtype=np.float32)
    return target_sum - diff

# ----------------------------
# GA configuration
# ----------------------------
sol_per_pop = 20
# On your Xeon: 12 physical cores -> cap workers to 12 (also never exceed pop size)
workers = min(12, sol_per_pop)
print(f"Using {workers} parallel workers")

ga_instance = pygad.GA(
    num_generations=20000,
    num_parents_mating=10,
    fitness_func=fitness_fun,
    sol_per_pop=sol_per_pop,
    num_genes=n_genes,

    # initialize genes uniformly in [0,1]
    init_range_low=0.0,
    init_range_high=1.0,

    # keep arrays compact & fast
    gene_type=np.float32,

    # selection/crossover/mutation: tutorial defaults (with a small nudge to mutation prob)
    parent_selection_type="sss",
    crossover_type="single_point",
    mutation_type="random",
    mutation_by_replacement=True,
    random_mutation_min_val=0.0,
    random_mutation_max_val=1.0,

    # mutation knobs
    mutation_percent_genes=0.01,   # ~1% genes per mutated offspring
    mutation_probability=0.2,      # mutate 20% of offspring (helps escape plateaus)

    # parallel fitness on CPU cores
    parallel_processing=("process", workers),

    # early stop if no improvement for 500 gens (optional safety)
    stop_criteria=["saturate_500"],
)

# Lightweight progress log so you can bail early if it's converged enough
def on_generation(ga):
    gc = ga.generations_completed
    if gc == 1 or gc % 500 == 0:
        best_fit = ga.best_solution()[1]
        # convert back to mean absolute error for interpretability
        current_best = ga.best_solution()[0]
        mae = float(np.abs(current_best - target_chromosome).mean())
        print(f"Gen {gc:6d} | best fitness={best_fit:.4f} | MAE={mae:.6f}")
ga_instance.on_generation = on_generation

# ----------------------------
# Run & time
# ----------------------------
start_time = time.time()
ga_instance.run()
elapsed = time.time() - start_time
# Elapsed time: 13359.438 seconds  (workers=12)
print(f"Elapsed time: {elapsed:.3f} seconds  (workers={workers})")

### save the GA instance (optional)
PCKL_PATH = "/home/delaram/SpatialPeeler/result_fruit_pygad_instance.pkl"
ga_instance.save(PCKL_PATH)

'''
#####################################################################
### load it back (optional)
# ----- load & patch -----
ga = pygad.load(PCKL_PATH)

# 1) replace problematic attributes with picklable top-level callables or None
ga.fitness_func = fitness_fun
ga.on_generation = None  # disable callback entirely

# 2) avoid process-based pools (no pickling); use threads or single-thread
ga.parallel_processing = ("thread", 1)   # safest; you can try ("thread", 8)

# ----- get best solution -----
# Option A: call best_solution() (now safe because we disabled process workers)
solution, best_fit, best_idx = ga.best_solution()
print(f"Stored best fitness = {best_fit} | best index = {best_idx}")
#####################################################################
'''
# ----------------------------
# Plot & save fitness curve
# ----------------------------
best_fit_curve = getattr(ga_instance, "best_solutions_fitness", [])
plt.figure(figsize=(6,4))
plt.plot(best_fit_curve, marker="o", markersize=3)
plt.xlabel("Generation")
plt.ylabel("Best Fitness")
plt.title("Fitness over Generations")
plt.grid(alpha=0.3)
plt.savefig("/home/delaram/SpatialPeeler/result_fruit_pyGADtut_fitness.png",
            dpi=300, bbox_inches="tight")
plt.show()
plt.close()

# ----------------------------
# Best solution & reconstruction
# ----------------------------
# Prefer the built-in getter; if it ever complains, fall back to population
try:
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
except Exception:
    pop = ga_instance.population
    if pop is None or len(pop) == 0:
        raise RuntimeError("No population stored; cannot recover best solution.")
    # Your fitness: target_sum - L1
    l1 = np.abs(pop - target_chromosome).sum(axis=1, dtype=np.float32)
    fit = target_sum - l1
    solution_idx = int(np.argmax(fit))
    solution_fitness = float(fit[solution_idx])
    solution = pop[solution_idx]

print(f"Fitness value of the best solution = {solution_fitness}")
print(f"Index of the best solution        = {solution_idx}")

# reshape with positional args (avoid 'a=' keyword)
result = np.asarray(solution, dtype=np.float32).reshape(target_im.shape)

plt.imshow(result)
plt.axis("off")
plt.savefig("/home/delaram/SpatialPeeler/result_fruit_pyGADtut.png",
            dpi=300, bbox_inches="tight")
plt.show()
plt.close()


