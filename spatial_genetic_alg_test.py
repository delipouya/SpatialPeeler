import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng(42)
import numpy as np
import matplotlib.pyplot as plt
import pygad

# ----------------------------
# 1) Binary grids (0/1)
# ----------------------------
H, W = 200, 300
y, x = np.ogrid[:H, :W]
cy, cx, r = H/2, W/2, min(H, W)*0.25

circle = (x - cx)**2 + (y - cy)**2 <= r**2
top    = circle & (y < cy)
bottom = circle & (y >= cy)

# seed1: bottom half = 1, else 0
seed1 = np.zeros((H, W), dtype=int)
seed1[bottom] = 1

# seed2: top half = 1, else 0
seed2 = np.zeros((H, W), dtype=int)
seed2[top] = 1

# ground truth: full circle = 1, else 0
gt = np.zeros((H, W), dtype=int)
gt[circle] = 1

# Visualize seeds & GT (optional)
for title, img in [("SEED1 (bottom half)", seed1),
                   ("SEED2 (top half)",    seed2),
                   ("Ground Truth (full)", gt)]:
    plt.figure(figsize=(6, 4))
    plt.imshow(img, origin='lower', interpolation='nearest', cmap="gray_r", vmin=0, vmax=1)
    plt.title(title); plt.axis("off"); plt.show()


''' creating gaussian circles instead of binary ones
# Gaussian parameters
mu_top, sigma_top       = 2.0, 0.5   # top half
mu_bottom, sigma_bottom = -1.0, 0.5  # bottom half
mu_out, sigma_out       = 0.0, 0.5   # outside circle

# SEED1 = Sample field - a half-circle on the bottom field
seed1 = np.empty((H, W), dtype=float)
seed1[~circle_mask] = rng.normal(mu_out, sigma_out, size=(~circle_mask).sum()) ## Outside circle
seed1[top_half]    = rng.normal(mu_out, sigma_out, size=top_half.sum()) ## Top half same as outside
seed1[bottom_half] = rng.normal(mu_bottom, sigma_bottom, size=bottom_half.sum()) ## Bottom half 

# SEED2 = Sample field - a half-circle on the top field
seed2 = np.empty((H, W), dtype=float)
seed2[~circle_mask] = rng.normal(mu_out, sigma_out, size=(~circle_mask).sum()) ## Outside circle
seed2[bottom_half] = rng.normal(mu_out, sigma_out, size=bottom_half.sum()) ## Bottom half same as outside
seed2[top_half] = rng.normal(mu_top, sigma_top, size=top_half.sum()) ## Top half

# GROUND TRUTH = Sample field - a full circle in the middle of field
ground_truth = np.empty((H, W), dtype=float)
ground_truth[~circle_mask] = rng.normal(mu_out, sigma_out, size=(~circle_mask).sum()) ## Outside circle
ground_truth[top_half]    = rng.normal(mu_top, sigma_top, size=top_half.sum()) ## Top half
ground_truth[bottom_half] = rng.normal(mu_bottom, sigma_bottom, size=bottom_half.sum()) ## Bottom half

# Visualize
plt.figure(figsize=(6, 4))
plt.imshow(ground_truth, origin='lower', interpolation='nearest', cmap="coolwarm")
plt.title("Full circle in the middle of field")
plt.clim(-3, 3) 
plt.colorbar(label='Value')
plt.axis('off')
plt.show()
'''


# Flatten to genome vectors
seed1_vec = seed1.ravel()
seed2_vec = seed2.ravel()
gt_vec    = gt.ravel()
n_genes   = gt_vec.size  # 60000

# ----------------------------
# 2) Fitness: 1/(1 + Hamming)
# ----------------------------
def fitness_func(ga_instance, solution, solution_idx):
    diff = np.count_nonzero(solution != gt_vec)
    return 1.0 / (1.0 + diff)   # maximize (1.0 when identical)

# ----------------------------
# 3) GA configuration (defaults for operators; better population)
# ----------------------------
# Start from your two seeds, then add more individuals to allow offspring.
# We duplicate seeds and add tiny random flips so population isn't degenerate.
rng = np.random.default_rng(123)
base_pop = [seed1_vec.copy(), seed2_vec.copy()]
extra = 8#38  # total population = 40
for k in range(extra):
    # alternate between seed1/seed2 and flip ~0.2% bits to diversify
    base = seed1_vec if (k % 2 == 0) else seed2_vec
    indiv = base.copy()
    #m = max(1, int(0.002 * n_genes))  # ~120 flips on 60k genes
    m = max(1, int(0.3 * n_genes))  # ~120 flips on 60k genes
    flip_idx = rng.choice(n_genes, size=m, replace=False)
    indiv[flip_idx] = 1 - indiv[flip_idx]
    base_pop.append(indiv)
initial_population = base_pop

for indiv in initial_population:
    ### visualize the chormosome 
    plt.figure(figsize=(6, 4))
    plt.imshow(indiv.reshape(H, W), origin='lower', interpolation='nearest', cmap="gray_r", vmin=0, vmax=1)
    plt.title(f"Initial Population Chromosome {indiv}")
    plt.axis("off")
    plt.show()

other_crossovers = ['single_point', 'two_points', 'uniform', 'scattered']
def_crossover = other_crossovers[2]  # 'uniform' is a good default
print(f"Using crossover: {def_crossover}")

best_fitnesses = []
hemming_distances = []
generation_vector = [10, 20, 100, 200] #, 300, 400, 500, 1000, 1500, 2000

for num_gen in generation_vector:
    print(f"\n--- Running GA for {num_gen} generations ---")
    ga = pygad.GA(
        fitness_func=fitness_func,
        num_generations=num_gen,         # <-- respect loop variable
        num_parents_mating=round(len(initial_population)/2),           # half of pop become parents
        sol_per_pop=len(initial_population),  # 40
        num_genes=n_genes,
        gene_space=[0, 1],
        gene_type=int,
        initial_population=initial_population,
        parent_selection_type="sss",
        keep_parents=2,                  # keep elites, but leave room for offspring
        crossover_type=def_crossover,    # default operator (no custom)
        crossover_probability=0.9,       # encourage mixing
        mutation_type="random",          # default mutation
        mutation_percent_genes=1,        # ~1% bits per offspring
        mutation_probability=0.2,        # mutate 20% of offspring
        stop_criteria=["reach_1.0", "saturate_100"],
    )

    ga.run()
    best, best_fit, _ = ga.best_solution()
    hamming = np.count_nonzero(best != gt_vec)

    best_fitnesses.append(best_fit)
    hemming_distances.append(hamming)

    print(f"Best fitness: {best_fit:.6f}")
    print(f"Hamming distance to ground truth: {hamming} / {n_genes}")

    best_grid = best.reshape(H, W)
    plt.figure(figsize=(6, 4))
    plt.imshow(best_grid, origin='lower', interpolation='nearest', cmap="gray_r", vmin=0, vmax=1)
    plt.title(f"GA Best Solution for {num_gen} generations; crossover: {def_crossover}", color='red', fontsize=10)
    plt.suptitle(f"Hamming to GT: {hamming} / {n_genes}; best fitness: {best_fit:.6f}",
                 fontsize=10, y=0.89, color='blue')
    plt.axis("off")
    plt.show()

# Plot fitness vs generations
plt.figure(figsize=(10, 4))
plt.plot(generation_vector, best_fitnesses, 'o-')
plt.xlabel("Number of generations")
plt.ylabel("Best fitness")
plt.title("Best fitness vs generations")
plt.grid()
plt.show()

# Plot Hamming distance vs generations
plt.figure(figsize=(10, 4))
plt.plot(generation_vector, hemming_distances, 'o-')
plt.xlabel("Number of generations")
plt.ylabel("Hamming distance to ground truth")
plt.title("Hamming distance vs generations")
plt.grid()
plt.ylim(0, max(hemming_distances)*1.05)
plt.show()