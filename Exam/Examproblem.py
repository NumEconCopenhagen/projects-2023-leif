import numpy as np
from scipy.optimize import minimize

# We firstly define the Griewank function
def griewank(x):
    n = len(x)
    sum_term = np.sum(x**2 / 4000)
    prod_term = np.prod(np.cos(x / np.sqrt(np.arange(1, n + 1))))
    return sum_term - prod_term + 1

# Refined global optimizer with multi-start
def refined_global_optimizer(bounds, tolerance, warmup_iterations, max_iterations):
    x_star = np.zeros(len(bounds))  # Initialize x_star as a zero vector
    f_star = np.inf

    for k in range(max_iterations):
        # Step 3.A: Draw random x^k uniformly within chosen bounds
        x_k = np.random.uniform(bounds[:, 0], bounds[:, 1])

        if k >= warmup_iterations:
            # Step 3.C: Calculate chi^k
            chi_k = 0.5 * 2 / (1 + np.exp((k - warmup_iterations) / 100))

            # Step 3.D: Calculate x_k0
            x_k0 = chi_k * x_k + (1 - chi_k) * x_star

            # Step 3.E: Run optimizer with x_k0 as initial guess
            res = minimize(griewank, x_k0, method='BFGS', tol=tolerance)
            x_k_star = res.x
            f_k_star = res.fun

            # Step 3.F: Update x_star and f_star
            if k == warmup_iterations or f_k_star < f_star:
                x_star = x_k_star
                f_star = f_k_star

        # Step 3.G: Check if f_star is below tolerance
        if f_star < tolerance:
            break

    return x_star

def generate_effective_initial_guesses(bounds, warmup_iterations, max_iterations, x_star):
    effective_initial_guesses = []

    for k in range(max_iterations):
        if k < warmup_iterations:
            effective_initial_guesses.append(np.random.uniform(bounds[:, 0], bounds[:, 1]))
        else:
            chi_k = 0.5 * 2 / (1 + np.exp((k - warmup_iterations) / 100))
            x_k0 = chi_k * effective_initial_guesses[k-warmup_iterations] + (1 - chi_k) * x_star
            effective_initial_guesses.append(x_k0)

    return np.array(effective_initial_guesses)
