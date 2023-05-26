from typing import Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from types import SimpleNamespace
import math
from typing import Any

class Q1_1:
    def __init__(self):
        self.params = {"alpha": 0.5, "tau": 0.3, "w": 1.0, "kappa": 1.0, "nu": 1 / (2*16**2)}
        self.G_values = [1.0, 2.0]

    def utility_function(self, L, G):
        alpha, tau, w, kappa, nu = self.params["alpha"], self.params["tau"], self.params["w"], self.params["kappa"], self.params["nu"]
        C = kappa + (1 - tau) * w * L
        return np.log(C**alpha * G**(1 - alpha)) - nu * (L**2 / 2)

    def optimal_labor_choice(self, G):
        # Define the objective function to maximize (it is negative due to python only can minimize)
        objective = lambda L: -self.utility_function(L, G)

        # Set the bounds for labor supply (L) between 0 and 24
        possible_hours = (0, 24)

        # Provide an initial guess for L 
        initial_guess = 10

        # Use numerical optimization to find the maximum of the utility function
        result = minimize(objective, x0=initial_guess, bounds=[possible_hours])

        return result.x
    
