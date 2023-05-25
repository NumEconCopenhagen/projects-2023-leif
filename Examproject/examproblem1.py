import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

# Defining parameters
params = {
    "problem1": {"alpha": 0.5, "kappa": 1, "ny": 1/(2*16**2), "w": 1.0, "theta": 0.30, "L": {0:24}}, 

}

def def_labour(params):
    alpha, kappa, ny, wage, theta = params["alpha"], params["kappa"], params["ny"], params["w"], params["theta"], params["L"]
    V = np.log(C**alpha*G**{1-alpha})-ny*1/2*L**2
    C = kappa + (1-theta)*wage*L

    return V, C

