from typing import Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import math


def optimal_labor_supply(wage, tax, alpha, kappa, nu, G):
    tau = tax
    tilde_w = (1 - tau) * wage
    discriminant = kappa**2 + 4 * alpha / nu * tilde_w**2
    L_star = (-kappa + math.sqrt(discriminant)) / (2 * tilde_w)
    return L_star
# Baseline parameter values
alpha = 0.5
kappa = 1.0
nu = 1 / (2 * 16**2)
wage = 1.0
tax = 0.30



