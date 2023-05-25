import numpy as np
from scipy.optimize import minimize


# Problem 2, Question 1 
# Define the profit function
def calculate_profit(l_t, k, n, w):
    return k * l_t**(1-n) - w * l_t

# Define the negative profit function (to be minimized)
def negative_profit(l_t, k, n, w):
    return -calculate_profit(l_t, k, n, w)

# Define the constraint for l_t
def constraint(l_t):
    return l_t

# Define the optimization function
def optimize_profit(k, n, w):
    result = minimize(negative_profit, 1.0, args=(k, n, w), bounds=[(0, None)], constraints={'type': 'ineq', 'fun': constraint})
    return result.x[0]

# Problem 2, Question 2


