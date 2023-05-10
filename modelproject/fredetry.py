import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

# Set up the baseline model/parameters
params = {
    "model1": {"alpha": 0.33, "theta": 0.9, "beta": 0.50, "k_0": 1.0, "delta": 0.05},
}

def def_steady_state(params):
    alpha, theta, beta, k_0 = params["alpha"], params["theta"], params["beta"], params["k_0"]
    k_star = ((beta*alpha)/(1-beta))**(1/(1-alpha))
    c_star = k_star**alpha
    return k_star, c_star

# Calculate steady state
ss = def_steady_state(params["model1"])


def productionfunction(params, T=100):
    alpha, theta, beta, k_0 = params["alpha"], params["theta"], params["beta"], params["k_0"]
    k = np.zeros(T+1)
    c = np.zeros(T+1)
    w = np.zeros(T+1)
    r = np.zeros(T+1)

    """initial values"""
    k[0]=k_0
    c[0]=k_0**alpha
    w[0]=(1-alpha)*k_0**alpha
    r[0]=(alpha)*k_0**(alpha-1)

    """transition path"""
    for t in range(T):
        k[t+1] = w[t] + (1-r[t])*k[t] - c[t]
        c[t+1] = c[t]*(beta*(1+r[t+1]))**(1/theta)
        w[t+1] = (1-alpha)*k[t+1]**alpha
        r[t+1] = alpha*k[t+1]**(alpha-1)
    
    return k, c

# Define production function and investment curve
def k_lokus(k, alpha):
    return k**alpha

def c_lokus(alpha, beta):
    return ((beta*alpha)/(1-beta))**(1/(1-alpha))