import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

# Code for country A, alpha 
params = {
    "model1": {"alpha": 0.4, "theta": 0.9, "beta": 0.90, "k_0": 1.0, "delta": 0.05}, 

}


# Code for country A, alpha 
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

# define the loki
def k_lokus(k, alpha):
    return k**alpha

def c_lokus(alpha, beta):
    return ((beta*alpha)/(1-beta))**(1/(1-alpha))


# Code for country B, alpha 
paramsB1 = {
    "modelB1": {"alphaB1": 0.2, "thetaB1": 0.9, "betaB1": 0.90, "k_0B1": 1.0, "deltaB1": 0.05}, 

}


# Code for country B, alpha 
def def_steady_stateB1(paramsB1):
    alphaB1, thetaB1, betaB1, k_0B1 = paramsB1["alphaB1"], paramsB1["thetaB1"], paramsB1["betaB1"], paramsB1["k_0B1"]
    k_starB1 = ((betaB1*alphaB1)/(1-betaB1))**(1/(1-alphaB1))
    c_starB1 = k_starB1**alphaB1
    return k_starB1, c_starB1

# Calculate steady state
ssB1 = def_steady_stateB1(paramsB1["modelB1"])



def productionfunctionB1(paramsB1, T=100):
    alphaB1, thetaB1, betaB1, k_0B1 = paramsB1["alphaB1"], paramsB1["thetaB1"], paramsB1["betaB1"], paramsB1["k_0B1"]
    kB1 = np.zeros(T+1)
    cB1 = np.zeros(T+1)
    wB1 = np.zeros(T+1)
    rB1 = np.zeros(T+1)

    """initial values"""
    kB1[0]=k_0B1
    cB1[0]=k_0B1**alphaB1
    wB1[0]=(1-alphaB1)*k_0B1**alphaB1
    rB1[0]=(alpha)*k_0B1**(alphaB1-1)

    """transition path"""
    for t in range(T):
        kB1[t+1] = wB1[t] + (1-rB1[t])*kB1[t] - cB1[t]
        cB1[t+1] = cB1[t]*(betaB1*(1+rB1[t+1]))**(1/thetaB1)
        wB1[t+1] = (1-alphaB1)*kB1[t+1]**alphaB1
        rB1[t+1] = alphaB1*kB1[t+1]**(alphaB1-1)
    
    return kB1, cB1

# define the loki
def k_lokusB1(kB1, alphaB1):
    return kB1**alphaB1

def c_lokusB1(alphaB1, betaB1):
    return ((betaB1*alphaB1)/(1-betaB1))**(1/(1-alphaB1))

# Code for country B, beta

paramsB2 = {
    "modelB2": {"alphaB2": 0.4, "thetaB2": 0.5, "betaB2": 0.50, "k_0B2": 1.0, "deltaB2": 0.05},

}

def def_steady_stateB2(paramsB2):
    alphaB2, thetaB2, betaB2, k_0B2 = paramsB2["alphaB2"], paramsB2["thetaB2"], paramsB2["betaB2"], paramsB2["k_0B2"]
    k_starB2 = ((betaB2*alphaB2)/(1-betaB2))**(1/(1-alphaB2))
    c_starB2 = k_starB2**alphaB2
    return k_starB2, c_starB2

# Calculate steady state
ssB2 = def_steady_stateB2(paramsB2["modelB2"])

def productionfunctionB2(paramsB2, T=100):
    alphaB2, thetaB2, betaB2, k_0B2 = paramsB2["alphaB2"], paramsB2["thetaB2"], paramsB2["betaB2"], paramsB2["k_0B2"]
    kB2 = np.zeros(T+1)
    cB2 = np.zeros(T+1)
    wB2 = np.zeros(T+1)
    rB2 = np.zeros(T+1)

    """initial values"""
    kB2[0]=k_0B2
    cB2[0]=k_0B2**alphaB2
    wB2[0]=(1-alphaB2)*k_0B2**alphaB2
    rB2[0]=(alpha)*k_0B2**(alphaB2-1)

    """transition path"""
    for t in range(T):
        kB2[t+1] = wB2[t] + (1-rB2[t])*kB2[t] - cB2[t]
        cB2[t+1] = cB2[t]*(betaB2*(1+rB2[t+1]))**(1/thetaB2)
        wB2[t+1] = (1-alphaB2)*kB2[t+1]**alphaB2
        rB2[t+1] = alphaB2*kB2[t+1]**(alphaB2-1)
    
    return kB2, cB2

# define the loki
def k_lokusB2(kB2, alphaB2):
    return kB2**alphaB2

def c_lokusB2(alphaB2, betaB2):
    return ((betaB2*alphaB2)/(1-betaB2))**(1/(1-alphaB2))



