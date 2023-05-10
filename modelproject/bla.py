import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

# Code for country A, alpha 
params = {
    "model1": {"alpha": 0.4, "theta": 0.9, "beta": 0.50, "k_0": 1.0, "delta": 0.05}, 

}

params = {
    "model1A": {"alpha": 0.4, "theta": 0.9, "beta": 0.50, "k_0": 1.0, "delta": 0.05}, #capital plays a larger role in generating output
    "model1B": {"alpha": 0.2, "theta": 0.9, "beta": 0.50, "k_0": 1.0, "delta": 0.05}, #capital plays a smaller role in generating output
    "model2A": {"alpha": 0.33, "theta": 0.9, "beta": 0.20, "k_0": 1.0, "delta": 0.05}, #high level of impatience - values utility today more than country B
    "model2B": {"alpha": 0.33, "theta": 0.95, "beta": 0.95, "k_0": 1.0, "delta": 0.05}, #moderate level of impatience - values consumption tomorrow more than country A

}

# Code for country A, alpha 
def def_steady_state(params):
    alpha, theta, beta, k_0 = params["alpha"], params["theta"], params["beta"], params["k_0"]
    k_star = ((beta*alpha)/(1-beta))**(1/(1-alpha))
    c_star = k_star**alpha
    return k_star, c_star

# Calculate steady state
ss = def_steady_state(params["model1"])
A1_ss = def_steady_state(params["model1A"])
B1_ss = def_steady_state(params["model1B"])
A2_ss = def_steady_state(params["model2A"])
B2_ss = def_steady_state(params["model2B"])


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

#simulating the model for each country and variables 
A1_k, A1_c = productionfunction(params["model1A"])
B1_k, B1_c = productionfunction(params["model1B"])
A2_k, A2_c = productionfunction(params["model2A"])
B2_k, B2_c = productionfunction(params["model2B"])


# define the loki for each country
def A1_k_lokus(A1_k, params):
    alpha = params["model1A"]["alpha"]
    return A1_k**alpha

def A1_c_lokus(params):
    alpha = params["model1A"]["alpha"]
    beta = params["model1A"]["beta"]
    return ((beta*alpha)/(1-beta))**(1/(1-alpha))

def B1_k_lokus(B1_k, params):
    alpha = params["model1B"]["alpha"]
    return B1_k**alpha

def B1_c_lokus(params):
    alpha = params["model1B"]["alpha"]
    beta = params["model1B"]["beta"]
    return ((beta*alpha)/(1-beta))**(1/(1-alpha))

def A2_k_lokus(A2_k, params):
    alpha = params["model2A"]["alpha"]
    return A2_k**alpha

def A2_c_lokus(params):
    alpha = params["model2A"]["alpha"]
    beta = params["model2A"]["beta"]
    return ((beta*alpha)/(1-beta))**(1/(1-alpha))

def B2_k_lokus(B2_k, params):
    alpha = params["model2B"]["alpha"]
    return B2_k**alpha

def B2_c_lokus(params):
    alpha = params["model2B"]["alpha"]
    beta = params["model2B"]["beta"]
    return ((beta*alpha)/(1-beta))**(1/(1-alpha))



from types import SimpleNamespace
import scipy.optimize as optimize

class Ramseymodelclass:
    def __init__(self):
        par = self.par = SimpleNamespace()

        # parameters
        par.alpha = 0.33
        par.beta = 0.50
        par.delta = 0.05
        par.theta = 0.9

    @staticmethod
    def ss_equations(x, par):
        k, c, w, rk = x
        k_star = ((par.beta*par.alpha)/(1-par.beta))**(1/(1-par.alpha))
        c_star = k_star**par.alpha
        w_star = (1-par.alpha)*k_star**par.alpha
        rk_star = par.alpha*k_star**(par.alpha-1)
        equation1 = k_star - k
        equation2 = c_star - c
        equation3 = w_star - w
        equation4 = rk_star - rk
        return [equation1, equation2, equation3, equation4]
    
    def ss_values(self):
        par = self.par
        x0 = [1, 1, 1, 1] # initial guess for the steady state values needed for the algorithm to work
        ss_sol = optimize.root(Ramseymodelclass.ss_equations, x0, args=(par,), method='hybr')
        k_star, c_star, w_star, rk_star = ss_sol.x
        y_star = k_star**par.alpha
        pi_star = y_star - w_star - rk_star*k_star
        return k_star, c_star, w_star, rk_star, y_star, pi_star



import numpy as np
import matplotlib.pyplot as plt

class Ramseymodelclass:
    def __init__(self):
        self.par = {
            "model1": {"alpha": 0.4, "beta": 0.9},
            "modelB1": {"alphaB1": 0.5, "betaB1": 0.8},
            "modelB2": {"alphaB2": 0.3, "betaB2": 0.7}
        }
        self.ss = (1, 1)
        self.ssB1 = (0.5, 0.5)
        self.ssB2 = (0.2, 0.1)

    @staticmethod
    def k_lokus(k, alpha):
        return k ** alpha

    def c_lokus(self, alpha, beta):
        return ((beta * alpha) / (1 - beta)) ** (1 / (1 - alpha))

    def c_lokusB1(self, alphaB1, betaB1):
        return ((betaB1 * alphaB1) / (1 - betaB1)) ** (1 / (1 - alphaB1))

    def c_lokusB2(self, alphaB2, betaB2):
        return ((betaB2 * alphaB2) / (1 - betaB2)) ** (1 / (1 - alphaB2))

ramsey_model = Ramseymodelclass()

# Set up the figure and axes
fig, ax = plt.subplots()

# Plot for model 1
k = np.linspace(0, 10, 100)
c = ramsey_model.c_lokus(ramsey_model.par["model1"]["alpha"], ramsey_model.par["model1"]["beta"])
ax.plot(k, ramsey_model.k_lokus(k, ramsey_model.par["model1"]["alpha"]), color="purple", label="k lokus")
ax.plot([c, c], [ramsey_model.k_lokus(0, ramsey_model.par["model1"]["alpha"]), ramsey_model.k_lokus(c, ramsey_model.par["model1"]["alpha"])], "--", color="blue", label="c lokus")


# Plot for model B1
kB1 = np.linspace(0, 10, 100)
cB1 = ramsey_model.c_lokusB1(ramsey_model.par["modelB1"]["alphaB1"], ramsey_model.par["modelB1"]["betaB1"])
ax.plot(kB1, ramsey_model.k_lokus(kB1, ramsey_model.par["modelB1"]["alphaB1"]), color="red", label="k lokus B1")
ax.plot([cB1, cB1], [ramsey_model.k_lokus(0, ramsey_model.par["modelB1"]["alphaB1"]), ramsey_model.k_lokus(cB1, ramsey_model.par["modelB1"]["alphaB1"])], "--", color="orange", label="c lokus B1")


# Plot for model B2
kB2 = np.linspace(0, 10, 100)
cB2 = ramsey_model.c_lokusB2(ramsey_model.par["modelB2"]["alphaB2"], ramsey_model.par["modelB2"]["betaB2"])
ax.plot(kB2, ramsey_model.k_lokus(kB2, ramsey_model.par["modelB2"]["alphaB2"]), color="green", label="k lokus B2")
ax.plot([cB2, cB2], [ramsey_model.k_lokus(0, ramsey_model.par["modelB2"]["alphaB2"]), ramsey_model.k_lokus(cB2, ramsey_model.par["modelB2"]["alphaB2"])], "--", color="yellow", label="c lokus B2")

# Set axis labels and title
ax.set_xlabel("k")
ax.set_ylabel("c")
ax.set_title("Phase diagram")

# Set axis limits and tick positions
ax.set_xlim([0, 10])
ax.set_ylim([0, 3])
ax.set_xticks(np.arange(0, 11, 1))
ax.set_yticks(np.arange(0, 3.5, 0.5))

# Add legend
ax.legend()

# Add text labels
ax.text(1.0, 0.85, "k lokus A")
ax.text(8.6, 0.95, "c lokus A")
ax.text(0.5, 0.78, "k lokus B1")
ax.text(2.15, 0.2, "c lokus B1")
ax.text(0.054, 0.28, "k lokus B2")
ax.text(0.22, 0.2, "c lokus B2")

# Display the plot
plt.show()
