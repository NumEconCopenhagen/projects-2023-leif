import numpy as np
import mpmath as mp
import sympy
import matplotlib.pyplot as plt
import matplotlib as mpl
from types import SimpleNamespace
from scipy import optimize

from types import SimpleNamespace
import scipy.optimize as optimize

class Ramseymodelclass:
    def __init__(self):
        par = self.par = SimpleNamespace()

        # parameters
        par.alpha = 0.40
        par.beta = 0.90
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
        return k_star, c_star, w_star, rk_star, y_star, pi_star
