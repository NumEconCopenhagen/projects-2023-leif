
from types import SimpleNamespace

import numpy as np
from scipy import optimize

import pandas as pd
import matplotlib.pyplot as plt
import ipywidgets

import math

class HouseholdSpecializationModelClass:

    def __init__(self):
        """ setup model """

        # a. create namespaces
        par = self.par = SimpleNamespace()
        sol = self.sol = SimpleNamespace()

        # b. preferences
        par.rho = 2.0
        par.nu = 0.001
        par.epsilon = 1.0
        par.omega = 0.5 

        # c. household production
        par.alpha = 0.5
        par.sigma = 1.0

        # d. wages
        par.wM = 1.0
        par.wF = 1.0
        par.wF_vec = np.linspace(0.8,1.2,5)

        # e. targets
        par.beta0_target = 0.4
        par.beta1_target = -0.1

        # f. solution
        sol.LM_vec = np.zeros(par.wF_vec.size)
        sol.HM_vec = np.zeros(par.wF_vec.size)
        sol.LF_vec = np.zeros(par.wF_vec.size)
        sol.HF_vec = np.zeros(par.wF_vec.size)

        sol.beta0 = np.nan
        sol.beta1 = np.nan

    def calc_utility(self,LM,HM,LF,HF):
        """ calculate utility """

        par = self.par
        sol = self.sol

        # a. consumption of market goods
        C = par.wM*LM + par.wF*LF

         # b. home production, first defining the power
        if par.sigma == 0:
            s_power = (par.sigma-1)/(par.sigma+1e-8)

        elif par.sigma == 1:
            s_power = 0

        else: 
            s_power = (par.sigma-1)/(par.sigma)

        # Then defining the home production
        if par.sigma == 0:
            H = pd.min(HM,HF)

        elif par.sigma == 1:
            H = HM**(1-par.alpha)*HF**par.alpha
            
        else:
            H = ((1-par.alpha)*HM**(s_power)+par.alpha*HF**(s_power))**(1/s_power)

        # c. total consumption utility, starting with defining the power
        if par.rho == 1:
            r_power = (1-par.rho+1e-8)

        else:
            r_power = (1-par.rho)

        # Then doing the total consumption
        Q = C**par.omega*H**(1-par.omega)
        utility = np.fmax(Q,1e-8)**(r_power)/(r_power)

        # d. disutlity of work, starting with defining the power
        if par.epsilon == 0:
            e_power = 1+1/(par.epsilon+1e-8)
        else: 
            e_power = 1+1/par.epsilon

        # Then doing the disutility of work
        TM = LM+HM
        TF = LF+HF
        disutility = par.nu*(TM**e_power/e_power+TF**e_power/e_power)
        
        return utility - disutility

    def solve_discrete(self,do_print=False):
        """ solve model discretely """
        
        par = self.par
        sol = self.sol
        opt = SimpleNamespace()
        
        # a. all possible choices
        x = np.linspace(0,24,49)
        LM,HM,LF,HF = np.meshgrid(x,x,x,x) # all combinations
    
        LM = LM.ravel() # vector
        HM = HM.ravel()
        LF = LF.ravel()
        HF = HF.ravel()

        # b. calculate utility
        u = self.calc_utility(LM,HM,LF,HF)
    
        # c. set to minus infinity if constraint is broken
        I = (LM+HM > 24) | (LF+HF > 24) # | is "or"
        u[I] = -np.inf
    
        # d. find maximizing argument
        j = np.argmax(u)
        
        opt.LM = LM[j]
        opt.HM = HM[j]
        opt.LF = LF[j]
        opt.HF = HF[j]

        # e. print
        if do_print:
            for k,v in opt.__dict__.items():
                print(f'{k} = {v:6.4f}')

        return opt

    def solve(self, do_print=False):
        """ solve model continously """

        par = self.par
        sol = self.sol
        opt = SimpleNamespace()
        
        # We provide an initial guess
        LM_guess, LF_guess, HM_guess, HF_guess = 6,6,6,6
        x_guess = np.array([LM_guess, LF_guess, HM_guess, HF_guess])

        # We define the objective function
        obj = lambda x: -self.calc_utility(x[0], x[1], x[2], x[3])

        # We set the bounds
        bounds = ((1e-8, 24-1e-8),(1e-8, 24-1e-8),(1e-8, 24-1e-8),(1e-8, 24-1e-8))

        # We optimize using the Nelder-Mead method
        result = optimize.minimize(obj, x_guess, method="Nelder-Mead", bounds=bounds, tol=1e-8)

        opt.LM = result.x[0]
        opt.HM = result.x[1]
        opt.LF = result.x[2]
        opt.HF = result.x[3]

         # e. print
        if do_print:
            for k,v in opt.__dict__.items():
                print(f'{k} = {v:6.4f}')

        return opt  

    def solve_wF_vec(self,discrete=False, do_plot=False, do_print=False):
        """ solve model for vector of female wages """

        par = self.par 
        sol = self.sol

        for i, w in enumerate(self.par.wF_vec):
            par.wF = w

            if discrete:
                opt = self.solve_discrete()

            else:
                opt = self.solve()

            sol.LF_vec[i] = opt.LF
            sol.HF_vec[i] = opt.HF
            sol.LM_vec[i] = opt.LM
            sol.HM_vec[i] = opt.HM

        par.wF = 1.0

    
    def run_regression(self):
        """ run regression """

        par = self.par
        sol = self.sol

        x = np.log(par.wF_vec)
        y = np.log(sol.HF_vec/sol.HM_vec)
        A = np.vstack([np.ones(x.size),x]).T
        sol.beta0,sol.beta1 = np.linalg.lstsq(A,y,rcond=None)[0]
        

    def estimate(self, do_print=False):
        """ estimate alpha and sigma """

        sol = self.sol
        par = self.par

        def target(x):
            par.alpha = x[0]
            par.sigma = x[1]

            self.solve_wF_vec()
            self.run_regression()
            sol.residual = (sol.beta0-par.beta0_target)**2 + (sol.beta1-par.beta1_target)**2

            return sol.residual
            
        x0 = [0.8, 0.3]

        # Bounds
        bounds = ((0.1, 0.99), (0.05,0.99))

        solution = optimize.minimize(target, x0, method="Nelder-Mead", bounds=bounds, tol=1e-9)

        par.alpha = solution.x[0]
        par.sigma = solution.x[1]

        if do_print:
                print(f"\u03B1_opt = {par.alpha:6.10f}")
                print(f"\u03C3_opt = {par.sigma:6.10f}")
                print(f"Residual_opt = {sol.residual:6.6f}")

        
    pass