
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
        par.kappa = 1

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

    def calc_utility(self,LM,HM,LF,HF):
        """ calculate utility """

        par = self.par

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

        # Then doing the disutility of work where we have added the parameter kappa
        TM = LM+HM
        TF = LF+HF
        disutility = par.nu*(TM**e_power/e_power+TF**e_power/e_power)+par.kappa*HM
        
        return utility - disutility
    
    #For the continuously function we need the negative value of the utility function. We multiply by 100 to achieve precise estimates
    def u_function(self, L):
        return -self.calc_utility(L[0],L[1],L[2],L[3])*100
    

    def solve(self, do_print=False):
        """ solve model continously """

        par = self.par
        opt = self.opt
        
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

        opt.HF_HM = result.x[3]/result.x[1]
        # This is done to calculate the ratio

        return opt  

    def solve_wF_vec(self):
        # solve model for vector of female wages 

        par = self.par 
        opt = self.opt

        log_HF_HM = np.zeros(par.wF_vec.size)

        for i, w in enumerate(par.wF_vec):
            par.wF = w

            opt = self.solve()

            log_HF_HM[i] = np.log(opt.HF_HM)

        par.wF = 1.0

        return log_HF_HM

    
    def target(self, params):
        par = self.par
        sigma_target, kappa_target = params
        beta0_target = 0.4
        beta1_target = -0.1

        beta0, beta1 = self.run_regression(sigma_target, kappa_target)

        return (beta0_target - beta0)**2 + (beta1_target - beta1)**2

    def run_regression(self, sigma_optimal, kappa_optimal):
        """ run regression """

        par = self.par
        opt = self.opt

        par.sigma = sigma_optimal
        par.kappa = kappa_optimal

        x = np.log(par.wF_vec)
        y = self.solve_wF_vec()
        A = np.vstack([np.ones(x.size),x]).T
        opt.beta0, opt.beta1 = np.linalg.lstsq(A,y,rcond=None)[0]
        
        return opt.beta0, opt.beta1

    def estimate(self, used_seed):
        # estimate kappa and sigma

        # We use a random seed
        np.random.seed(used_seed)

        # We set the initial guess
        init_guess = [np.random.uniform(0.01, 1)]
        
        # We set the bounds
        bounds = [(0.01, 100),(-10,10)]

        # We optimize over the regression
        results = optimize.minimize(self.target, init_guess, method="Nelder-Mead", bounds=bounds)

        # We save the results
        sigma_results = results.x[0]
        kappa_results = results.x[1]

        params_results =sigma_results, kappa_results

        return sigma_results, kappa_results, self.target(params_results)

        
    pass