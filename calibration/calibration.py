# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 16:19:40 2021

@author: charl
"""

import pandas as pd
import numpy as np
import scipy as sc
from scipy import optimize

from model.model import *
from calibration.calibration import *

##### CALIBRATION

def calibration2(city, rent, density, size, trans_price, INTEREST_RATE, selected_cells, HOUSEHOLD_SIZE, coeff_land, agricultural_rent, income):
    
    bounds = ((0.1,0.99),
              (0.001,None), #Ro
              (0.001,0.95), #b
              (0.000001, None) #kappa
              )
    
    X0 = np.array([0.25, #beta
                   np.percentile(rent[selected_cells], 98), #Ro
                   0.64, #b
                   0.25] ) 
    
    def minus_log_likelihood(X0):
        print(X0)
        (simul_rent, 
         simul_dwelling_size, 
         simul_density) = model2(X0, 
                                coeff_land, 
                                trans_price, 
                                income,
                                INTEREST_RATE, 
                                HOUSEHOLD_SIZE, agricultural_rent)

    
        (sum_ll,
         ll_R,
         ll_D,
         ll_Q,
         detail_ll_R,
         detail_ll_D,
         detail_ll_Q) = log_likelihood(simul_rent, simul_dwelling_size, simul_density, rent, density, size, selected_cells)
        return -sum_ll
    
    if city == 'Nottingham':
        result_calibration = optimize.minimize(minus_log_likelihood, X0, bounds=bounds,
               options={'maxiter': 100}) 
    else:
        result_calibration = optimize.minimize(minus_log_likelihood, X0, bounds=bounds) 
    
    return result_calibration

def log_likelihood(simul_rent, simul_dwelling_size, simul_density, rent, density, size, selected_cells):
    """ Compute Log-Likelihood on rents, density and dwelling size based on model oputputs. """
    
    x_R = (np.log(rent[selected_cells])) - (np.log(simul_rent[selected_cells]))
    x_Q = (np.log(size[selected_cells])) - (np.log(simul_dwelling_size[selected_cells]))
    x_D = (np.log(density[selected_cells])) - (np.log(simul_density[selected_cells]))
    
    sigma_r2 = (1/sum(selected_cells)) * np.nansum(x_R ** 2)
    sigma_q2 = (1/sum(selected_cells)) * np.nansum(x_Q ** 2)
    sigma_d2 = (1/sum(selected_cells)) * np.nansum(x_D ** 2)
        
    (ll_R, detail_ll_R) = ll_normal_distribution(x_R, sigma_r2)
    (ll_Q, detail_ll_Q) = ll_normal_distribution(x_Q, sigma_q2)
    (ll_D, detail_ll_D) = ll_normal_distribution(x_D, sigma_d2)
    
    return (ll_R  + ll_Q + ll_D,
            ll_R,
            ll_D,
            ll_Q,
            detail_ll_R,
            detail_ll_D,
            detail_ll_Q)

def ll_normal_distribution(error, sigma2):
    """ normal distribution probability density function """
    
    
    log_pdf = -(error ** 2)/(2 * (sigma2))-1/2*np.log(sigma2)-1/2*np.log(2 * np.pi)
    
    return (np.nansum(log_pdf),log_pdf)


class Residuals:  
    """ Classe définissant les résidus à l'issue de la calibration """
    
    def __init__(self,
                 density_residual=0,
                 rent_residual=0,
                 size_residual=0):
        self.density_residual = density_residual
        self.rent_residual = rent_residual
        self.size_residual = size_residual
        
def compute_residuals(density, simul_density, rent, simul_rent, size, simul_size, option):
    residuals = Residuals(
        density_residual = np.log(density / simul_density),
        rent_residual = np.log(rent / simul_rent),
        size_residual = np.log(size / simul_size))
            
    residuals_for_simulation = Residuals(
        density_residual = np.where(np.isnan(density), 0, residuals.density_residual),
        rent_residual = np.where(np.isnan(rent), 0, residuals.rent_residual),
        size_residual = np.where(np.isnan(size), 0, residuals.size_residual))
    
    residuals_for_simulation.density_residual = np.where(
       density == 0, 
       np.log(1 / simul_density), 
       residuals_for_simulation.density_residual)
    
    residuals_for_simulation.density_residual = np.where(
        (simul_density == 0) & (density > 0), 
        np.log(density), #pb
        residuals_for_simulation.density_residual)
    residuals_for_simulation.density_residual = np.where(
        (density == 0) & (simul_density == 0), 
        0, 
        residuals_for_simulation.density_residual)
    
    #residuals_for_simulation.density_residual = np.where(
    #    density == 0, 0, residuals_for_simulation.density_residual)
    
    #residuals_for_simulation.density_residual = np.where(
    #    simul_density == 0, 0, residuals_for_simulation.density_residual)
    
    residuals_for_simulation.density_residual[np.abs(residuals_for_simulation.density_residual) > 4] = 0
    
    if option["add_residuals"] == False:
        residuals_for_simulation.density_residual = np.zeros(len(density))
        residuals_for_simulation.rent_residual = np.zeros(len(rent))
        residuals_for_simulation.size_residual = np.zeros(len(size))
        
    return residuals_for_simulation