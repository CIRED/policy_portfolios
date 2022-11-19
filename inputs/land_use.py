# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 16:44:44 2021

@author: charl
"""

import pandas as pd
import numpy as np
import scipy as sc
from scipy import optimize
from scipy.stats import norm

### LAND-USE

def predict_urbanized_area(beta, density):
    urban_footprint = density * beta
    urban_footprint[urban_footprint > 0.95] = 1
    return urban_footprint

def predict_urbanized_area2(beta, density):
    urban_footprint = (density > 1000)
    return urban_footprint

def convert_density_to_urban_footprint(city, path_data_city, list_city):
    
    path_data_city=path_data_city + "Data/"
    country = list_city.Country[list_city.City == city].iloc[0]
    proj = str(int(list_city.GridEPSG[list_city.City == city].iloc[0]))
    
    urban_footprint = pd.read_csv(path_data_city + country + '/' + city + 
                       '/Land_Cover/grid_ESACCI_LandCover_2015_' + 
                       str.upper(city) + '_' + proj +'.csv')
    
    urban_footprint = urban_footprint.ESACCI190 / urban_footprint.AREA
    
    density = pd.read_csv(path_data_city + country + '/' + city +
                      '/Population_Density/grille_GHSL_density_2015_' +
                      str.upper(city) + '.txt', sep = '\s+|,', engine='python')
    
    density_ghsl = (density.loc[:,density.columns.str.startswith("density")].squeeze())
    #density_ghsl[np.isneginf(density_ghsl)] = 0
    #_carto2(density_ghsl, grid)
    #carto2(urban_footprint, grid)
    
    coeff_corr = np.corrcoef(density_ghsl, urban_footprint)[0, 1]

    def log_likelihood_censored (X0, urban_footprint, density_ghsl):
        
        beta = X0[0]
        sigma = X0[1]
        
        obs = (urban_footprint < 0.95)
        target_obs, x_obs = urban_footprint[obs], density_ghsl[obs]
        target_cens, x_cens = urban_footprint[~obs], density_ghsl[~obs]
        
        error_obs = (beta * x_obs) - target_obs
        pdf = np.exp(-(error_obs ** 2)/(2 * (sigma ** 2)))/(sigma * np.sqrt(2 * np.pi))
        ll_obs = np.nansum(np.log(pdf))
    
        ll_cens = norm((beta * x_cens), sigma).logsf(target_cens).sum()

        return ll_obs

    def minus_log_likelihood_censored (X0):
        return (- log_likelihood_censored(X0, urban_footprint, density_ghsl))

    X0 = np.array([1e-5, 0.3])
    calibration = optimize.minimize(minus_log_likelihood_censored, X0, 
                                    method = 'Nelder-Mead')
    beta = calibration.x[0]
    
    observed_urban_footprint = sum(urban_footprint)
    urban_footprint_pred = np.array(density_ghsl) * beta
    urban_footprint_pred = np.where(urban_footprint_pred > 0.95, 1, urban_footprint_pred)
    predicted_urban_footprint = sum(urban_footprint_pred)
    
    urban_footprint_residual = urban_footprint - urban_footprint_pred
    
    #return beta, coeff_corr, observed_urban_footprint, predicted_urban_footprint, urban_footprint_residual
    return beta
