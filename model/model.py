# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 16:38:16 2021

@author: charl
"""

import numpy as np
import pandas as pd
import copy

#### MODEL

def model2(init, coeff_land, trans_price, income, INTEREST_RATE, HOUSEHOLD_SIZE, agricultural_rent, housing_supply = None, OPTION = None):
    """ Compute rents, densities and dwelling sizes """    

    ## Starting point
    beta = init[0]
    Ro = init[1]
    b = init[2]
    kappa = init[3]
    
    a=1-b
    
    income_net_of_transport_costs = np.fmax(income - trans_price, np.zeros(len(trans_price)))             
    rent = (Ro * (income_net_of_transport_costs**(1/beta)) /(income**(1/beta)))
    dwelling_size = beta * income_net_of_transport_costs / rent
    dwelling_size[rent == 0] = 0
    if OPTION == 1:
        housing = copy.deepcopy(housing_supply)
        housing[rent < agricultural_rent] = 0
    elif OPTION == 2:
        housing = coeff_land * ((kappa**(1/a)) * (((b / INTEREST_RATE) * rent) ** (b/(a))))
        housing[housing < 0.99*housing_supply] = 0.99*housing_supply[housing < 0.99*housing_supply]
        housing[rent < agricultural_rent] = 0
    else:
        housing = coeff_land * ((kappa**(1/a)) * (((b / INTEREST_RATE) * rent) ** (b/(a))))
        housing[rent < agricultural_rent] = 0
    #np.seterr(divide = 'ignore', invalid = 'ignore')
    rent[rent < agricultural_rent] = agricultural_rent
    density = copy.deepcopy(housing / dwelling_size)
    density[dwelling_size == 0] = 0
    #np.seterr(divide = 'warn', invalid = 'warn')        
    density[np.isnan(density)] = 0    
    density[density == 0] = 1
    density[rent < agricultural_rent] = 1
    #dwelling_size[dwelling_size > 1000] = np.nan
    dwelling_size[dwelling_size == 0] = 1
    rent[rent == 0] = 1
    housing[np.isinf(housing)] = 0
    return rent, dwelling_size, density

def compute_housing_supply(housing_supply_t1_without_inertia, housing_supply_t0, TIME_LAG, DEPRECIATION_TIME):
        diff_housing = ((housing_supply_t1_without_inertia - housing_supply_t0) / TIME_LAG) - (housing_supply_t0 / DEPRECIATION_TIME)
        for i in range(0, len(housing_supply_t1_without_inertia)):
            if housing_supply_t1_without_inertia[i] <= housing_supply_t0[i]:
                diff_housing[i] = - (housing_supply_t0[i] / DEPRECIATION_TIME)

        housing_supply_t1 = housing_supply_t0 + diff_housing
        return housing_supply_t1
    
