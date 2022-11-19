# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 18:54:41 2021

@author: charl
"""

############# CODE SIMPLE #####

import pandas as pd
import numpy as np
import copy
import pickle
import scipy as sc
from scipy import optimize
import matplotlib.pyplot as plt
import os
i = 0
WALKING_SPEED = 5
BETA = 0.3
B = 0.7
KAPPA = 0.01
INTEREST_RATE = 0.05
HOUSEHOLD_SIZE = 1


def model(init, coeff_land, trans_price, income, INTEREST_RATE, HOUSEHOLD_SIZE, housing_supply = None, OPTION = None):
    """ Compute rents, densities and dwelling sizes """    

    ## Starting point
    beta = init[0]
    Ro = init[1]
    b = init[2]
    kappa = init[3]
    
    a=1-b
    
    income_net_of_transport_costs = np.fmax(income - trans_price, np.zeros(len(trans_price)))             
    rent = (Ro * income_net_of_transport_costs**(1/beta) /income**(1/beta))
    np.seterr(divide = 'ignore', invalid = 'ignore')
    dwelling_size = beta * income_net_of_transport_costs / rent
    dwelling_size[rent == 0] = 0
    np.seterr(divide = 'warn', invalid = 'warn')
    housing = coeff_land * ((kappa**(1/a)) * (((b / INTEREST_RATE) * rent) ** (b/(a))))
    density = copy.deepcopy(housing / dwelling_size)
    density[dwelling_size == 0] = 0
    density[np.isnan(density)] = 0    
    housing[np.isinf(housing)] = 10000000
    return rent, dwelling_size, density

i = 0
rows = []

for Y in [1000, 2000, 10000]:
    for N in [300000, 2000000, 10000000]:
        for speed_c in [15, 45, 90]:
            for speed_p in [15, 45, 90]:
                for cost_c in [0, 5, 10]:
                    for cost_p in [0, 5, 10]:
                        for Ra in [0, 2, 10]:
                            print(Y, N, speed_c, speed_p, cost_c, cost_p)
                            i += 1
                            
                            #create grid
                            grid = GridSimulation()                         
                            grid.create_grid(100)
                            
                            #create coeff land
                            coeff_land = np.ones(len(grid.distance_centre))
                            
                            #create transport price and modal shares
                            transport_price_c = ((grid.distance_centre / speed_c) * (Y / (365 * 8))) + cost_c * grid.distance_centre 
                            transport_price_p = ((grid.distance_centre / speed_p) * (Y / (365 * 8))) + cost_p * grid.distance_centre 
                            transport_price_w = ((grid.distance_centre / WALKING_SPEED) * (Y / (365 * 8)))
                            transport_price = 365 * 2 * np.amin(np.vstack((transport_price_c, transport_price_p, transport_price_w)), axis=0)
                            transport_mode = np.argmin(np.vstack((transport_price_c, transport_price_p, transport_price_w)), axis=0)
            
                            #equilibrium
                            def compute_residual(R_0):
                                rent, dwelling_size, density = model(np.array([BETA, R_0, B, KAPPA]), coeff_land, transport_price, Y, INTEREST_RATE, HOUSEHOLD_SIZE)
                                return N - sum(density)    
                            R_0 = sc.optimize.fsolve(compute_residual, 200)
                            rent, dwelling_size, density = model(np.array([BETA, R_0, B, KAPPA]), coeff_land, transport_price, Y, INTEREST_RATE, HOUSEHOLD_SIZE)
                            
                            utility = (Y  * ((1 - BETA) ** (1 - BETA)) * (BETA ** BETA)) / (R_0 ** BETA)
                            utility2 = np.nanmedian(((Y - transport_price - (rent * dwelling_size)) ** (1 - BETA)) * ((dwelling_size) ** (BETA)))
                            vkt = np.nansum(density[transport_mode == 0] * grid.distance_centre[transport_mode == 0] * 365 * 2)
                            pkt = np.nansum(density[transport_mode == 1] * grid.distance_centre[transport_mode == 1] * 365 * 2)
                            wkt = np.nansum(density[transport_mode == 2] * grid.distance_centre[transport_mode == 2] * 365 * 2)
                            
                            rows.append([i, Y, N, speed_c, speed_p, cost_c, cost_p, Ra, utility, utility2, vkt, pkt, wkt])