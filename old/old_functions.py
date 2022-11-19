# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 12:32:19 2021

@author: charl
"""

import re 
import pandas as pd 
import numpy as np
from scipy.stats import norm
from scipy import optimize
import matplotlib.pyplot as plt
import copy




    

def import_profile(distance_cbd, coeff_land, grille, method):
    profil = np.zeros(len(range(int(max(distance_cbd)))))
    sum_cells = np.zeros(len(range(int(max(distance_cbd)))))
    total_cells = np.zeros(len(range(int(max(distance_cbd)))))

    for i in range(int(max(distance_cbd))):
        profil[i] = np.mean(coeff_land[(distance_cbd > i) & (distance_cbd < i + 1)])
        sum_cells[i] = np.sum(coeff_land[(distance_cbd > i) & (distance_cbd < i + 1)] * (grille.AREA / 1000000)[(distance_cbd > i) & (distance_cbd < i + 1)])
        total_cells[i] = (np.pi * ((i + 1) ** 2)) - (np.pi * ((i) ** 2))
    
    if method == 'grid_only':
        return profil

    if method == 'out_of_grid':
        return sum_cells / total_cells
    

def calibration(city, rent, density, size, trans_price, INTEREST_RATE, selected_cells, income, HOUSEHOLD_SIZE, coeff_land, agricultural_rent):
    
    bounds = ((0.1,0.99), #beta
              (0.001,None), #Ro
              (0.0,0.95), #b
              (0, 1) #kappa
              )
    
    
        
    if ((city == "Nottingham") | (city == "Oslo") | (city == 'Tampere') | (city == 'Patras')):
        bounds = ((0.1,0.99), #beta
              (0.001,None), #Ro
              (0.0,0.95), #b
              (0.000001, 1) #kappa
              )
        
    X0 = np.array([0.25, #beta
                   np.percentile(rent[selected_cells], 98), #Ro
                   0.64, #b
                   0.5] )  
    
    def minus_log_likelihood(X0):
    
        (simul_rent, 
         simul_dwelling_size, 
         simul_density) = model(X0, 
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
        print(ll_R)
        print(ll_D)
        print(ll_Q)
        return -sum_ll
    
    if city == 'Nottingham':
        result_calibration = optimize.minimize(minus_log_likelihood, X0, bounds=bounds,
               options={'maxiter': 100}) 
    else:
        result_calibration = optimize.minimize(minus_log_likelihood, X0, bounds=bounds) 
    
    return result_calibration



def compute_r2(density, rent, size, simul_density, simul_rent, simul_size, selected_cells):
    """ Explained variance / Total variance """
    
    sst_rent = ((rent[selected_cells] - 
                 np.nanmean(rent[selected_cells])) ** 2).sum()
    
    sst_density = (((density[selected_cells] - 
                 np.nanmean(density[selected_cells])) ** 2).sum())
    
    sst_size = ((size[selected_cells] - 
                 np.nanmean(size[selected_cells])) ** 2).sum()
    
    sse_rent = ((rent[selected_cells] - 
                 simul_rent[selected_cells]) ** 2).sum()
    
    sse_density = (((density[selected_cells] - 
                 simul_density[selected_cells]) ** 2).sum())
    
    sse_size = ((size[selected_cells] - 
                 simul_size[selected_cells]) ** 2).sum()
    
    r2_rent = 1 - (sse_rent / sst_rent)
    r2_density = 1 - (sse_density / sst_density)
    r2_size = 1 - (sse_size / sst_size)
    
    return r2_rent, r2_density, r2_size

def import_modal_shares_wiki(path_data):
    modal_shares = pd.read_excel(path_data + "modal_shares.xlsx", sheet_name = "Feuille3", header = 0) 
    for i in range(len(modal_shares)):
        modal_shares.iloc[i, 0] = re.sub(r'.*?\xa0', '', modal_shares.iloc[i, 0])
    modal_shares.cycling[modal_shares.city == "Brussels"] = 0.025
    modal_shares.cycling[modal_shares.city == "Hong Kong"] = 0.005
    modal_shares.cycling[modal_shares.city == "Madrid"] = 0.005
    modal_shares.cycling[modal_shares.city == "Prague"] = 0.004
    modal_shares.cycling[modal_shares.city == "Jakarta"] = 0.002
    modal_shares["private car"][modal_shares.city == "Jakarta"] = 0.78
    modal_shares.cycling[modal_shares.city == "Calgary"] = 0.015
    modal_shares.walking[modal_shares.city == "Calgary"] = 0.047
    modal_shares["private car"][modal_shares.city == "Calgary"] = 0.794
    modal_shares["public transport"][modal_shares.city == "Calgary"] = 0.144
    modal_shares.cycling[modal_shares.city == "Edmonton"] = 0.01
    modal_shares.walking[modal_shares.city == "Edmonton"] = 0.037
    modal_shares["private car"][modal_shares.city == "Edmonton"] = 0.84
    modal_shares["public transport"][modal_shares.city == "Edmonton"] = 0.113
    modal_shares["transit_share"] = (modal_shares["walking"] + modal_shares["public transport"]) / (modal_shares["walking"] + modal_shares["public transport"] + modal_shares["private car"])
    
    modal_shares["city"][modal_shares.city == "Hong Kong"] = 'Hong_Kong'
    modal_shares["city"][modal_shares.city == "Los Angeles"] = 'Los_Angeles'
    modal_shares["city"][modal_shares.city == "New York City"] = 'New_York'
    modal_shares["city"][modal_shares.city == "Rio de Janeiro"] = 'Rio_de_Janeiro'
    modal_shares["city"][modal_shares.city == "San Diego"] = 'San_Diego'
    modal_shares["city"][modal_shares.city == "San Francisco"] = 'San_Fransisco'
    modal_shares["city"][modal_shares.city == "Washington, D.C."] = 'Washington_DC'
    modal_shares["city"][modal_shares.city == "Frankfurt"] = 'Frankfurt_am_Main'
    modal_shares["city"][modal_shares.city == "The Hague"] = 'The_Hague'
    modal_shares["city"][modal_shares.city == "Zürich"] = 'Zurich'
    modal_shares["city"][modal_shares.city == "Córdoba"] = 'Cordoba'
    modal_shares["city"][modal_shares.city == "São Paulo"] = 'Sao_Paulo'
    modal_shares["city"][modal_shares.city == "Málaga"] = 'Malaga'
    modal_shares["city"][modal_shares.city == "Gent"] = 'Ghent'
    
    
    return modal_shares

def import_emissions_per_km(country, path_data):
    emissions_data = pd.read_excel(path_data + "GFEIAnnexC.xlsx", sheet_name = "Average CO2 emissions per km", header = 2, index_col = 0)
    emissions_data.columns = emissions_data.columns.map(str)
    emissions_data.rename(index={'United States':'United States of America'},inplace=True)
    emissions_data.rename(index={'Czech Republic':'Czechia'},inplace=True)

    for i in range(len(emissions_data)):    
        if np.isnan(emissions_data["2017"][i]):
            emissions_data["2017"][i] = copy.deepcopy(emissions_data["2015"][i])
    
        if np.isnan(emissions_data["2017"][i]):
            emissions_data["2017"][i] = copy.deepcopy(emissions_data["2013"][i])
            
    emissions_per_km = np.nanmedian(emissions_data["2017"])
    try:
        emissions_per_km = copy.deepcopy(emissions_data["2017"][country])
        
    except KeyError:
        print('Not in the database - we take the median')
        
    return emissions_per_km

def model(init, coeff_land, trans_price, income, INTEREST_RATE, HOUSEHOLD_SIZE, agricultural_rent, housing_supply = None, OPTION = None):
    """ Compute rents, densities and dwelling sizes """    

    ## Starting point
    beta = init[0]
    Ro = init[1]
    b = init[2]
    kappa = init[3]
    
    a=1-b
    
    income_net_of_transport_costs = np.fmax(income - trans_price, np.zeros(len(trans_price)))             
    rent = (Ro * income_net_of_transport_costs**(1/beta) /income**(1/beta))
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
    dwelling_size[dwelling_size == 0] = 1
    rent[rent == 0] = 1
    housing[np.isinf(housing)] = 0
    return rent, dwelling_size, density



def model_tax(init, coeff_land, trans_price, income, INTEREST_RATE, HOUSEHOLD_SIZE, agricultural_rent, distance_cbd, housing_supply = None, OPTION = None):
    """ Compute rents, densities and dwelling sizes """    

    ## Starting point
    beta = init[0]
    Ro = init[1]
    b = init[2]
    kappa = init[3]
    
    a=1-b
    
    income_net_of_transport_costs = np.fmax(income - trans_price, np.zeros(len(trans_price)))             
    rent = (Ro * income_net_of_transport_costs**(1/beta) /income**(1/beta))
    dwelling_size = beta * income_net_of_transport_costs / rent
    dwelling_size[rent == 0] = 0
    if OPTION == 1:
        housing = copy.deepcopy(housing_supply)
        housing[rent < agricultural_rent] = 0
    elif OPTION == 2:
        housing = coeff_land * ((kappa**(1/a)) * (((b / INTEREST_RATE) * ((1 - (distance_cbd / 400)) * rent)) ** (b/(a))))
        housing[housing < 0.99*housing_supply] = 0.99*housing_supply[housing < 0.99*housing_supply]
        housing[((1 - (distance_cbd / 400)) * rent) < agricultural_rent] = 0
    else:
        housing = coeff_land * ((kappa**(1/a)) * (((b / INTEREST_RATE) * ((1 - (distance_cbd / 400)) * rent)) ** (b/(a))))
        housing[((1 - (distance_cbd / 400)) * rent) < agricultural_rent] = 0
    #np.seterr(divide = 'ignore', invalid = 'ignore')
    rent[(1 - (distance_cbd / 400)) * rent < agricultural_rent] = agricultural_rent
    density = copy.deepcopy(housing / dwelling_size)
    density[dwelling_size == 0] = 0
    #np.seterr(divide = 'warn', invalid = 'warn')        
    density[np.isnan(density)] = 0    
    density[density == 0] = 1
    dwelling_size[dwelling_size == 0] = 1
    rent[rent == 0] = 1
    housing[np.isinf(housing)] = 0
    return rent, dwelling_size, density

class GridSimulation:
    """Define a grid defined by :
        
        - coord_X
        - coord_Y
        - distance_centre
        """
    
    def __init__(self,coord_X=0,coord_Y=0,distance_centre=0,area=0): 
        
        self.coord_X = coord_X
        self.coord_Y = coord_Y
        self.distance_centre = distance_centre
        self.area = area
        
        
    def create_grid(self,n):
        """Create a n*n grid, centered on 0"""
        
        coord_X = np.zeros(n*n)
        coord_Y = np.zeros(n*n)
    
        index = 0
        
        for i in range(n):
            for j in range(n):
                coord_X[index] = i - n/2
                coord_Y[index] = j - n/2
                index = index + 1
        distance_centre = (coord_X**2 + coord_Y**2) ** 0.5
    
        self.coord_X = coord_X
        self.coord_Y = coord_Y
        self.distance_centre = distance_centre
        self.area = 1

    def __repr__(self):
        
        return "Grid:\n  coord_X: {}\n  coord_Y: {}\n  distance_centre: {}\n  area: {}".format(
                self.coord_X, self.coord_Y, self.distance_centre, self.area)  


