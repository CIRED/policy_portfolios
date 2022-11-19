# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 15:27:03 2022

@author: charl
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 12:26:26 2021

@author: charl
"""


import pandas as pd
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import math
import os

option_ssp = 'SSP5'
BRT_scenario = 'speed_40_0_12_50_5'

#'speed_40_0_12_50_5'
#capital_evolution_25_0_12_50_income
#capital_evolution_25_0_12_15_income

option_ugb = "predict_urba" #density, data_urba, predict_urba. Résultats actuels avec predict_urba

FUEL_EFFICIENCY_DECREASE = 0.98 #0.963
BASELINE_EFFICIENCY_DECREASE = 0.99
LIFESPAN = 15

path_data = "C:/Users/charl/OneDrive/Bureau/City_dataStudy/"
path_folder = "C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Data/"
path_calibration = "C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/calibration_20211124/" #calibration_20211124
#os.mkdir(path_calibration)
#path_outputs = "C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/" + BRT_scenario + '/'
#path_outputs = "C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/sensitivity/kappa_agri_rent/kappa/BAU/"
#path_outputs = "C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/BAU_resid4_predict_urba/"
path_outputs = "C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/ssp5_20220805/"
os.mkdir(path_outputs)

from calibration.calibration import *
from calibration.validation import *
from inputs.data import *
#from inputs.geo_data import *
from inputs.land_use import *
from model.model import *
from outputs.outputs import *
from inputs.parameters import *
from inputs.transport import *

#from inputs.geo_data import *
from density_functions import *


option = {}
option["validation"] = 0
option["add_residuals"] = True
option["do_calibration"] = False
policy = 'None'

#List of cities
list_city = list_of_cities_and_databases(path_data,'cityDatabase')#cityDatabase_intermediaire

#Parameters and city characteristics
INTEREST_RATE, HOUSEHOLD_SIZE, TIME_LAG, DEPRECIATION_TIME, DURATION, COEFF_URB, FIXED_COST_CAR, WALKING_SPEED, CO2_EMISSIONS_TRANSIT = import_parameters()

#compute_density = pd.DataFrame(index = np.delete(np.unique(list_city.City), 153), columns = ['City', 'data_pop_2015', 'data_land_cover_2015', 'predicted_land_cover_2015', 'predicted_land_cover_2015_corrected', 'ESA_population_2015', 'predicted_population_2015', 'predicted_population_2015_corrected', 'data_pop_2035', 'predicted_population_2035', 'predicted_land_cover_2035', 'predicted_population_2035_corrected', 'predicted_land_cover_2035_corrected'])
#informal_housing = import_informal_housing(list_city, path_folder)
#gdp_capita_ppp = pd.read_excel(path_folder + "gdp_capita_ppp.xlsx")
#gdp_capita_ppp["income"] =, 'data_pop_2035'
#gdp_capita_ppp.income[np.isnan(gdp_capita_ppp.income)] = gdp_capita_ppp.brookings
#gdp_capita_ppp["source"] = ""
#gdp_capita_ppp["source"][np.isnan(gdp_capita_ppp.income)] = "WB"
#gdp_capita_ppp.income[np.isnan(gdp_capita_ppp.income)] = gdp_capita_ppp.world_bank

if option["validation"] == 1:
    (d_beta, d_b, d_kappa, d_Ro, d_selected_cells, d_share_car, d_share_walking, 
     d_share_transit, d_emissions_per_capita, d_utility, d_income, d_corr_density0, 
     d_corr_rent0, d_corr_size0, d_corr_density1, d_corr_rent1, d_corr_size1, 
     d_corr_density2, d_corr_rent2, d_corr_size2, r2_density0, r2_rent0, 
     r2_size0, r2_density1, r2_rent1, r2_size1, r2_density2, r2_rent2, r2_size2, 
     mae_density0, mae_rent0, mae_size0, mae_density1, mae_rent1, mae_size1, 
     mae_density2, mae_rent2, mae_size2, rae_density0, rae_rent0, rae_size0, 
     rae_density1, rae_rent1, rae_size1, rae_density2, rae_rent2, 
     rae_size2) = initialize_dict()

#df_cobenefits = pd.DataFrame(columns = ['City', 'Pol', 'cost_air_pollution', 'cost_active_modes', 'cost_noise', 
#                                        'cost_accidents', 'total_cost', 'emissions', 'avg_utility', 
#                                        'welfare_with_cobenefits', 'welfare_without_cobenefits', 'city_GDP'],
#                             index = [sub + '_baseline' for sub in np.unique(list_city.City)] + [sub + '_carbon_tax' for sub in np.unique(list_city.City)] + [sub + '_BRT' for sub in np.unique(list_city.City)] + [sub + '_fuel_effiency' for sub in np.unique(list_city.City)] + [sub + '_UGB' for sub in np.unique(list_city.City)])

DURATION = 35

for city in np.delete(np.unique(list_city.City), 153):
#for city in ["Turku"]:
    
    print("\n*** " + city + " ***\n")
    index = 0

    ### IMPORT DATA
    
    print("\n** Import data **\n")

    #Import city data
    (country, proj, density, rents_and_size, land_use, land_cover_ESACCI, driving, transit, grille, 
     centre, distance_cbd, conversion_rate) = import_data(list_city, path_data, city, path_folder)        
    
    #Density, rents and dwelling sizes
    density = density.loc[:,density.columns.str.startswith("density")]
    density = np.array(density).squeeze()
    rent = (rents_and_size.avgRent / conversion_rate) * 12
    size = rents_and_size.medSize
    if (city == 'Mashhad') |(city == 'Isfahan') | (city == 'Tehran'):
        rent = 100 * rent
    if city == "Abidjan":
        rent[size > 100] = np.nan
        size[size > 100] = np.nan
    if (city == "Casablanca") | (city == "Yerevan"):
        rent[size > 100] = np.nan
        size[size > 100] = np.nan
    if city == "Toluca":
        rent[size > 250] = np.nan
        size[size > 250] = np.nan
    if (city == "Manaus")| (city == "Rabat")| (city == "Sao_Paulo"):
        rent[rent > 200] = np.nan
        size[rent > 200] = np.nan
    if (city == "Salvador"):
        rent[rent > 250] = np.nan
        size[rent > 250] = np.nan
    if (city == "Karachi") | (city == "Lahore"):
        rent[size > 200] = np.nan
        size[size > 200] = np.nan
    if city == "Addis_Ababa":
        rent[rent > 400] = rent / 100
        size = size / 10
    size.mask((size > 1000), inplace = True)
    #informal_housing_city = informal_housing.informal_housing[informal_housing.City == city]
    #rent = (rentemp_capita_ppp.city == city].squeeze() == "WB":
    #    income = income * 1.33
    agricultural_rent = import_agricultural_rent(path_folder, country)
    population = np.nansum(density)
    region = pd.read_excel(path_folder + "city_country_region.xlsx").region[pd.read_excel(path_folder + "city_country_region.xlsx").city == city].squeeze()
    income = pd.read_excel(path_folder + "income.xlsx").income[pd.read_excel(path_folder + "income.xlsx").city == city].squeeze()
    coeff_land = (land_use.OpenedToUrb / land_use.TotalArea) * COEFF_URB
    density_to_cover = convert_density_to_urban_footprint(city, path_data, list_city)
    agricultural_rent_2015 = copy.deepcopy(agricultural_rent)
    
    #Import transport data
    fuel_price = import_fuel_price(country, 'gasoline', path_folder) #L/100km
    fuel_consumption = import_fuel_conso(country, path_folder) #dollars/L
    CO2_emissions_car = 2300 * fuel_consumption / 100
    monetary_cost_pt = import_public_transport_cost_data(path_folder, city).squeeze()
        
    #Import street network
    if (policy == 'BRT') | (policy == 'synergy'):
        orig_dist, target_dist, transit_dist, length_network = import_street_network(list_city, city, country, proj, centre, path_folder, path_data)
    else:
        orig_dist = np.nan
        target_dist = np.nan
        transit_dist = np.nan
        
    BRT_SPEED, BRT_OPERATING_COST, BRT_CAPITAL_COST, option_capital_cost_evolution = import_BRT_parameters(BRT_scenario)

    #Import scenarios
    #imaclim = pd.read_excel(path_folder + "Charlotte_ResultsIncomePriceAutoEmissionsAuto_V1.xlsx", sheet_name = 'Extraction_Baseline')
    #population_growth = import_city_scenarios(city, country, path_folder)
    #if isinstance(population_growth["2015-2020"], pd.Series) == True:
    #    population_growth = import_country_scenarios(country, path_folder)
    #income_growth = imaclim[(imaclim.Region == region) & (imaclim.Variable == "Index_income")].squeeze()
    
    ssp_population = np.array(load_ssp_nedum('OECD', option_ssp, city))[:, 0:18]
    population_growth = np.interp(np.arange(2015, 2101, 1), np.arange(2015, 2101, 5), ssp_population.squeeze())
    ssp_income = np.array(load_ssp_nedum('OECD', option_ssp, city))[:, 18:39]
    income_growth = np.interp(np.arange(2015, 2101, 1), np.arange(2015, 2101, 5), ssp_income.squeeze())
    
    
    #compute_density.predicted_density_2015[compute_density.index == city] = population /np.nansum(predicted_land_cover_2015)
    #compute_density.predicted_density_2015_corrected[compute_density.index == city] = population /np.nansum(np.fmax(predicted_land_cover_2015, land_cover_ESACCI.ESACCI190 / 1000000))

    #compute_density.to_excel("C:/Users/charl/OneDrive/Bureau/densities.xlsx")
         
    ### TRANSPORT MODELLING
    
    print("\n** Transport modelling **\n")
    
    prix_transport, mode_choice = transport_modeling(driving, transit, income, fuel_price, fuel_consumption, FIXED_COST_CAR, monetary_cost_pt, distance_cbd, WALKING_SPEED, policy, index, city, grille, centre, orig_dist, target_dist, transit_dist, BRT_SPEED)
    
    if (policy == 'BRT') | (policy == 'synergy'):
        BRT_OPERATING_COST = BRT_OPERATING_COST * length_network / 1000
        
    if (policy == 'BRT') | (policy == 'synergy'):        
        BRT_CAPITAL_COST = BRT_CAPITAL_COST * length_network / 1000   
        x = pd.Series({i: 1.767283 / (1.015 ** (2050 - i)) for i in range(2051, 2071)})
        income_growth = income_growth.append(x)         
        def compute_value_BRT(cost_per_year):
            if option_capital_cost_evolution == '50years_5percent':
                year_array = np.arange(0, 50)
                estimated_cost = sum(cost_per_year / (1.05 ** year_array))
            elif option_capital_cost_evolution == '15years_5percent':
                year_array = np.arange(0, 16)
                estimated_cost = sum(cost_per_year / (1.05 ** year_array)) 
            elif option_capital_cost_evolution == 'prop_income_50':
                year_array = np.arange(0, 50)
                estimated_cost = sum(cost_per_year * (income_growth[2020 + year_array] / income_growth[2020]))
            elif option_capital_cost_evolution == 'prop_income_15':
                year_array = np.arange(0, 16)
                estimated_cost = sum(cost_per_year * (income_growth[2020 + year_array] / income_growth[2020]))
            return np.abs(BRT_CAPITAL_COST - estimated_cost)
        capital_cost_year_one = sc.optimize.fsolve(compute_value_BRT, 50000000)
        if option_capital_cost_evolution == '50years_5percent':
            year_array = np.arange(0, 50)
            array_BRT_capital_cost = capital_cost_year_one / (1.05 ** year_array)
        elif option_capital_cost_evolution == '15years_5percent':
            year_array = np.arange(0, 16)
            array_BRT_capital_cost = capital_cost_year_one / (1.05 ** year_array)
        elif option_capital_cost_evolution == 'prop_income_50':
            year_array = np.arange(0, 50)
            array_BRT_capital_cost = capital_cost_year_one * (income_growth[2020 + year_array] / income_growth[2020])
        elif option_capital_cost_evolution == 'prop_income_15':
            year_array = np.arange(0, 16)
            array_BRT_capital_cost = capital_cost_year_one * (income_growth[2020 + year_array] / income_growth[2020])
    
    ### Calibration and parameters
    
    print("\n** Calibration and parameters **\n")
    
    selected_cells = np.array(prix_transport.notnull() & (prix_transport!=0)
                                  & coeff_land.notnull()  & (coeff_land !=0)
                                  & rent.notnull()  & (rent!=0)
                                  & size.notnull()  & (size!=0)
                                  & (~np.isnan(density)) & (density!=0) & (land_cover_ESACCI.ESACCI190 > 500000))
    
    selected_cells2 = np.array(prix_transport.notnull() & (prix_transport!=0)
                                  & coeff_land.notnull()  & (coeff_land !=0)
                                  & rent.notnull()  & (rent!=0)
                                  & size.notnull()  & (size!=0)
                                  & (~np.isnan(density)) & (density!=0))
    
    density_max = np.nanmax(density)
    #density_max = 1000000000
        
    if option["do_calibration"] == True:
        
        density_max = np.nanmax(density)
        #density_max = 1000000000
        
        result_calibration = calibration2(city, rent, density, size, 
                                          prix_transport, INTEREST_RATE,
                                          selected_cells, HOUSEHOLD_SIZE, 
                                          coeff_land, agricultural_rent, income)
    
        BETA = result_calibration.x[0]
        B = result_calibration.x[2]
        kappa = result_calibration.x[3] #compute_kappa(100, B, income)
        Ro = result_calibration.x[1]
        
    else:
        BETA = np.array(np.load(path_calibration + "beta.npy", allow_pickle = True), ndmin = 1)[0][city]
        B = np.array(np.load(path_calibration + "b.npy", allow_pickle = True), ndmin = 1)[0][city]
        kappa = np.array(np.load(path_calibration + "kappa.npy", allow_pickle = True), ndmin = 1)[0][city]
        Ro = np.array(np.load(path_calibration + "Ro.npy", allow_pickle = True), ndmin = 1)[0][city]

    param = np.array([BETA, Ro, B, kappa])
    simul_rent, simul_size, simul_density = model2(param, coeff_land, prix_transport, income, INTEREST_RATE, HOUSEHOLD_SIZE, agricultural_rent)
    simul_density[simul_density > density_max] = density_max
    
    if option["validation"] == 1: 
        
        print("export validation")
        
        d_selected_cells[city] = sum(selected_cells2)
        d_income[city] = income      
        d_beta[city] = result_calibration.x[0]
        d_b[city] = result_calibration.x[2]
        d_kappa[city] = result_calibration.x[3]
        d_Ro[city] = result_calibration.x[1]
        d_corr_density2[city], d_corr_rent2[city], d_corr_size2[city], d_corr_density1[city], d_corr_rent1[city], d_corr_size1[city], d_corr_density0[city], d_corr_rent0[city], d_corr_size0[city] = compute_corr(density, rent, size, simul_density, simul_rent, simul_size, selected_cells, selected_cells2)
        r2_density2[city], r2_rent2[city], r2_size2[city], r2_density1[city], r2_rent1[city], r2_size1[city], r2_density0[city], r2_rent0[city], r2_size0[city] = compute_r2(density, rent, size, simul_density, simul_rent, simul_size, selected_cells, selected_cells2)
        mae_density2[city], mae_rent2[city], mae_size2[city], mae_density1[city], mae_rent1[city], mae_size1[city], mae_density0[city], mae_rent0[city], mae_size0[city] = compute_maes(density, rent, size, simul_density, simul_rent, simul_size, selected_cells, selected_cells2)
        rae_density2[city], rae_rent2[city], rae_size2[city], rae_density1[city], rae_rent1[city], rae_size1[city], rae_density0[city], rae_rent0[city], rae_size0[city] = compute_raes(density, rent, size, simul_density, simul_rent, simul_size, selected_cells, selected_cells2)
        d_share_transit[city], d_share_car[city], d_share_walking[city] = compute_modal_shares(density, mode_choice, population)       
        export_charts(distance_cbd, simul_rent, rent, simul_density, density, simul_size, size, path_calibration, city)       
        d_emissions_per_capita[city] = compute_emissions(CO2_emissions_car, CO2_EMISSIONS_TRANSIT, density, mode_choice, driving.Distance / 1000, transit.Distance / 1000) / population
        d_utility[city] = compute_avg_utility(income, BETA, Ro)

#save_outputs_validation(path_calibration, d_beta, d_b, d_kappa, d_Ro, d_share_transit, d_share_car, d_share_walking, d_emissions_per_capita, d_utility, d_income, d_selected_cells, d_corr_density0, d_corr_rent0, d_corr_size0, d_corr_density1, d_corr_rent1, d_corr_size1, d_corr_density2, d_corr_rent2, d_corr_size2, r2_density0, r2_rent0, r2_size0, r2_density1, r2_rent1, r2_size1, r2_density2, r2_rent2, r2_size2, mae_density0, mae_rent0, mae_size0, mae_density1, mae_rent1, mae_size1, mae_density2, mae_rent2, mae_size2, rae_density0, rae_rent0, rae_size0, rae_density1, rae_rent1, rae_size1, rae_density2, rae_rent2, rae_size2)

    #Residuals
    residuals_for_simulation = compute_residuals(density, simul_density, rent, simul_rent, size, simul_size, option)
          
    #### INITIAL STATE
    
    print("\n** Initial state **\n")
    
    def compute_residual(ro):
        init = np.array([BETA, float(ro), B, kappa])
        simul_rent, simul_size, simul_density = model2(init, coeff_land, prix_transport, income, INTEREST_RATE, HOUSEHOLD_SIZE, agricultural_rent)
        simul_density[simul_density > density_max] = density_max
        if option["add_residuals"] == True:
            simul_rent = simul_rent * np.exp(residuals_for_simulation.rent_residual)
            simul_density = simul_density * np.exp(residuals_for_simulation.density_residual)
            simul_size = simul_size * np.exp(residuals_for_simulation.size_residual)
        delta_population = np.abs(population - np.nansum(simul_density))
        return delta_population

    R_0 = sc.optimize.fsolve(compute_residual, Ro) #432 / 458
    X0 = np.array([BETA, float(R_0), B, kappa])
    simul_rent, simul_size, simul_density = model2(X0, coeff_land, prix_transport, income, INTEREST_RATE, HOUSEHOLD_SIZE, agricultural_rent)
    simul_density[simul_density > density_max] = density_max
    housing_t0 = simul_size * simul_density
    if option["add_residuals"] == True:
        simul_rent = simul_rent * np.exp(residuals_for_simulation.rent_residual)
        simul_density = simul_density * np.exp(residuals_for_simulation.density_residual)
        simul_size = simul_size * np.exp(residuals_for_simulation.size_residual)

    #init outputs
    save_density, save_rent, save_dwelling_size, save_population, save_population2, save_income, save_R0, save_emissions, save_emissions_per_capita, save_avg_utility, save_total_welfare, save_total_welfare_with_cobenefits, save_urbanized_area, save_distance, save_modal_shares = init_outputs(DURATION, distance_cbd)
    
    save_cost_BRT_per_pers = np.zeros(DURATION + 1)
    
    #save outputs t = 0
    save_density[0, :] = simul_density
    save_rent[0, :] = simul_rent
    save_dwelling_size[0, :] = simul_size
    save_population[0] = population
    save_population2[0] = sum(simul_density)
    save_income[0] = income
    save_R0[0] = R_0
    save_emissions[0] = compute_emissions(CO2_emissions_car, CO2_EMISSIONS_TRANSIT, simul_density, mode_choice, driving.Distance / 1000, transit.Distance / 1000)
    save_emissions_per_capita[0] = save_emissions[0] / population
    save_avg_utility[0] = compute_avg_utility(income, BETA, save_R0[0])
    #save_total_welfare[0] = compute_total_welfare(income, prix_transport, simul_rent, simul_size, BETA, simul_density, mode_choice, distance_cbd, WALKING_SPEED, country, path_folder, 2015 + index, region, FUEL_EFFICIENCY_DECREASE, BASELINE_EFFICIENCY_DECREASE, LIFESPAN, imaclim, policy, cobenefits = False)
    #save_total_welfare_with_cobenefits[0] = compute_total_welfare(income, prix_transport, simul_rent, simul_size, BETA, simul_density, mode_choice, distance_cbd, WALKING_SPEED, country, path_folder, 2015 + index, region, FUEL_EFFICIENCY_DECREASE, BASELINE_EFFICIENCY_DECREASE, LIFESPAN, imaclim, policy, cobenefits = True)
    save_urbanized_area[0] = np.nansum(predict_urbanized_area(density_to_cover, save_density[0]))
    save_distance = distance_cbd
    save_modal_shares[0] = mode_choice
    
    #compute_density.City[compute_density.index == city] = city
    #compute_density.data_land_cover_2015[compute_density.index == city] = np.nansum(land_cover_ESACCI.ESACCI190 / 1000000)
    #compute_density.data_pop_2015[compute_density.index == city] = population    
    #predicted_land_cover_2015 = predict_urbanized_area(density_to_cover, simul_density)
    #compute_density.predicted_land_cover_2015[compute_density.index == city] = np.nansum(predicted_land_cover_2015)
    #compute_density.predicted_land_cover_2015_corrected[compute_density.index == city] = np.nansum(np.fmax(predicted_land_cover_2015, land_cover_ESACCI.ESACCI190 / 1000000))
    #compute_density.ESA_population_2015[compute_density.index == city] = np.nansum(simul_density[land_cover_ESACCI.ESACCI190 / 1000000 >0])    
    #compute_density.predicted_population_2015[compute_density.index == city] = np.nansum(simul_density[predicted_land_cover_2015 > 0])    
    #compute_density.predicted_population_2015_corrected[compute_density.index == city] = np.nansum(simul_density[np.fmax(predicted_land_cover_2015, land_cover_ESACCI.ESACCI190 / 1000000) > 0])    
    
    
    #welfare, air_pollution, active_modes, noise, car_accidents = compute_total_welfare2(income, prix_transport, simul_rent, simul_size, BETA, simul_density, mode_choice, distance_cbd, WALKING_SPEED, country, path_folder, 2015 + index, region, FUEL_EFFICIENCY_DECREASE, BASELINE_EFFICIENCY_DECREASE, LIFESPAN, imaclim, policy, cobenefits = True)
    
    while index < DURATION:
    
        ### ADJUST PARAMETERS     
        index = index + 1
        #population = update_population(population, population_growth, index)
        #income = income * (income_growth[2015 + index] / income_growth[2015 + index - 1])
        population = save_population[0] * population_growth[index]
        income = save_income[0] * income_growth[index]
        #agricultural_rent = agricultural_rent_2015 * income_growth[index]
        ##### !!!! UPDATE KAPPA ET AGRI RENT kappa * ((income / income_lag) ** 2)
        #kappa = update_kappa(kappa, save_income[index-1], income, B)
        
        #fuel_price = fuel_price  * income_growth[index]
        #monetary_cost_pt = monetary_cost_pt * (income_growth[2015 + index] / income_growth[2015 + index - 1])
        #capital_cost_BRT
        
        ### ACCOUNT FOR POLICIES
        if ((policy == 'BRT') | (policy == 'synergy')):
            BRT_OPERATING_COST = BRT_OPERATING_COST * (income_growth[2015 + index] / income_growth[2015 + index - 1])
            if index > 4:
                if (option_capital_cost_evolution == '50years_5percent') | (option_capital_cost_evolution == '15years_5percent'):
                    BRT_CAPITAL_COST = array_BRT_capital_cost[index - 5]
                elif (option_capital_cost_evolution == 'prop_income_50') | (option_capital_cost_evolution == 'prop_income_15'):
                    BRT_CAPITAL_COST = array_BRT_capital_cost[2015 + index]
                
        if ((policy == 'fuel_efficiency') & (index > 4)):
            #fuel_consumption = fuel_consumption * FUEL_EFFICIENCY_DECREASE
            fuel_consumption = (((LIFESPAN - 1)/LIFESPAN) + ((1/LIFESPAN) * (BASELINE_EFFICIENCY_DECREASE ** (5 - index + 15)) * (FUEL_EFFICIENCY_DECREASE ** (index - 5)))) * fuel_consumption
            
        else:
            #fuel_consumption = BASELINE_EFFICIENCY_DECREASE * fuel_consumption
            fuel_consumption = (((LIFESPAN - 1)/LIFESPAN) + ((BASELINE_EFFICIENCY_DECREASE ** LIFESPAN) /LIFESPAN)) * fuel_consumption
            
        CO2_emissions_car = 2300 * fuel_consumption / 100
        
        #if index > 4:
        #    WALKING_SPEED = 15
            
        if (policy == 'carbon_tax') | (policy == 'synergy'):
            if index == 5:
                fuel_price = fuel_price * 1.3
        
        if policy == 'UGB':
            if index > 4:
                if option_ugb == "predict_urba":
                    urban_footprint = predict_urbanized_area(density_to_cover, save_density[4])
                    coeff_land = np.fmin(coeff_land, 0.62 * urban_footprint) 
                elif option_ugb == 'density':
                    coeff_land[save_density[4] < 400] = 0
                elif option_ugb == 'data_urba':
                    coeff_land = np.fmin(coeff_land, 0.62 * land_cover_ESACCI.ESACCI190 / land_cover_ESACCI.AREA)
                
        #ADJUST TRANSPORT AND LAND-USE
        prix_transport, mode_choice = transport_modeling(driving, transit, income, fuel_price, fuel_consumption, FIXED_COST_CAR, monetary_cost_pt, distance_cbd, WALKING_SPEED, policy, index, city, grille, centre, orig_dist, target_dist, transit_dist, BRT_SPEED)
          
        ### COMPUTE EQUILIBRIUM WITHOUT INERTIA
        
        if ((policy == 'BRT') & (index > 4)) | ((policy == 'synergy') & (index > 4)):
            cost_BRT_per_pers = (BRT_OPERATING_COST + BRT_CAPITAL_COST) / population
        else:
            cost_BRT_per_pers = 0
            
        if ((policy == 'None') | (policy == "UGB") | (policy == "fuel_efficiency") | (policy == 'BRT') | ((policy == 'carbon_tax') & (index < 5)) | ((policy == 'synergy') & (index < 5))):           
            def compute_residual(ro):
                simul_rent, simul_size, simul_density = model2(np.array([BETA, float(ro), B, kappa]), coeff_land, prix_transport, income - cost_BRT_per_pers, INTEREST_RATE, HOUSEHOLD_SIZE, agricultural_rent, housing_t0, 2)
                if option["add_residuals"] == True:
                    simul_rent = simul_rent * np.exp(residuals_for_simulation.rent_residual)
                    simul_density = simul_density * np.exp(residuals_for_simulation.density_residual)
                    simul_size = simul_size * np.exp(residuals_for_simulation.size_residual)
                delta_population = np.abs(population - np.nansum(simul_density))
                return delta_population   
        
            R_0_without_inertia = sc.optimize.fsolve(compute_residual, R_0)          
            rent_without_inertia, dwelling_size_without_inertia, density_without_inertia = model2(np.array([BETA, R_0_without_inertia, B, kappa]), coeff_land, prix_transport, income - cost_BRT_per_pers, INTEREST_RATE, HOUSEHOLD_SIZE, agricultural_rent, housing_t0, 2)
           
        elif ((policy == 'carbon_tax') & (index > 4)):
            def compute_residual(init_array):
                ro = init_array[0]
                agg_tax_input = init_array[1]
                simul_rent, simul_size, simul_density = model2(np.array([BETA, float(ro), B, kappa]), coeff_land, prix_transport, income + (agg_tax_input / population), INTEREST_RATE, HOUSEHOLD_SIZE, agricultural_rent, housing_t0, 2)
                if option["add_residuals"] == True:
                    simul_rent = simul_rent * np.exp(residuals_for_simulation.rent_residual)
                    simul_density = simul_density * np.exp(residuals_for_simulation.density_residual)
                    simul_size = simul_size * np.exp(residuals_for_simulation.size_residual)
                delta_population = np.abs(population - np.nansum(simul_density))
                simul_agg_tax = 2*365 * (np.nansum(simul_density[mode_choice == 0] * driving.Distance[mode_choice == 0] / 1000) * ((0.3/1.3) * fuel_price * fuel_consumption / 100))
                delta_tax = np.abs(simul_agg_tax - agg_tax_input)
                return np.array([delta_population, delta_tax], dtype='float64')                          
            R_0_without_inertia, aggregated_tax_without_inertia = sc.optimize.fsolve(compute_residual, np.array([R_0.squeeze(), 0]))       
            rent_without_inertia, dwelling_size_without_inertia, density_without_inertia = model2(np.array([BETA, R_0_without_inertia, B, kappa]), coeff_land, prix_transport, income + (aggregated_tax_without_inertia / population), INTEREST_RATE, HOUSEHOLD_SIZE, agricultural_rent, housing_t0, 2)
        
        elif ((policy == 'synergy') & (index > 4)):
            def compute_residual(init_array):
                ro = init_array[0]
                agg_tax_input = init_array[1]
                simul_rent, simul_size, simul_density = model2(np.array([BETA, float(ro), B, kappa]), coeff_land, prix_transport, income - cost_BRT_per_pers + (agg_tax_input / population), INTEREST_RATE, HOUSEHOLD_SIZE, agricultural_rent, housing_t0, 2)
                if option["add_residuals"] == True:
                    simul_rent = simul_rent * np.exp(residuals_for_simulation.rent_residual)
                    simul_density = simul_density * np.exp(residuals_for_simulation.density_residual)
                    simul_size = simul_size * np.exp(residuals_for_simulation.size_residual)
                delta_population = np.abs(population - np.nansum(simul_density))
                simul_agg_tax = 2*365 * (np.nansum(simul_density[mode_choice == 0] * driving.Distance[mode_choice == 0] / 1000) * ((0.3/1.3) * fuel_price * fuel_consumption / 100))
                delta_tax = np.abs(simul_agg_tax - agg_tax_input)
                return np.array([delta_population, delta_tax], dtype='float64')                          
            R_0_without_inertia, aggregated_tax_without_inertia = sc.optimize.fsolve(compute_residual, np.array([R_0.squeeze(), 0]))       
            rent_without_inertia, dwelling_size_without_inertia, density_without_inertia = model2(np.array([BETA, R_0_without_inertia, B, kappa]), coeff_land, prix_transport, income - cost_BRT_per_pers + (aggregated_tax_without_inertia / population), INTEREST_RATE, HOUSEHOLD_SIZE, agricultural_rent, housing_t0, 2)
      
        housing_without_inertia = density_without_inertia * dwelling_size_without_inertia
        if option["add_residuals"] == True:
            rent_without_inertia = rent_without_inertia * np.exp(residuals_for_simulation.rent_residual)
            dwelling_size_without_inertia = dwelling_size_without_inertia * np.exp(residuals_for_simulation.size_residual)
            density_without_inertia = density_without_inertia * np.exp(residuals_for_simulation.density_residual)
        
    
        #COMPUTE EQUILIBRIUM WITH INERTIAM
        
        housing_supply_t1 = compute_housing_supply(housing_without_inertia, housing_t0, TIME_LAG, DEPRECIATION_TIME)
            
        if ((policy == 'None') | (policy == "UGB") | (policy == "fuel_efficiency") | (policy == 'BRT') | ((policy == 'carbon_tax') & (index < 5)) | ((policy == 'synergy') & (index < 5))):
            def compute_residual(ro):
                simul_rent, simul_size, simul_density = model2(np.array([BETA, float(ro), B, kappa]), coeff_land, prix_transport, income - cost_BRT_per_pers, INTEREST_RATE, HOUSEHOLD_SIZE, agricultural_rent, housing_supply_t1, 1)            
                if option["add_residuals"] == True:
                    simul_rent = simul_rent * np.exp(residuals_for_simulation.rent_residual)
                    simul_density = simul_density * np.exp(residuals_for_simulation.density_residual)
                    simul_size = simul_size * np.exp(residuals_for_simulation.size_residual)
                delta_population = np.abs(population - np.nansum(simul_density))
                return delta_population
                
            R_0 = sc.optimize.fsolve(compute_residual, R_0_without_inertia)     
            simul_rent, simul_size, simul_density = model2(np.array([BETA, float(R_0), B, kappa]), coeff_land, prix_transport, income - cost_BRT_per_pers, INTEREST_RATE, HOUSEHOLD_SIZE, agricultural_rent, housing_supply_t1, 1)
        
        elif ((policy == 'carbon_tax') & (index > 4)):     
            def compute_residual(init_array):
                ro = init_array[0]
                agg_tax_input = init_array[1]
                simul_rent, simul_size, simul_density = model2(np.array([BETA, float(ro), B, kappa]), coeff_land, prix_transport, income + (agg_tax_input / population), INTEREST_RATE, HOUSEHOLD_SIZE, agricultural_rent, housing_supply_t1, 1)            
                if option["add_residuals"] == True:
                    simul_rent = simul_rent * np.exp(residuals_for_simulation.rent_residual)
                    simul_density = simul_density * np.exp(residuals_for_simulation.density_residual)
                    simul_size = simul_size * np.exp(residuals_for_simulation.size_residual)
                delta_population = np.abs(population - np.nansum(simul_density))
                simul_agg_tax = 2*365 * (np.nansum(simul_density[mode_choice == 0] * driving.Distance[mode_choice == 0] / 1000) * ((0.3/1.3) * fuel_price * fuel_consumption / 100))
                delta_tax = np.abs(simul_agg_tax - agg_tax_input)
                return np.array([delta_population, delta_tax], dtype='float64')
        
            R_0, aggregated_tax = sc.optimize.fsolve(compute_residual, np.array([R_0_without_inertia, aggregated_tax_without_inertia]))        
            simul_rent, simul_size, simul_density = model2(np.array([BETA, float(R_0), B, kappa]), coeff_land, prix_transport, income + (aggregated_tax / population), INTEREST_RATE, HOUSEHOLD_SIZE, agricultural_rent, housing_supply_t1, 1)
        
        elif ((policy == 'synergy') & (index > 4)):     
            def compute_residual(init_array):
                ro = init_array[0]
                agg_tax_input = init_array[1]
                simul_rent, simul_size, simul_density = model2(np.array([BETA, float(ro), B, kappa]), coeff_land, prix_transport, income - cost_BRT_per_pers + (agg_tax_input / population), INTEREST_RATE, HOUSEHOLD_SIZE, agricultural_rent, housing_supply_t1, 1)            
                if option["add_residuals"] == True:
                    simul_rent = simul_rent * np.exp(residuals_for_simulation.rent_residual)
                    simul_density = simul_density * np.exp(residuals_for_simulation.density_residual)
                    simul_size = simul_size * np.exp(residuals_for_simulation.size_residual)
                delta_population = np.abs(population - np.nansum(simul_density))
                simul_agg_tax = 2*365 * (np.nansum(simul_density[mode_choice == 0] * driving.Distance[mode_choice == 0] / 1000) * ((0.3/1.3) * fuel_price * fuel_consumption / 100))
                delta_tax = np.abs(simul_agg_tax - agg_tax_input)
                return np.array([delta_population, delta_tax], dtype='float64')
        
            R_0, aggregated_tax = sc.optimize.fsolve(compute_residual, np.array([R_0_without_inertia, aggregated_tax_without_inertia]))        
            simul_rent, simul_size, simul_density = model2(np.array([BETA, float(R_0), B, kappa]), coeff_land, prix_transport, income - cost_BRT_per_pers + (aggregated_tax / population), INTEREST_RATE, HOUSEHOLD_SIZE, agricultural_rent, housing_supply_t1, 1)
        
        housing_t0 = simul_size * simul_density
        if index == 4:
            copy_simul_density = copy.deepcopy(simul_density)
        if option["add_residuals"] == True:
            simul_rent = simul_rent * np.exp(residuals_for_simulation.rent_residual)
            simul_size = simul_size * np.exp(residuals_for_simulation.size_residual)
            simul_density = simul_density * np.exp(residuals_for_simulation.density_residual)
    
        
        #save outputs
        save_density[index, :] = simul_density
        save_rent[index, :] = simul_rent
        save_dwelling_size[index, :] = simul_size
        save_population[index] = population
        save_population2[index] = sum(simul_density)
        save_income[index] = income
        save_R0[index] = R_0
        save_emissions[index] = compute_emissions(CO2_emissions_car, CO2_EMISSIONS_TRANSIT, simul_density, mode_choice, driving.Distance / 1000, transit.Distance / 1000)
        save_emissions_per_capita[index] = save_emissions[index] / population
        if ((policy == 'None') | (policy == "UGB") | (policy == "fuel_efficiency") | (policy == 'BRT') | ((policy == 'carbon_tax') & (index < 5))| ((policy == 'synergy') & (index < 5))):
            #save_avg_utility[index] = compute_avg_utility(income - cost_BRT_per_pers, BETA, save_R0[index])
            #save_total_welfare[index] = compute_total_welfare(income - cost_BRT_per_pers, prix_transport, simul_rent, simul_size, BETA, simul_density, mode_choice, distance_cbd, WALKING_SPEED, country, path_folder, 2015 + index, region, FUEL_EFFICIENCY_DECREASE, BASELINE_EFFICIENCY_DECREASE, LIFESPAN, imaclim, policy, cobenefits = False)
            #save_total_welfare_with_cobenefits[index] = compute_total_welfare(income - cost_BRT_per_pers, prix_transport, simul_rent, simul_size, BETA, simul_density, mode_choice, distance_cbd, WALKING_SPEED, country, path_folder, 2015 + index, region, FUEL_EFFICIENCY_DECREASE, BASELINE_EFFICIENCY_DECREASE, LIFESPAN, imaclim, policy, cobenefits = True)
            print("we don't save welfare")
        elif ((policy == 'carbon_tax') & (index > 4)): 
            save_avg_utility[index] = compute_avg_utility(income + (aggregated_tax / population), BETA, save_R0[index])
            save_total_welfare[index] = compute_total_welfare(income + (aggregated_tax / population), prix_transport, simul_rent, simul_size, BETA, simul_density, mode_choice, distance_cbd, WALKING_SPEED, country, path_folder, 2015 + index, region, FUEL_EFFICIENCY_DECREASE, BASELINE_EFFICIENCY_DECREASE, LIFESPAN, imaclim, policy, cobenefits = False)
            save_total_welfare_with_cobenefits[index] = compute_total_welfare(income + (aggregated_tax / population), prix_transport, simul_rent, simul_size, BETA, simul_density, mode_choice, distance_cbd, WALKING_SPEED, country, path_folder, 2015 + index, region, FUEL_EFFICIENCY_DECREASE, BASELINE_EFFICIENCY_DECREASE, LIFESPAN, imaclim, policy, cobenefits = True)
        elif ((policy == 'synergy') & (index > 4)): 
            save_avg_utility[index] = compute_avg_utility(income - cost_BRT_per_pers + (aggregated_tax / population), BETA, save_R0[index])
            save_total_welfare[index] = compute_total_welfare(income - cost_BRT_per_pers + (aggregated_tax / population), prix_transport, simul_rent, simul_size, BETA, simul_density, mode_choice, distance_cbd, WALKING_SPEED, country, path_folder, 2015 + index, region, FUEL_EFFICIENCY_DECREASE, BASELINE_EFFICIENCY_DECREASE, LIFESPAN, imaclim, policy, cobenefits = False)
            save_total_welfare_with_cobenefits[index] = compute_total_welfare(income - cost_BRT_per_pers + (aggregated_tax / population), prix_transport, simul_rent, simul_size, BETA, simul_density, mode_choice, distance_cbd, WALKING_SPEED, country, path_folder, 2015 + index, region, FUEL_EFFICIENCY_DECREASE, BASELINE_EFFICIENCY_DECREASE, LIFESPAN, imaclim, policy, cobenefits = True)
        save_urbanized_area[index] = np.nansum(predict_urbanized_area(density_to_cover, save_density[index]))
        save_modal_shares[index] = mode_choice
        save_cost_BRT_per_pers[index] = cost_BRT_per_pers
        
        #welfare, air_pollution, active_modes, noise, car_accidents = compute_total_welfare2(income, prix_transport, simul_rent, simul_size, BETA, simul_density, mode_choice, distance_cbd, WALKING_SPEED, country, path_folder, 2015 + index, region, FUEL_EFFICIENCY_DECREASE, BASELINE_EFFICIENCY_DECREASE, LIFESPAN, imaclim, policy, cobenefits = True)
        
        #df_cobenefits.loc[city + '_UGB', 'City'] = city
        #df_cobenefits.loc[city + '_UGB', 'Pol'] = policy
        #df_cobenefits.loc[city + '_UGB', 'cost_air_pollution'] = air_pollution
        #df_cobenefits.loc[city + '_UGB', 'cost_active_modes'] = active_modes
        #df_cobenefits.loc[city + '_UGB', 'cost_noise'] = noise
        #df_cobenefits.loc[city + '_UGB', 'cost_accidents'] = car_accidents
        #df_cobenefits.loc[city + '_UGB', 'emissions'] = save_emissions_per_capita[index]
        #df_cobenefits.loc[city + '_UGB', 'avg_utility'] = save_avg_utility[index]
        #df_cobenefits.loc[city + '_UGB', 'welfare_with_cobenefits'] = save_total_welfare_with_cobenefits[index]
        #df_cobenefits.loc[city + '_UGB', 'welfare_without_cobenefits'] = save_total_welfare[index]
        #df_cobenefits.loc[city + '_UGB', 'city_GDP'] = income * population
        
    ### EXPORT OUTPUTS
    #plot_emissions_and_welfare(save_emissions_per_capita, save_total_welfare, save_total_welfare_with_cobenefits, path_outputs, city)      
    #save_outputs(save_emissions, save_emissions_per_capita, save_population, save_R0, save_rent, save_dwelling_size, save_density, save_avg_utility, save_total_welfare, save_total_welfare_with_cobenefits, save_income, save_urbanized_area, save_distance, save_modal_shares, path_outputs, city)
    np.save(path_outputs + city + "_ua.npy", save_urbanized_area)
    np.save(path_outputs + city + "_pop.npy", save_population)
    np.save(path_outputs + city + "_income.npy", save_income)
    np.save(path_outputs + city + "_density.npy", save_density)
    #np.save(path_outputs + city + "_cost_BRT_per_pers.npy", save_cost_BRT_per_pers)

    #predicted_land_cover_2035 = predict_urbanized_area(density_to_cover, simul_density)
    #predicted_land_cover_2035_v2 = np.fmax(predicted_land_cover_2035, land_cover_ESACCI.ESACCI190 / 1000000)
    #compute_density.predicted_land_cover_2035_corrected[compute_density.index == city] = np.nansum(predicted_land_cover_2035_v2)
    #compute_density.predicted_population_2035_corrected[compute_density.index == city] = np.nansum(simul_density * predicted_land_cover_2035_v2)
    
    #compute_density.data_pop_2035[compute_density.index == city] = population   
    #predicted_land_cover_2035 = predict_urbanized_area(density_to_cover, simul_density)
    
    #compute_density.predicted_land_cover_2035[compute_density.index == city] = np.nansum(predicted_land_cover_2035)
    #compute_density.predicted_land_cover_2035_corrected[compute_density.index == city] = np.nansum(np.fmax(predicted_land_cover_2035, land_cover_ESACCI.ESACCI190 / 1000000))
    #compute_density.predicted_population_2035[compute_density.index == city] = np.nansum(simul_density[predicted_land_cover_2035 > 0])    
    #compute_density.predicted_population_2035_corrected[compute_density.index == city] = np.nansum(simul_density[np.fmax(predicted_land_cover_2035, land_cover_ESACCI.ESACCI190 / 1000000) > 0])    
    

      
#compute_density.to_excel("C:/Users/charl/OneDrive/Bureau/scenarios_densities_20220309.xlsx")

    