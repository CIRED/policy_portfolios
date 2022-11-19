# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 16:22:10 2021

@author: charl
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from outputs.cobenefits import *

##### COMPUTE OUTPUTS

def compute_emissions(CO2_emissions_car, CO2_EMISSIONS_TRANSIT, density, trans_mode, driving_distance, transit_distance, scenar_emissions = None, initial_year = None, index = None):
    
    return 2*365 * ((np.nansum(density[trans_mode == 0] * driving_distance[trans_mode == 0]) * CO2_emissions_car) + (np.nansum(density[trans_mode == 1] * transit_distance[trans_mode == 1]) * (CO2_EMISSIONS_TRANSIT)))

def compute_avg_utility(income, BETA, R0):
    return (income  * ((1 - BETA) ** (1 - BETA)) * (BETA ** BETA)) / (R0 ** BETA)
    
def compute_total_welfare(income, prix_transport, simul_rent, simul_size, BETA, simul_density, mode_choice, distance_cbd, WALKING_SPEED, Country, path_folder, year, Region, FUEL_EFFICIENCY_DECREASE, BASELINE_EFFICIENCY_DECREASE, LIFESPAN, imaclim = 0, policy = 0, cobenefits = False):
    if cobenefits == True:
        air_pollution = compute_air_pollution(simul_density, mode_choice, distance_cbd, Country, path_folder, year, Region, imaclim, policy, FUEL_EFFICIENCY_DECREASE, BASELINE_EFFICIENCY_DECREASE, LIFESPAN)
        active_modes = compute_active_modes(simul_density, mode_choice, distance_cbd, WALKING_SPEED, Country, path_folder, Region, year, imaclim)
        noise = compute_noise(simul_density, mode_choice, distance_cbd, Country, Region, year, imaclim, path_folder)
        car_accidents = compute_car_accidents(simul_density, mode_choice, distance_cbd, Country, Region, year, imaclim, path_folder)
    elif cobenefits == False:
        air_pollution = 0
        active_modes = 0
        noise = 0
        car_accidents = 0
    monetary_value_cobenefits = (active_modes - air_pollution - noise - car_accidents) / np.nansum(simul_density)
    vec_utility = (((income - prix_transport - (simul_rent * simul_size) + monetary_value_cobenefits) ** (1 - BETA)) * ((simul_size) ** (BETA)))
    return np.nansum(vec_utility * simul_density)

def compute_total_welfare2(income, prix_transport, simul_rent, simul_size, BETA, simul_density, mode_choice, distance_cbd, WALKING_SPEED, Country, path_folder, year, Region, FUEL_EFFICIENCY_DECREASE, BASELINE_EFFICIENCY_DECREASE, LIFESPAN, imaclim = 0, policy = 0, cobenefits = False):
    if cobenefits == True:
        air_pollution = compute_air_pollution(simul_density, mode_choice, distance_cbd, Country, path_folder, year, Region, imaclim, policy, FUEL_EFFICIENCY_DECREASE, BASELINE_EFFICIENCY_DECREASE, LIFESPAN)
        active_modes = compute_active_modes(simul_density, mode_choice, distance_cbd, WALKING_SPEED, Country, path_folder, Region, year, imaclim)
        noise = compute_noise(simul_density, mode_choice, distance_cbd, Country, Region, year, imaclim, path_folder)
        car_accidents = compute_car_accidents(simul_density, mode_choice, distance_cbd, Country, Region, year, imaclim, path_folder)
    elif cobenefits == False:
        air_pollution = 0
        active_modes = 0
        noise = 0
        car_accidents = 0
    monetary_value_cobenefits = (active_modes - air_pollution - noise - car_accidents) / np.nansum(simul_density)
    vec_utility = (((income - prix_transport - (simul_rent * simul_size) + monetary_value_cobenefits) ** (1 - BETA)) * ((simul_size) ** (BETA)))
    return np.nansum(vec_utility * simul_density), air_pollution, active_modes, noise, car_accidents

def compute_total_welfare_all_welfare_increasing(income, prix_transport, simul_rent, simul_size, BETA, simul_density, mode_choice, distance_cbd, WALKING_SPEED, Country, path_folder, year, Region, FUEL_EFFICIENCY_DECREASE, BASELINE_EFFICIENCY_DECREASE, LIFESPAN, imaclim = 0, policy_fe = 0, cobenefits = False):
    if cobenefits == True:
        air_pollution = compute_air_pollution_all_welfare_increasing(simul_density, mode_choice, distance_cbd, Country, path_folder, year, Region, imaclim, policy_fe, FUEL_EFFICIENCY_DECREASE, BASELINE_EFFICIENCY_DECREASE, LIFESPAN)
        active_modes = compute_active_modes(simul_density, mode_choice, distance_cbd, WALKING_SPEED, Country, path_folder, Region, year, imaclim)
        noise = compute_noise(simul_density, mode_choice, distance_cbd, Country, Region, year, imaclim, path_folder)
        car_accidents = compute_car_accidents(simul_density, mode_choice, distance_cbd, Country, Region, year, imaclim, path_folder)
    elif cobenefits == False:
        air_pollution = 0
        active_modes = 0
        noise = 0
        car_accidents = 0
    monetary_value_cobenefits = (active_modes - air_pollution - noise - car_accidents) / np.nansum(simul_density)
    vec_utility = (((income - prix_transport - (simul_rent * simul_size) + monetary_value_cobenefits) ** (1 - BETA)) * ((simul_size) ** (BETA)))
    return np.nansum(vec_utility * simul_density)

def compute_total_welfare2_all_welfare_increasing(income, prix_transport, simul_rent, simul_size, BETA, simul_density, mode_choice, distance_cbd, WALKING_SPEED, Country, path_folder, year, Region, FUEL_EFFICIENCY_DECREASE, BASELINE_EFFICIENCY_DECREASE, LIFESPAN, imaclim = 0, policy_fe = 0, cobenefits = False):
    if cobenefits == True:
        air_pollution = compute_air_pollution_all_welfare_increasing(simul_density, mode_choice, distance_cbd, Country, path_folder, year, Region, imaclim, policy_fe, FUEL_EFFICIENCY_DECREASE, BASELINE_EFFICIENCY_DECREASE, LIFESPAN)
        active_modes = compute_active_modes(simul_density, mode_choice, distance_cbd, WALKING_SPEED, Country, path_folder, Region, year, imaclim)
        noise = compute_noise(simul_density, mode_choice, distance_cbd, Country, Region, year, imaclim, path_folder)
        car_accidents = compute_car_accidents(simul_density, mode_choice, distance_cbd, Country, Region, year, imaclim, path_folder)
    elif cobenefits == False:
        air_pollution = 0
        active_modes = 0
        noise = 0
        car_accidents = 0
    monetary_value_cobenefits = (active_modes - air_pollution - noise - car_accidents) / np.nansum(simul_density)
    vec_utility = (((income - prix_transport - (simul_rent * simul_size) + monetary_value_cobenefits) ** (1 - BETA)) * ((simul_size) ** (BETA)))
    return np.nansum(vec_utility * simul_density), air_pollution, active_modes, noise, car_accidents


def init_outputs(DURATION, distance_cbd):
    save_density = np.zeros((DURATION + 1, len(distance_cbd)))
    save_rent = np.zeros((DURATION + 1, len(distance_cbd)))
    save_dwelling_size = np.zeros((DURATION + 1, len(distance_cbd)))
    save_population = np.zeros(DURATION + 1)
    save_population2 = np.zeros(DURATION + 1)
    save_income = np.zeros(DURATION + 1)
    save_R0 = np.zeros(DURATION + 1)
    save_emissions = np.zeros(DURATION + 1)
    save_emissions_per_capita = np.zeros(DURATION + 1)
    save_avg_utility = np.zeros(DURATION + 1)
    save_total_welfare = np.zeros(DURATION + 1)
    save_total_welfare_with_cobenefits = np.zeros(DURATION + 1)
    save_urbanized_area = np.zeros(DURATION + 1)
    save_distance = np.zeros(len(distance_cbd))
    save_modal_shares = np.zeros((DURATION + 1, len(distance_cbd)))
    return save_density, save_rent, save_dwelling_size, save_population, save_population2, save_income, save_R0, save_emissions, save_emissions_per_capita, save_avg_utility, save_total_welfare, save_total_welfare_with_cobenefits, save_urbanized_area, save_distance, save_modal_shares

def plot_emissions_and_welfare(save_emissions_per_capita, save_total_welfare, save_total_welfare_with_cobenefits, path_outputs, city):
    plt.plot(save_emissions_per_capita)
    plt.xlabel("Year")
    plt.ylabel("Emissions per capita")
    plt.savefig(path_outputs + city + "_emissions_per_capita.png")
    plt.close()

    plt.plot(save_total_welfare)
    plt.xlabel("Year")
    plt.ylabel("Total welfare")
    plt.savefig(path_outputs + city + "_total_welfare.png")
    plt.close()
    
    plt.plot(save_total_welfare_with_cobenefits)
    plt.xlabel("Year")
    plt.ylabel("Total welfare")
    plt.savefig(path_outputs + city + "_total_welfare_with_cobenefits.png")
    plt.close()

def save_outputs(save_emissions, save_emissions_per_capita, save_population, save_R0, save_rent, save_dwelling_size, save_density, save_avg_utility, save_total_welfare, save_total_welfare_with_cobenefits, save_income, save_urbanized_area, save_distance, save_modal_shares, path_outputs, city):
    np.save(path_outputs + city + "_emissions.npy", save_emissions)
    np.save(path_outputs + city + "_emissions_per_capita.npy", save_emissions_per_capita)
    np.save(path_outputs + city + "_population.npy", save_population)
    np.save(path_outputs + city + "_R0.npy", save_R0)
    np.save(path_outputs + city + "_rent.npy", save_rent)
    np.save(path_outputs + city + "_dwelling_size.npy", save_dwelling_size)
    np.save(path_outputs + city + "_density.npy", save_density)
    np.save(path_outputs + city + "_avg_utility.npy", save_avg_utility)
    np.save(path_outputs + city + "_total_welfare.npy", save_total_welfare)
    np.save(path_outputs + city + "_total_welfare_with_cobenefits.npy", save_total_welfare_with_cobenefits)
    np.save(path_outputs + city + "_income.npy", save_income)
    np.save(path_outputs + city + "_urbanized_area.npy", save_urbanized_area)
    np.save(path_outputs + city + "_distance.npy", save_distance)
    np.save(path_outputs + city + "_modal_shares.npy", save_modal_shares)