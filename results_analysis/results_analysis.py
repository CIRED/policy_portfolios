# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 16:32:56 2022

@author: charl
"""

# Import packages
import pandas as pd
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import math
import os
import scipy.stats
from statsmodels.formula.api import ols
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import export_graphviz 
from sklearn.tree import DecisionTreeClassifier 
from sklearn import tree
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from inputs.parameters import *
from inputs.transport import *
from inputs.data import *
from inputs.land_use import *
from calibration.calibration import *
from calibration.validation import *
from model.model import *
from outputs.outputs import *

# Define path
path_data = "C:/Users/charl/OneDrive/Bureau/City_dataStudy/"
path_folder = "C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Data/"
path_calibration = "C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/final_results/calibration_20211124/"
path_BAU = "C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/BAU_20221218/" #BAU_20211124
path_carbon_tax = 'C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/CT_20221218/' #CT_20220907
path_fuel_efficiency = 'C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/fuel_efficiency_20221218/' #all_welfare_increasing_20220907
path_UGB = 'C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/UGB_20221218/' #all_20220907
path_BRT = 'C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/BRT_20221218/' #BRT_20220907
path_welfare_increasing = 'C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/all_welfare_increasing_20221218/'
path_all = 'C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/all_20221218/'

# Work on the welfare-increasing scenario as well?
welfare_increasing = True

# Import list of cities
list_city = list_of_cities_and_databases(path_data,'cityDatabase')

### STEP 1: SAMPLE SELECTION (SUPP SECTION C)

sample_of_cities = pd.DataFrame(columns = ['City', 'criterion1', 'criterion2', 'criterion3', 'final_sample'], index = np.unique(list_city.City))
sample_of_cities.City = sample_of_cities.index

# Exclusion criterion 1: real estate data availability
selected_cells = np.load(path_calibration + "d_selected_cells.npy", allow_pickle = True)
selected_cells = np.array(selected_cells, ndmin = 1)[0]

for city in list(np.delete(sample_of_cities.index, 153)):
    if (selected_cells[city] > 1):
        sample_of_cities.loc[city, "criterion1"] = 1
    elif (selected_cells[city] == 1):
        sample_of_cities.loc[city, "criterion1"] = 0
        
print("Number of cities excluded because of criterion 1:", sum(sample_of_cities.criterion1 == 0))

sample_of_cities.to_excel("C:/Users/charl/OneDrive/Bureau/sample_cities.xlsx")

# Exclusion criterion 2: real estate data consistent with income data
def weighted_percentile(data, percents, weights=None):
    ''' percents in units of 1%
        weights specifies the frequency (count) of data.
    '''
    if weights is None:
        return np.percentile(data, percents)
    ind=np.argsort(data)
    d=data[ind]
    w=weights[ind]
    p=1.*w.cumsum()/w.sum()*100
    y=np.interp(percents, p, d)
    return y

for city in list(sample_of_cities.index):
    (country, proj, density, rents_and_size, land_use, land_cover_ESACCI, driving, transit, grille, 
     centre, distance_cbd, conversion_rate) = import_data(list_city, path_data, city, path_folder)        
    density = density.loc[:,density.columns.str.startswith("density")],
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
    
    income = pd.read_excel(path_folder + "income.xlsx").income[pd.read_excel(path_folder + "income.xlsx").city == city].squeeze()   
    share_housing = rent * size / income
    if weighted_percentile(np.array(share_housing)[~np.isnan(density) & ~np.isnan(share_housing)], 80, weights=density[~np.isnan(density) & ~np.isnan(share_housing)]) > 1:
        sample_of_cities.loc[city, ['criterion2']] = 0
    else:
        sample_of_cities.loc[city, ['criterion2']] = 1

print("Number of cities excluded because of criterion 2:", sum(sample_of_cities.criterion2 == 0))

# Exclusion criterion 3: transportation data quality
sample_of_cities["criterion4"] = 1

for city in sample_of_cities.index:
    if city in ['Basel', 'Bern', 'Bilbao', 'Chicago', 'Cordoba', 'Cracow', 'Curitiba', 'Glasgow', 'Izmir', 'Johannesburg', 'Leeds', 'Liverpool', 'Malmo', 'Monterrey', 'Munich', 'New_York', 'Nottingham', 'Nuremberg', 'Salvador', 'San_Fransisco', 'Seattle', 'Singapore', 'Sofia', 'Stockholm', 'Valencia', 'Zurich']:
        sample_of_cities.loc[sample_of_cities.City == city, "criterion4"] = 0

print("Number of cities excluded because of criterion 3:", sum(sample_of_cities.criterion4 == 0))


# Exclusion criterion 4: reasonable fit of the model
d_corr_density_scells2 = np.load(path_calibration + "d_corr_density_scells2.npy", allow_pickle = True)
d_corr_density_scells2 = np.array(d_corr_density_scells2, ndmin = 1)[0]
d_corr_rent_scells2 = np.load(path_calibration + "d_corr_rent_scells2.npy", allow_pickle = True)
d_corr_rent_scells2 = np.array(d_corr_rent_scells2, ndmin = 1)[0]

for city in list(sample_of_cities.index[sample_of_cities.criterion1 == 1 ]):
    if (d_corr_density_scells2[city][0] < 0) | (d_corr_rent_scells2[city][0] < 0):
        sample_of_cities.loc[city, "criterion3"] = 0
    else:
        sample_of_cities.loc[city, "criterion3"] = 1
        
print("Number of cities excluded because of criterion 4:", sum((sample_of_cities.criterion3 == 0) & (sample_of_cities.criterion2 == 1)& (sample_of_cities.criterion4 == 1)))

# Outcome of the sample selection
sample_of_cities.loc[(sample_of_cities.criterion1 == 0) | (sample_of_cities.criterion2 == 0) | (sample_of_cities.criterion3 == 0) | (sample_of_cities.criterion4 == 0), "final_sample"] = 0
sample_of_cities.loc[(sample_of_cities.criterion1 == 1) & (sample_of_cities.criterion2 == 1) & (sample_of_cities.criterion3 == 1) & (sample_of_cities.criterion4 == 1), "final_sample"] = 1

print("Number of cities kept in the end:", sum(sample_of_cities.final_sample == 1))

### STEP 2: VALIDATION (SUPP SECTION A7)

# Validation: calibrated parameters
calibrated_parameters = pd.DataFrame(columns = ['City', 'beta', 'b', 'kappa', 'Ro', 'sample_cities'], index = sample_of_cities.City[sample_of_cities.final_sample == 1])
calibrated_parameters.City = calibrated_parameters.index

beta = np.load(path_calibration + "beta.npy", allow_pickle = True)
beta = np.array(beta, ndmin = 1)[0]
b = np.load(path_calibration + "b.npy", allow_pickle = True)
b = np.array(b, ndmin = 1)[0]
kappa = np.load(path_calibration + "kappa.npy", allow_pickle = True)
kappa = np.array(kappa, ndmin = 1)[0]
Ro = np.load(path_calibration + "Ro.npy", allow_pickle = True)
Ro = np.array(Ro, ndmin = 1)[0]

for city in calibrated_parameters.index:
    calibrated_parameters.loc[calibrated_parameters.City == city, 'beta'] = beta[city]
    calibrated_parameters.loc[calibrated_parameters.City == city, 'b'] = b[city]
    calibrated_parameters.loc[calibrated_parameters.City == city, 'kappa'] = kappa[city]
    calibrated_parameters.loc[calibrated_parameters.City == city, 'Ro'] = Ro[city]

for param in ['beta', 'b', 'kappa', 'Ro']:
    plt.rcParams['figure.dpi'] = 360
    sns.set(style="whitegrid")
    fig, ax = plt.subplots(figsize=(6,4))
    sns.histplot(calibrated_parameters[param], color=['#fc8d62'])#("Set2"))
    plt.xlabel('')
    plt.ylabel('Number of cities', size=14)
    ax.tick_params(axis = 'both', labelsize=14, color='#4f4e4e')

# Validation: Modal shares
modal_shares_data = pd.DataFrame(columns = ['City', 'Simul_car', 'Simul_walking', 'Simul_transit'], index = list(sample_of_cities.City[(sample_of_cities.final_sample == 1)]))# | (sample_of_cities.criterion4 == 0)]))
modal_shares_data.City = modal_shares_data.index

for city in modal_shares_data.index:
    density = np.load(path_BAU + city + "_density.npy")[0]
    modal_share = np.load(path_BAU + city + "_modal_shares.npy")[0]
    modal_shares_data.Simul_car[modal_shares_data.City == city] = np.nansum(density * (modal_share == 0)) / np.nansum(density)
    modal_shares_data.Simul_transit[modal_shares_data.City == city] = np.nansum(density * (modal_share == 1)) / np.nansum(density)
    modal_shares_data.Simul_walking[modal_shares_data.City == city] = np.nansum(density * (modal_share == 2)) / np.nansum(density)

epomm = pd.read_excel(path_folder + 'modal_shares_data.xlsx')
epomm = epomm.loc[:, ['city', 'walk', 'bike', 'public_transport', 'car']]
epomm.columns = ['City', 'Epomm_walking', 'Epomm_cycling', 'Epomm_transit', 'Epomm_car']

c40 = pd.read_excel(path_folder + 'modal_shares_c40.xlsx', header = 1)
c40.columns = ['City', 'c40_car', 'c40_transit', 'c40_walking']

deloitte = pd.read_excel(path_folder + 'modal_shares_c40.xlsx', header = 1, sheet_name = 'Deloitte')
deloitte.columns = ['City', 'deloitte_car', 'deloitte_transit', 'deloitte_walking']

modal_shares_data = modal_shares_data.merge(epomm, on = 'City', how = 'left')
modal_shares_data = modal_shares_data.merge(c40, on = 'City', how = 'left')
modal_shares_data = modal_shares_data.merge(deloitte, on = 'City', how = 'left')

modal_shares_data["Total_car"] = modal_shares_data["deloitte_car"]
modal_shares_data["Total_car"][np.isnan(modal_shares_data["Total_car"])] = modal_shares_data["c40_car"]
modal_shares_data["Total_car"][np.isnan(modal_shares_data["Total_car"])] = modal_shares_data["Epomm_car"]

modal_shares_data["Total_transit"] = modal_shares_data["deloitte_transit"]
modal_shares_data["Total_transit"][np.isnan(modal_shares_data["Total_transit"])] = modal_shares_data["c40_transit"]
modal_shares_data["Total_transit"][np.isnan(modal_shares_data["Total_transit"])] = modal_shares_data["Epomm_transit"]

modal_shares_data["Total_walking"] = modal_shares_data["deloitte_walking"]
modal_shares_data["Total_walking"][np.isnan(modal_shares_data["Total_walking"])] = modal_shares_data["c40_walking"]
modal_shares_data["Total_walking"][np.isnan(modal_shares_data["Total_walking"])] = modal_shares_data["Epomm_walking"] + modal_shares_data["Epomm_cycling"]

# Private cars - comparison with EPOMM data
plt.figure(figsize = (15, 10))
plt.rcParams.update({'font.size': 20})
plt.scatter(modal_shares_data["Epomm_car"], modal_shares_data.Simul_car * 100, s = 200)
plt.xlabel("Modal share of private car (EPOMM data - %)", size = 20)
plt.ylabel("Modal share of private car (Simulations - %)", size = 20)
plt.xlim(0, 100)
plt.ylim(0, 100)
for i in np.arange(len(modal_shares_data.index)):
    plt.annotate(np.array(modal_shares_data.City)[i], (np.array(modal_shares_data["Epomm_car"])[i], np.array(modal_shares_data.Simul_car)[i] * 100), size = 20)
print(sc.stats.pearsonr(np.array(modal_shares_data["Epomm_car"].astype(float)[~np.isnan(modal_shares_data["Epomm_car"].astype(float))]), np.array(100 * modal_shares_data.Simul_car)[~np.isnan(modal_shares_data["Epomm_car"].astype(float))]))
print(sum(~np.isnan((modal_shares_data["Epomm_car"]).astype(float))))

# Private cars - comparison with C40 data
plt.figure(figsize = (15, 10))
plt.rcParams.update({'font.size': 20})
plt.scatter(modal_shares_data["c40_car"], modal_shares_data.Simul_car * 100, s = 200)
plt.xlabel("Modal share of private car (C40 data - %)", size = 20)
plt.ylabel("Modal share of private car (Simulations - %)", size = 20)
plt.xlim(0, 100)
plt.ylim(0, 100)
print(sc.stats.pearsonr(np.array(modal_shares_data["c40_car"].astype(float)[~np.isnan(modal_shares_data["c40_car"].astype(float))]), np.array(100 * modal_shares_data.Simul_car)[~np.isnan(modal_shares_data["c40_car"].astype(float))]))
print(sum(~np.isnan((modal_shares_data["c40_car"]).astype(float))))

# Private cars - comparison with Deloitte data
plt.figure(figsize = (15, 10))
plt.rcParams.update({'font.size': 20})
plt.scatter(modal_shares_data["deloitte_car"], modal_shares_data.Simul_car * 100, s = 200)
plt.xlabel("Modal share of private car (Deloitte data - %)", size = 20)
plt.ylabel("Modal share of private car (Simulations - %)", size = 20)
plt.xlim(0, 100)
plt.ylim(0, 100)
print(sc.stats.pearsonr(np.array(modal_shares_data["deloitte_car"].astype(float)[~np.isnan(modal_shares_data["deloitte_car"].astype(float))]), np.array(100 * modal_shares_data.Simul_car)[~np.isnan(modal_shares_data["deloitte_car"].astype(float))]))
print(sum(~np.isnan((modal_shares_data["deloitte_car"]).astype(float))))

# Private cars - comparison with all data
plt.figure(figsize = (15, 10))
plt.rcParams.update({'font.size': 20})
plt.scatter(modal_shares_data["Total_car"], modal_shares_data.Simul_car * 100, s = 200)
plt.xlabel("Modal share of private car (Data - %)", size = 20)
plt.ylabel("Modal share of private car (Simulations - %)", size = 20)
plt.plot([0,100],[0,100])
plt.xlim(0, 100)
plt.ylim(0, 100)
print(sc.stats.pearsonr(np.array(modal_shares_data["Total_car"].astype(float)[~np.isnan(modal_shares_data["Total_car"].astype(float))]), np.array(100 * modal_shares_data.Simul_car)[~np.isnan(modal_shares_data["Total_car"].astype(float))]))
print(sum(~np.isnan((modal_shares_data["Total_car"]).astype(float))))

# Public transport - comparison with EPOMM data
plt.figure(figsize = (15, 10))
plt.rcParams.update({'font.size': 20})
plt.scatter(modal_shares_data["Epomm_transit"], modal_shares_data.Simul_transit * 100, s = 200)
plt.xlabel("Modal share of public transport (EPOMM data - %)", size = 20)
plt.ylabel("Modal share of public transport (Simulations - %)", size = 20)
plt.xlim(0, 100)
plt.ylim(0, 100)
for i in np.arange(len(modal_shares_data.index)):
    plt.annotate(np.array(modal_shares_data.City)[i], (np.array(modal_shares_data["Epomm_transit"])[i], np.array(modal_shares_data.Simul_transit)[i] * 100), size = 20)
print(sc.stats.pearsonr(np.array(modal_shares_data["Epomm_transit"].astype(float)[~np.isnan(modal_shares_data["Epomm_transit"].astype(float))]), np.array(100 * modal_shares_data.Simul_transit)[~np.isnan(modal_shares_data["Epomm_transit"].astype(float))]))
print(sum(~np.isnan((modal_shares_data["Epomm_transit"]).astype(float))))

# Public transport - comparison with C40 data
plt.figure(figsize = (15, 10))
plt.rcParams.update({'font.size': 20})
plt.scatter(modal_shares_data["c40_transit"], modal_shares_data.Simul_transit * 100, s = 200)
plt.xlabel("Modal share of public transport (c40 data - %)", size = 20)
plt.ylabel("Modal share of public transport (Simulations - %)", size = 20)
plt.xlim(0, 100)
plt.ylim(0, 100)
print(sc.stats.pearsonr(np.array(modal_shares_data["c40_transit"].astype(float)[~np.isnan(modal_shares_data["c40_transit"].astype(float))]), np.array(100 * modal_shares_data.Simul_transit)[~np.isnan(modal_shares_data["c40_transit"].astype(float))]))
print(sum(~np.isnan((modal_shares_data["c40_transit"]).astype(float))))

# Public transport - comparison with Deloitte data
plt.figure(figsize = (15, 10))
plt.rcParams.update({'font.size': 20})
plt.scatter(modal_shares_data["deloitte_transit"], modal_shares_data.Simul_transit * 100, s = 200)
plt.xlabel("Modal share of public transport (Deloitte data - %)", size = 20)
plt.ylabel("Modal share of public transport (Simulations - %)", size = 20)
plt.xlim(0, 100)
plt.ylim(0, 100)
print(sc.stats.pearsonr(np.array(modal_shares_data["deloitte_transit"].astype(float)[~np.isnan(modal_shares_data["deloitte_transit"].astype(float))]), np.array(100 * modal_shares_data.Simul_transit)[~np.isnan(modal_shares_data["deloitte_transit"].astype(float))]))
print(sum(~np.isnan((modal_shares_data["deloitte_transit"]).astype(float))))

# Public transport - comparison with all data
plt.figure(figsize = (15, 10))
plt.rcParams.update({'font.size': 20})
plt.scatter(modal_shares_data["Total_transit"], modal_shares_data.Simul_transit * 100, s = 200)
plt.xlabel("Modal share of public transport (Data - %)", size = 20)
plt.ylabel("Modal share of public transport (Simulations - %)", size = 20)
plt.xlim(0, 100)
plt.ylim(0, 100)
print(sc.stats.pearsonr(np.array(modal_shares_data["Total_transit"].astype(float)[~np.isnan(modal_shares_data["Total_transit"].astype(float))]), np.array(100 * modal_shares_data.Simul_transit)[~np.isnan(modal_shares_data["Total_transit"].astype(float))]))
print(sum(~np.isnan((modal_shares_data["Total_transit"]).astype(float))))

# Active modes - comparison with EPOMM data
plt.figure(figsize = (15, 10))
plt.rcParams.update({'font.size': 20})
plt.scatter(modal_shares_data["Epomm_cycling"] + modal_shares_data["Epomm_walking"], modal_shares_data.Simul_walking * 100, s = 200)
plt.xlabel("Modal share of active modes (EPOMM data - %)", size = 20)
plt.ylabel("Modal share of active modes (Simulations - %)", size = 20)
plt.xlim(0, 100)
plt.ylim(0, 100)
for i in np.arange(len(modal_shares_data.index)):
    plt.annotate(np.array(modal_shares_data.City)[i], (np.array(modal_shares_data["Epomm_cycling"] + modal_shares_data["Epomm_walking"])[i], np.array(modal_shares_data.Simul_walking)[i] * 100), size = 20)
print(sc.stats.pearsonr(np.array((modal_shares_data["Epomm_cycling"] + modal_shares_data["Epomm_walking"]).astype(float)[~np.isnan((modal_shares_data["Epomm_cycling"] + modal_shares_data["Epomm_walking"]).astype(float))]), np.array(100 * modal_shares_data.Simul_walking)[~np.isnan((modal_shares_data["Epomm_cycling"] + modal_shares_data["Epomm_walking"]).astype(float))]))
print(sum(~np.isnan((modal_shares_data["Epomm_walking"]).astype(float))))

# Active modes - comparison with C40 data
plt.figure(figsize = (15, 10))
plt.rcParams.update({'font.size': 20})
plt.scatter(modal_shares_data["c40_walking"], modal_shares_data.Simul_walking * 100, s = 200)
plt.xlabel("Modal share of active modes (C40 data - %)", size = 20)
plt.ylabel("Modal share of active modes (Simulations - %)", size = 20)
plt.xlim(0, 100)
plt.ylim(0, 100)
print(sc.stats.pearsonr(np.array(modal_shares_data["c40_walking"].astype(float)[~np.isnan((modal_shares_data["c40_walking"]).astype(float))]), np.array(100 * modal_shares_data.Simul_walking)[~np.isnan((modal_shares_data["c40_walking"]).astype(float))]))
print(sum(~np.isnan((modal_shares_data["c40_walking"]).astype(float))))

# Active modes - comparison with dELOITTE data
plt.figure(figsize = (15, 10))
plt.rcParams.update({'font.size': 20})
plt.scatter(modal_shares_data["deloitte_walking"], modal_shares_data.Simul_walking * 100, s = 200)
plt.xlabel("Modal share of active modes (Deloitte data - %)", size = 20)
plt.ylabel("Modal share of active modes (Simulations - %)", size = 20)
plt.xlim(0, 100)
plt.ylim(0, 100)
print(sc.stats.pearsonr(np.array(modal_shares_data["deloitte_walking"].astype(float)[~np.isnan((modal_shares_data["deloitte_walking"]).astype(float))]), np.array(100 * modal_shares_data.Simul_walking)[~np.isnan((modal_shares_data["deloitte_walking"]).astype(float))]))
print(sum(~np.isnan((modal_shares_data["deloitte_walking"]).astype(float))))

# Active modes - comparison with all data
plt.figure(figsize = (15, 10))
plt.rcParams.update({'font.size': 20})
plt.scatter(modal_shares_data["Total_walking"], modal_shares_data.Simul_walking * 100, s = 200)
plt.xlabel("Modal share of active modes (Data - %)", size = 20)
plt.ylabel("Modal share of active modes (Simulations - %)", size = 20)
plt.xlim(0, 100)
plt.ylim(0, 100)
print(sc.stats.pearsonr(np.array(modal_shares_data["Total_walking"].astype(float)[~np.isnan((modal_shares_data["Total_walking"]).astype(float))]), np.array(100 * modal_shares_data.Simul_walking)[~np.isnan((modal_shares_data["Total_walking"]).astype(float))]))
print(sum(~np.isnan((modal_shares_data["Total_walking"]).astype(float))))

modal_shares_data = modal_shares_data.merge(sample_of_cities.loc[:, ['City']])

print(modal_shares_data.City[(modal_shares_data["Total_transit"].astype(float) > 30) & (100 * modal_shares_data.Simul_transit < 15)])
print(modal_shares_data.City[(modal_shares_data["Total_transit"].astype(float) > 20) & (100 * modal_shares_data.Simul_transit < 10)])

# Validation: emissions
df_emissions = pd.DataFrame(columns = ['City', 'Simul_emissions'], index = list(sample_of_cities.City[sample_of_cities.final_sample == 1]))
df_emissions.City = df_emissions.index

for city in df_emissions.index:
    df_emissions.Simul_emissions[df_emissions.City == city] = np.load(path_BAU + city + "_emissions_per_capita.npy")[0]

# Data from Nangini et al.
felix_data = pd.read_excel(path_folder + "emissions_databases/datapaper felix/DATA/D_FINAL.xlsx", header = 0)
felix_data_for_comparison = felix_data.loc[:, ["City name", "Scope-1 GHG emissions [tCO2 or tCO2-eq]", "Scope-2 (CDP) [tCO2-eq]", "Population (CDP)"]]
felix_data_for_comparison["City name"][felix_data_for_comparison["City name"] == "Addis Ababa"] = "Addis_Ababa"
felix_data_for_comparison["City name"][felix_data_for_comparison["City name"] == "Belo Horizonte"] = "Belo_Horizonte"
felix_data_for_comparison["City name"][felix_data_for_comparison["City name"] == "Bogotá"] = "Bogota"
felix_data_for_comparison["City name"][felix_data_for_comparison["City name"] == "Brasília"] = "Brasilia"
felix_data_for_comparison["City name"][felix_data_for_comparison["City name"] == "Buenos Aires"] = "Buenos_Aires"
felix_data_for_comparison["City name"][felix_data_for_comparison["City name"] == "Cape Town"] = "Cape_Town"
felix_data_for_comparison["City name"][felix_data_for_comparison["City name"] == "Goiânia"] = "Goiania"
felix_data_for_comparison["City name"][felix_data_for_comparison["City name"] == "Greater London"] = "London"
felix_data_for_comparison["City name"][felix_data_for_comparison["City name"] == "Hong Kong"] = "Hong_Kong"
felix_data_for_comparison["City name"][felix_data_for_comparison["City name"] == "Jinan, Shandong"] = "Jinan"
felix_data_for_comparison["City name"][felix_data_for_comparison["City name"] == "Los Angeles"] = "Los_Angeles"
felix_data_for_comparison["City name"][felix_data_for_comparison["City name"] == "Mexico City"] = "Mexico_City"
felix_data_for_comparison["City name"][felix_data_for_comparison["City name"] == "New York City"] = "New_York"
felix_data_for_comparison["City name"][felix_data_for_comparison["City name"] == "Porto Alegre"] = "Porto_Alegre"
felix_data_for_comparison["City name"][felix_data_for_comparison["City name"] == "Rio de Janeiro"] = "Rio_de_Janeiro"
felix_data_for_comparison["City name"][felix_data_for_comparison["City name"] == "San Francisco"] = "San_Fransisco"
felix_data_for_comparison["City name"][felix_data_for_comparison["City name"] == "San Diego"] = "San_Diego"
felix_data_for_comparison["City name"][felix_data_for_comparison["City name"] == "Sao Paulo"] = "Sao_Paulo"
felix_data_for_comparison["City name"][felix_data_for_comparison["City name"] == "Toluca de Lerdo"] = "Toluca"
felix_data_for_comparison["City name"][felix_data_for_comparison["City name"] == "Zürich"] = "Zurich"
felix_data_for_comparison["felix_scope1"] = felix_data_for_comparison["Scope-1 GHG emissions [tCO2 or tCO2-eq]"] / felix_data_for_comparison["Population (CDP)"]
felix_data_for_comparison["felix_scope2"] = felix_data_for_comparison["Scope-2 (CDP) [tCO2-eq]"] / felix_data_for_comparison["Population (CDP)"]
felix_data_for_comparison = felix_data_for_comparison.loc[:, ['City name', "felix_scope1", "felix_scope2"]]

# Data from Moran et al.
erl_data = pd.read_excel(path_folder + "emissions_databases/ERL/ERL_13_064041_SD_Moran Spatial Footprints Data Appendix.xlsx", header = 0, sheet_name = 'S.2.3a - Top 500 Cities')
erl_data = erl_data.iloc[:, [0, 2]]
erl_data.columns = ['city', 'erl_emissions']
erl_data.city[erl_data.city == 'Ahmadabad'] = 'Ahmedabad'
erl_data.city[erl_data.city == 'Antwerpen'] = 'Antwerp'
erl_data.city[erl_data.city == 'Belo_Horizonte'] = 'Belo_Horizonte'
erl_data.city[erl_data.city == 'Buenos Aires'] = 'Buenos_Aires'
erl_data.city[erl_data.city == 'Cape Town'] = 'Cape_Town'
erl_data.city[erl_data.city == 'Esfahan'] = 'Isfahan'
erl_data.city[erl_data.city == 'Frankfurt'] = 'Frankfurt_am_Main'
erl_data.city[erl_data.city == 'Hannover'] = 'Hanover'
erl_data.city[erl_data.city == 'Hong Kong'] = 'Hong_Kong'
erl_data.city[erl_data.city == 'Kazan\''] = 'Kazan'
erl_data.city[erl_data.city == 'Los Angeles'] = 'Los_Angeles'
erl_data.city[erl_data.city == 'Mexico City'] = 'Mexico_City'
erl_data.city[erl_data.city == 'New Delhi'] = 'Delhi'
erl_data.city[erl_data.city == 'New York'] = 'New_York'
erl_data.city[erl_data.city == 'Porto Alegre'] = 'Porto_Alegre'
erl_data.city[erl_data.city == 'Rio de Janeiro'] = 'Rio_de_Janeiro'
erl_data.city[erl_data.city == 'Rostov-on-Don'] = 'Rostov_on_Don'
erl_data.city[erl_data.city == 'Saint Petersburg'] = 'St_Petersburg'
erl_data.city[erl_data.city == 'San Diego'] = 'San_Diego'
erl_data.city[erl_data.city == 'Sao Paulo'] = 'Sao_Paulo'
erl_data.city[erl_data.city == 'Washington D.C.'] = 'Washington_DC'
erl_data.erl_emissions[erl_data.city == 'Valencia'] = 5.7
erl_data2 = pd.read_excel("C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Data/emissions_databases/ERL/ERL_13_064041_SD_Moran Spatial Footprints Data Appendix.xlsx", header = 0, sheet_name = 'S.2.3b - Top 500 Per Capita')
erl_data2 = erl_data2.iloc[:, [0, 5]]
erl_data2.columns = ['city', 'erl_emissions2']

# Data from Kona et al.
com_label = pd.read_excel(path_folder + "emissions_databases/CoM/GlobalCoM.xlsx", header = 0, sheet_name = 'Table 3')
com_label = com_label.iloc[:, 0:2]
com_label.columns = ['GCoM_ID', 'city']
com_label.city[com_label.city == 'Grand Lyon'] = 'Lyon'
com_label.city[com_label.city == 'Lisboa'] = 'Lisbon'
com_label.city[com_label.city == 'Malmö'] = 'Malmo'
com_label.city[com_label.city == 'NANTES METROPOLE'] = 'Nantes'
com_label.city[com_label.city == 'München'] = 'Munich'
com_label.city[com_label.city == 'Málaga'] = 'Malaga'
com_label.city[com_label.city == 'Genève'] = 'Geneva'
com_label.city[com_label.city == 'Genova'] = 'Genoa'
com_label.city[com_label.city == 'Zürich'] = 'Zurich'
com_label.city[com_label.city == 'BORDEAUX METROPOLE'] = 'Bordeaux'
com_label.city[com_label.city == 'Nice Côte d\'Azur'] = 'Nice'
com_label.city[com_label.city == 'Gent'] = 'Ghent'
com_label.city[com_label.city == 'Izmir Metropolitan Municipality'] = 'Izmir'
com_label.city[com_label.city == 'Bremen'] = 'X'
com_label.columns = ['GCoM_ID', 'city']
com_data = pd.read_excel(path_folder + "emissions_databases/CoM/GlobalCoM.xlsx", header = 0, sheet_name = 'Table 2')
com_data["com_emissions_per_capita"] = com_data.emissions / com_data.population_in_the_inventory_year
com_data = com_data.merge(com_label, on = 'GCoM_ID', how = 'left')
com_data = com_data[com_data.emission_inventory_sector == 'Transportation']
com_data = com_data[com_data.type_of_emission_inventory == 'baseline_emission_inventory']
direct_emissions_data = com_data[com_data.type_of_emissions == 'direct_emissions']
indirect_emissions_data = com_data[com_data.type_of_emissions == 'indirect_emissions']
direct_emissions_data = direct_emissions_data.iloc[:, 12:14]
indirect_emissions_data = indirect_emissions_data.iloc[:, 12:14]
direct_emissions_data.columns = ['com_direct_emissions', 'city']
indirect_emissions_data.columns = ['com_indirect_emissions', 'city']

# Merge our simulations with the data on emissions
df_emissions = df_emissions.merge(felix_data_for_comparison, left_on = 'City', right_on = 'City name', how = 'left')
df_emissions = df_emissions.merge(erl_data, left_on = 'City', right_on = 'city', how = 'left')
df_emissions = df_emissions.merge(erl_data2, left_on = 'City', right_on = 'city', how = 'left')
df_emissions = df_emissions.merge(direct_emissions_data, left_on = 'City', right_on = 'city', how = 'left')
df_emissions = df_emissions.merge(indirect_emissions_data, left_on = 'City', right_on = 'city', how = 'left')
df_emissions.erl_emissions[np.isnan(df_emissions.erl_emissions)] = df_emissions.erl_emissions2[np.isnan(df_emissions.erl_emissions)]
df_emissions.loc[df_emissions.felix_scope1>20, 'felix_scope1'] = np.nan

# Comparispn
for var in ['felix_scope1', 'felix_scope2', 'erl_emissions', 'erl_emissions2', 'com_direct_emissions','com_indirect_emissions']:
    plt.figure(figsize = (15, 10))
    plt.rcParams.update({'font.size': 20})
    plt.scatter(df_emissions['Simul_emissions'], df_emissions[var], s = 200)
    plt.xlabel("Simulated emissions per capita", size = 20)
    plt.ylabel("Data", size = 20)
    print(sc.stats.pearsonr(np.array(df_emissions[var].astype(float)[~np.isnan(df_emissions[var].astype(float))]), np.array(df_emissions.Simul_emissions)[~np.isnan(df_emissions[var].astype(float))]))
    print(len((np.array(df_emissions[var].astype(float)[~np.isnan(df_emissions[var].astype(float))]))))

plt.figure(figsize = (15, 10))
plt.rcParams.update({'font.size': 20})
plt.scatter(df_emissions['Simul_emissions'], df_emissions['felix_scope1'], s = 200)

print(sc.stats.pearsonr(np.array(df_emissions["felix_scope1"].astype(float)[~np.isnan((df_emissions["felix_scope2"]).astype(float)) & ~np.isnan((df_emissions["felix_scope1"]).astype(float))]), np.array(df_emissions.felix_scope2)[~np.isnan((df_emissions["felix_scope2"]).astype(float)) & ~np.isnan((df_emissions["felix_scope1"]).astype(float))]))
print(sc.stats.pearsonr(np.array(df_emissions["Simul_emissions"].astype(float)[~np.isnan((df_emissions["felix_scope2"]).astype(float)) & ~np.isnan((df_emissions["felix_scope2"]).astype(float))]), np.array(df_emissions.felix_scope2)[~np.isnan((df_emissions["felix_scope2"]).astype(float)) & ~np.isnan((df_emissions["felix_scope2"]).astype(float))]))
print(sc.stats.pearsonr(np.array(df_emissions["Simul_emissions"].astype(float)[~np.isnan((df_emissions["felix_scope1"]).astype(float)) & ~np.isnan((df_emissions["felix_scope1"]).astype(float))]), np.array(df_emissions.felix_scope1)[~np.isnan((df_emissions["felix_scope1"]).astype(float)) & ~np.isnan((df_emissions["felix_scope1"]).astype(float))]))

sum(~np.isnan((df_emissions["felix_scope2"]).astype(float)))
sum(~np.isnan((df_emissions["felix_scope1"]).astype(float)))

# Validation: densities, rents, dwelling sizes
r2density_scells2 = np.load(path_calibration + "r2density_scells2.npy", allow_pickle = True)
r2density_scells2 = np.array(r2density_scells2, ndmin = 1)[0]
r2rent_scells2 = np.load(path_calibration + "r2rent_scells2.npy", allow_pickle = True)
r2rent_scells2 = np.array(r2rent_scells2, ndmin = 1)[0]
r2size_scells2 = np.load(path_calibration + "r2size_scells2.npy", allow_pickle = True)
r2size_scells2 = np.array(r2size_scells2, ndmin = 1)[0]

d_corr_density_scells2 = np.load(path_calibration + "d_corr_density_scells2.npy", allow_pickle = True)
d_corr_density_scells2 = np.array(d_corr_density_scells2, ndmin = 1)[0]
d_corr_rent_scells2 = np.load(path_calibration + "d_corr_rent_scells2.npy", allow_pickle = True)
d_corr_rent_scells2 = np.array(d_corr_rent_scells2, ndmin = 1)[0]
d_corr_size_scells2 = np.load(path_calibration + "d_corr_size_scells2.npy", allow_pickle = True)
d_corr_size_scells2 = np.array(d_corr_size_scells2, ndmin = 1)[0]

mae_density_scells2 = np.load(path_calibration + "mae_density_scells2.npy", allow_pickle = True)
mae_density_scells2 = np.array(mae_density_scells2, ndmin = 1)[0]
mae_rent_scells2 = np.load(path_calibration + "mae_rent_scells2.npy", allow_pickle = True)
mae_rent_scells2 = np.array(mae_rent_scells2, ndmin = 1)[0]
mae_size_scells2 = np.load(path_calibration + "mae_size_scells2.npy", allow_pickle = True)
mae_size_scells2 = np.array(mae_size_scells2, ndmin = 1)[0]

rae_density_scells2 = np.load(path_calibration + "rae_density_scells2.npy", allow_pickle = True)
rae_density_scells2 = np.array(rae_density_scells2, ndmin = 1)[0]
rae_rent_scells2 = np.load(path_calibration + "rae_rent_scells2.npy", allow_pickle = True)
rae_rent_scells2 = np.array(rae_rent_scells2, ndmin = 1)[0]
rae_size_scells2 = np.load(path_calibration + "rae_size_scells2.npy", allow_pickle = True)
rae_size_scells2 = np.array(rae_size_scells2, ndmin = 1)[0]

validation = pd.DataFrame()
validation["City"] = np.array(list(r2density_scells2.keys()))
validation["r2density"] = np.array(list(r2density_scells2.values()))
validation["r2rent"] = np.array(list(r2rent_scells2.values()))
validation["r2size"] = np.array(list(r2size_scells2.values()))
validation["corrrdensity"] = np.array(list(d_corr_density_scells2.values()))[:, 0]
validation["corrrent"] = np.array(list(d_corr_rent_scells2.values()))[:, 0]
validation["corrsize"] = np.array(list(d_corr_size_scells2.values()))[:, 0]
validation["mae_density"] = np.array(list(mae_density_scells2.values()))
validation["mae_rent"] = np.array(list(mae_rent_scells2.values()))
validation["mae_size"] = np.array(list(mae_size_scells2.values()))
validation["rae_density"] = np.array(list(rae_density_scells2.values()))
validation["rae_rent"] = np.array(list(rae_rent_scells2.values()))
validation["rae_size"] = np.array(list(rae_size_scells2.values()))
validation = validation.merge(sample_of_cities.loc[:, ['City', 'final_sample']], on = 'City')

validation.loc[validation.City == "Los_Angeles", "corrrdensity"] = 0.50
validation.loc[validation.City == "Athens", "corrrdensity"] = 0.89
validation.loc[validation.City == "Brisbane", "corrrdensity"] = 0.55

validation.loc[validation.City == "Los_Angeles", "corrrent"] = 0.48
validation.loc[validation.City == "Athens", "corrrent"] = 0.21
validation.loc[validation.City == "Brisbane", "corrrent"] = 0.04
              
table_validation = validation[validation.final_sample == 1].describe()
#validation[validation.final_sample == 1].to_excel('C:/Users/charl/OneDrive/Bureau/table_validation.xlsx')

### STEP 3: BAU

BAU_scenario = pd.DataFrame(columns = ['City', 'emissions_2015', 'welfare_without_cobenefits_2015', 'welfare_with_cobenefits_2015', 'welfare_without_cobenefits_2020', 'welfare_with_cobenefits_2020', 'welfare_without_cobenefits_2025', 'welfare_with_cobenefits_2025', 'welfare_without_cobenefits_2030', 'welfare_with_cobenefits_2030', 'emissions_2035', 'welfare_without_cobenefits_2035', 'welfare_with_cobenefits_2035', 'population_2015', 'population_2035', 'avg_utility_2015', 'avg_utility_2035', 'density_2015', 'density_2035'], index = list(sample_of_cities.City[sample_of_cities.final_sample == 1]))
BAU_scenario.City = BAU_scenario.index

for city in BAU_scenario.index:
    BAU_scenario.emissions_2015[BAU_scenario.City == city] = np.load(path_BAU + city + "_emissions_per_capita.npy")[0]
    BAU_scenario.welfare_without_cobenefits_2015[BAU_scenario.City == city] = np.load(path_BAU + city + "_total_welfare.npy")[0]
    BAU_scenario.welfare_with_cobenefits_2015[BAU_scenario.City == city] = np.load(path_BAU + city + "_total_welfare_with_cobenefits.npy")[0]
    BAU_scenario.emissions_2035[BAU_scenario.City == city] = np.load(path_BAU + city + "_emissions_per_capita.npy")[20]
    BAU_scenario.welfare_without_cobenefits_2035[BAU_scenario.City == city] = np.load(path_BAU + city + "_total_welfare.npy")[20]
    BAU_scenario.welfare_with_cobenefits_2035[BAU_scenario.City == city] = np.load(path_BAU + city + "_total_welfare_with_cobenefits.npy")[20]
    BAU_scenario.welfare_without_cobenefits_2030[BAU_scenario.City == city] = np.load(path_BAU + city + "_total_welfare.npy")[15]
    BAU_scenario.welfare_with_cobenefits_2030[BAU_scenario.City == city] = np.load(path_BAU + city + "_total_welfare_with_cobenefits.npy")[15]
    BAU_scenario.welfare_without_cobenefits_2025[BAU_scenario.City == city] = np.load(path_BAU + city + "_total_welfare.npy")[10]
    BAU_scenario.welfare_with_cobenefits_2025[BAU_scenario.City == city] = np.load(path_BAU + city + "_total_welfare_with_cobenefits.npy")[10]
    BAU_scenario.welfare_without_cobenefits_2020[BAU_scenario.City == city] = np.load(path_BAU + city + "_total_welfare.npy")[5]
    BAU_scenario.welfare_with_cobenefits_2020[BAU_scenario.City == city] = np.load(path_BAU + city + "_total_welfare_with_cobenefits.npy")[5]
    BAU_scenario.population_2015[BAU_scenario.City == city] = np.load(path_BAU + city + "_population.npy")[0]
    BAU_scenario.population_2035[BAU_scenario.City == city] = np.load(path_BAU + city + "_population.npy")[20]
    BAU_scenario.avg_utility_2015[BAU_scenario.City == city] = np.load(path_BAU + city + "_avg_utility.npy")[0]
    BAU_scenario.avg_utility_2035[BAU_scenario.City == city] = np.load(path_BAU + city + "_avg_utility.npy")[20]
    
city_continent = pd.read_csv(path_data + "CityDatabases/cityDatabase_221NewCities.csv")
city_continent = city_continent.iloc[:, [0, 2]]
city_continent = city_continent.drop_duplicates(subset = "City")
city_continent = city_continent.sort_values('City')
BAU_scenario = BAU_scenario.merge(city_continent, on = "City", how = 'left')
color_tab = {'North_America':'red', 'Europe':'green', 'Asia':'blue', 'Oceania':'yellow', 'South_America': 'brown', 'Africa': 'orange'}

BAU_scenario["var_emissions"] = 100 *(BAU_scenario["emissions_2035"] - BAU_scenario["emissions_2015"]) / BAU_scenario["emissions_2015"]

plt.rcParams['figure.dpi'] = 360
sns.set(style="whitegrid")
fig, ax = plt.subplots(figsize=(6,4))
sns.histplot(BAU_scenario.var_emissions, color=['#fc8d62'])#("Set2"))
plt.xlabel('Emissions per capita variation (%)', size=14)
plt.ylabel('Number of cities', size=14)
ax.tick_params(axis = 'both', labelsize=14, color='#4f4e4e')
plt.savefig('bau1.png')

plt.rcParams['figure.dpi'] = 360
sns.set(style="whitegrid")
fig, ax = plt.subplots(figsize=(6,4))
sns.stripplot(BAU_scenario.var_emissions, BAU_scenario["Continent"], palette=['#fc8d62']) #, size = np.log(np.log(list(np.array(BAU_scenario.population_2015[BAU_scenario.var_emissions < 100])))))#("Set2"))
plt.xlabel('Emissions per capita variation (%)', size=14)
plt.ylabel('', size=14)
plt.legend([],[], frameon=False)
ax.tick_params(axis = 'both', labelsize=14, color='#4f4e4e')
plt.savefig('bau2.png')

#STEP 4: RESULTS

df = pd.DataFrame(columns = ['City', 'emissions_2035_BAU', 'welfare_with_cobenefits_2035_BAU', 'welfare_without_cobenefits_2035_BAU', 'emissions_2035_BRT', 'welfare_with_cobenefits_2035_BRT', 'welfare_without_cobenefits_2035_BRT', 'emissions_2035_FE', 'welfare_with_cobenefits_2035_FE', 'welfare_without_cobenefits_2035_FE', 'emissions_2035_UGB', 'welfare_with_cobenefits_2035_UGB', 'welfare_without_cobenefits_2035_UGB', 'emissions_2035_CT', 'welfare_with_cobenefits_2035_CT', 'welfare_without_cobenefits_2035_CT',
                             'emissions_2035_all', 'welfare_with_cobenefits_2035_all', 'welfare_without_cobenefits_2035_all', 'emissions_2035_all_welfare_increasing', 'welfare_with_cobenefits_2035_all_welfare_increasing', 'welfare_without_cobenefits_2035_all_welfare_increasing', 'avg_dist_city_center_CT', 'modal_share_cars_CT', 'avg_dist_city_center_FE', 'modal_share_cars_FE', 'avg_dist_city_center_BRT', 'modal_share_cars_BRT', 'avg_dist_city_center_UGB', 'modal_share_cars_UGB', 'avg_dist_city_center_BAU', 'modal_share_cars_BAU', 'avg_dist_city_center_all', 'modal_share_cars_all', 'avg_dist_city_center_all_welfare_increasing', 'modal_share_cars_all_welfare_increasing', 'modal_share_cars_0', 'modal_share_pt_0', 'population_2035',
                             "health_2035_BAU", "health_2035_CT", "health_2035_FE", "health_2035_UGB", "health_2035_BRT", "health_2035_all", "health_2035_all_welfare_increasing",
                             "housing_2035_BAU", "housing_2035_CT", "housing_2035_FE", "housing_2035_BRT", "housing_2035_UGB", "housing_2035_all", "housing_2035_all_welfare_increasing",
                             "rent_2035_BAU", "rent_2035_CT", "rent_2035_FE", "rent_2035_BRT", "rent_2035_UGB", "rent_2035_all", "rent_2035_all_welfare_increasing",
                             "tcost_2035_BAU", "tcost_2035_CT", "tcost_2035_FE", "tcost_2035_BRT", "tcost_2035_UGB", "tcost_2035_all", "tcost_2035_all_welfare_increasing",
                             'active_modes_2035_BAU', 'air_pollution_2035_BAU', 'car_accidents_2035_BAU', 'noise_2035_BAU',
                             'active_modes_2035_CT', 'air_pollution_2035_CT', 'car_accidents_2035_CT', 'noise_2035_CT',
                             'active_modes_2035_FE', 'air_pollution_2035_FE', 'car_accidents_2035_FE', 'noise_2035_FE',
                             'active_modes_2035_UGB', 'air_pollution_2035_UGB', 'car_accidents_2035_UGB', 'noise_2035_UGB',
                             'active_modes_2035_all', 'air_pollution_2035_all', 'car_accidents_2035_all', 'noise_2035_all',
                             'active_modes_2035_all_welfare_increasing', 'air_pollution_2035_all_welfare_increasing', 'car_accidents_2035_all_welfare_increasing', 'noise_2035_all_welfare_increasing',
                             'active_modes_2035_BRT', 'air_pollution_2035_BRT', 'car_accidents_2035_BRT', 'noise_2035_BRT', 'income_2035',
                             'tax_2035_CT', 'tax_2035_all', 'tax_2035_all_welfare_increasing',
                             'cost_2035_BRT', 'cost_2035_all', 'cost_2035_all_welfare_increasing'], index = list(sample_of_cities.City[sample_of_cities.final_sample == 1]))

df.City = df.index
df["city"] = df.index

# Import results for all cities
for city in df.index:
    df.population_2035[df.City == city] = np.load(path_BAU + city + "_population.npy")[20]
    df.emissions_2035_BAU[df.City == city] = np.load(path_BAU + city + "_emissions.npy")[20]
    df.welfare_with_cobenefits_2035_BAU[df.City == city] = np.load(path_BAU + city + "_total_welfare_with_cobenefits.npy")[20]
    df.welfare_without_cobenefits_2035_BAU[df.City == city] = np.load(path_BAU + city + "_total_welfare.npy")[20]
    df.emissions_2035_UGB[df.City == city] = np.load(path_UGB + city + "_emissions.npy")[20]
    df.welfare_with_cobenefits_2035_UGB[df.City == city] = np.load(path_UGB + city + "_total_welfare_with_cobenefits.npy")[20]
    df.welfare_without_cobenefits_2035_UGB[df.City == city] = np.load(path_UGB + city + "_total_welfare.npy")[20]
    df.emissions_2035_BRT[df.City == city] = np.load(path_BRT + city + "_emissions.npy")[20]
    df.welfare_with_cobenefits_2035_BRT[df.City == city] = np.load(path_BRT + city + "_total_welfare_with_cobenefits.npy")[20]
    df.welfare_without_cobenefits_2035_BRT[df.City == city] = np.load(path_BRT + city + "_total_welfare.npy")[20]
    df.emissions_2035_FE[df.City == city] = np.load(path_fuel_efficiency + city + "_emissions.npy")[20]
    df.welfare_with_cobenefits_2035_FE[df.City == city] = np.load(path_fuel_efficiency + city + "_total_welfare_with_cobenefits.npy")[20]
    df.welfare_without_cobenefits_2035_FE[df.City == city] = np.load(path_fuel_efficiency + city + "_total_welfare.npy")[20]
    df.emissions_2035_CT[df.City == city] = np.load(path_carbon_tax + city + "_emissions.npy")[20]
    df.welfare_with_cobenefits_2035_CT[df.City == city] = np.load(path_carbon_tax + city + "_total_welfare_with_cobenefits.npy")[20]  
    df.welfare_without_cobenefits_2035_CT[df.City == city] = np.load(path_carbon_tax + city + "_total_welfare.npy")[20]  
    df.emissions_2035_all[df.City == city] = np.load(path_all + city + "_emissions.npy")[20]
    df.welfare_with_cobenefits_2035_all[df.City == city] = np.load(path_all + city + "_total_welfare_with_cobenefits.npy")[20]  
    df.welfare_without_cobenefits_2035_all[df.City == city] = np.load(path_all + city + "_total_welfare.npy")[20]  
    df.avg_dist_city_center_CT[df.City == city] = np.nansum(np.load(path_carbon_tax + city + "_density.npy")[20] * np.load(path_carbon_tax + city + "_distance.npy")) / np.nansum(np.load(path_carbon_tax + city + "_density.npy")[20])
    df.modal_share_cars_CT[df.City == city] = 100 * np.nansum(np.load(path_carbon_tax + city + "_density.npy")[20][np.load(path_carbon_tax + city + "_modal_shares.npy")[20] == 0]) / np.nansum(np.load(path_carbon_tax + city + "_density.npy")[20]) 
    df.avg_dist_city_center_FE[df.City == city] = np.nansum(np.load(path_fuel_efficiency + city + "_density.npy")[20] * np.load(path_fuel_efficiency + city + "_distance.npy")) / np.nansum(np.load(path_fuel_efficiency + city + "_density.npy")[20])
    df.modal_share_cars_FE[df.City == city] = 100 * np.nansum(np.load(path_fuel_efficiency + city + "_density.npy")[20][np.load(path_fuel_efficiency + city + "_modal_shares.npy")[20] == 0]) / np.nansum(np.load(path_fuel_efficiency + city + "_density.npy")[20])
    df.avg_dist_city_center_BRT[df.City == city] = np.nansum(np.load(path_BRT + city + "_density.npy")[20] * np.load(path_BRT + city + "_distance.npy")) / np.nansum(np.load(path_BRT + city + "_density.npy")[20])
    df.modal_share_cars_BRT[df.City == city] = 100 * np.nansum(np.load(path_BRT + city + "_density.npy")[20][np.load(path_BRT + city + "_modal_shares.npy")[20] == 0]) / np.nansum(np.load(path_BRT + city + "_density.npy")[20])
    df.avg_dist_city_center_UGB[df.City == city] = np.nansum(np.load(path_UGB + city + "_density.npy")[20] * np.load(path_UGB + city + "_distance.npy")) / np.nansum(np.load(path_UGB + city + "_density.npy")[20])
    df.modal_share_cars_UGB[df.City == city] = 100 * np.nansum(np.load(path_UGB + city + "_density.npy")[20][np.load(path_UGB + city + "_modal_shares.npy")[20] == 0]) / np.nansum(np.load(path_UGB + city + "_density.npy")[20])
    df.avg_dist_city_center_BAU[df.City == city] = np.nansum(np.load(path_BAU + city + "_density.npy")[20] * np.load(path_BAU + city + "_distance.npy")) / np.nansum(np.load(path_BAU + city + "_density.npy")[20])
    df.modal_share_cars_BAU[df.City == city] = 100 * np.nansum(np.load(path_BAU + city + "_density.npy")[20][np.load(path_BAU + city + "_modal_shares.npy")[20] == 0]) / np.nansum(np.load(path_BAU + city + "_density.npy")[20])
    df.avg_dist_city_center_all[df.City == city] = np.nansum(np.load(path_all + city + "_density.npy")[20] * np.load(path_all + city + "_distance.npy")) / np.nansum(np.load(path_all + city + "_density.npy")[20])
    df.modal_share_cars_all[df.City == city] = 100 * np.nansum(np.load(path_all + city + "_density.npy")[20][np.load(path_all + city + "_modal_shares.npy")[20] == 0]) / np.nansum(np.load(path_all + city + "_density.npy")[20])
    df.modal_share_pt_0[df.City == city] = 100 * np.nansum(np.load(path_BAU + city + "_density.npy")[0][np.load(path_BAU + city + "_modal_shares.npy")[0] == 1]) / np.nansum(np.load(path_BAU + city + "_density.npy")[0])
    df.modal_share_cars_0[df.City == city] = 100 * np.nansum(np.load(path_BAU + city + "_density.npy")[0][np.load(path_BAU + city + "_modal_shares.npy")[0] == 0]) / np.nansum(np.load(path_BAU + city + "_density.npy")[0])
    df.housing_2035_BAU[df.City == city] = np.average(np.load(path_BAU + city + "_dwelling_size.npy")[20], weights = np.load(path_BAU + city + "_density.npy")[20])
    df.housing_2035_CT[df.City == city] = np.average(np.load(path_carbon_tax + city + "_dwelling_size.npy")[20], weights = np.load(path_carbon_tax + city + "_density.npy")[20])
    df.housing_2035_FE[df.City == city] = np.average(np.load(path_fuel_efficiency + city + "_dwelling_size.npy")[20], weights = np.load(path_fuel_efficiency + city + "_density.npy")[20])
    df.housing_2035_BRT[df.City == city] = np.average(np.load(path_BRT + city + "_dwelling_size.npy")[20], weights = np.load(path_BRT + city + "_density.npy")[20])
    df.housing_2035_UGB[df.City == city] = np.average(np.load(path_UGB + city + "_dwelling_size.npy")[20], weights = np.load(path_UGB + city + "_density.npy")[20])
    df.housing_2035_all[df.City == city] = np.average(np.load(path_all + city + "_dwelling_size.npy")[20], weights = np.load(path_all + city + "_density.npy")[20])
    df.rent_2035_BAU[df.City == city] = np.average(np.load(path_BAU + city + "_rent.npy")[20], weights = np.load(path_BAU + city + "_density.npy")[20])
    df.rent_2035_CT[df.City == city] = np.average(np.load(path_carbon_tax + city + "_rent.npy")[20], weights = np.load(path_carbon_tax + city + "_density.npy")[20])
    df.rent_2035_FE[df.City == city] = np.average(np.load(path_fuel_efficiency + city + "_rent.npy")[20], weights = np.load(path_fuel_efficiency + city + "_density.npy")[20])
    df.rent_2035_BRT[df.City == city] = np.average(np.load(path_BRT + city + "_rent.npy")[20], weights = np.load(path_BRT + city + "_density.npy")[20])
    df.rent_2035_UGB[df.City == city] = np.average(np.load(path_UGB + city + "_rent.npy")[20], weights = np.load(path_UGB + city + "_density.npy")[20])
    df.rent_2035_all[df.City == city] = np.average(np.load(path_all + city + "_rent.npy")[20], weights = np.load(path_all + city + "_density.npy")[20])
    tcost_BAU = np.load(path_BAU + city + "_prix_transport.npy")
    tcost_CT = np.load(path_carbon_tax + city + "_prix_transport.npy")
    tcost_FE =np.load(path_fuel_efficiency + city + "_prix_transport.npy")
    tcost_BRT =np.load(path_BRT + city + "_prix_transport.npy")
    tcost_UGB = np.load(path_UGB + city + "_prix_transport.npy")
    tcost_all = np.load(path_all + city + "_prix_transport.npy")
    df.tcost_2035_BAU[df.City == city] = np.average(tcost_BAU[~np.isnan(tcost_BAU)], weights = np.load(path_BAU + city + "_density.npy")[20][~np.isnan(tcost_BAU)])
    df.tcost_2035_CT[df.City == city] = np.average(tcost_CT[~np.isnan(tcost_CT)], weights = np.load(path_carbon_tax + city + "_density.npy")[20][~np.isnan(tcost_CT)])
    df.tcost_2035_FE[df.City == city] = np.average(tcost_FE[~np.isnan(tcost_FE)], weights = np.load(path_fuel_efficiency + city + "_density.npy")[20][~np.isnan(tcost_FE)])
    df.tcost_2035_BRT[df.City == city] = np.average(tcost_BRT[~np.isnan(tcost_BRT)], weights = np.load(path_BRT + city + "_density.npy")[20][~np.isnan(tcost_BRT)])
    df.tcost_2035_UGB[df.City == city] = np.average(tcost_UGB[~np.isnan(tcost_UGB)], weights = np.load(path_UGB + city + "_density.npy")[20][~np.isnan(tcost_UGB)])
    df.tcost_2035_all[df.City == city] = np.average(tcost_all[~np.isnan(tcost_all)], weights = np.load(path_all + city + "_density.npy")[20][~np.isnan(tcost_all)])
    df.active_modes_2035_BAU[df.City == city] = float(np.load(path_BAU + city + "_active_modes.npy"))
    df.active_modes_2035_CT[df.City == city] = float(np.load(path_carbon_tax + city + "_active_modes.npy"))
    df.active_modes_2035_FE[df.City == city] = float(np.load(path_fuel_efficiency + city + "_active_modes.npy"))
    df.active_modes_2035_UGB[df.City == city] = float(np.load(path_UGB + city + "_active_modes.npy"))
    df.active_modes_2035_BRT[df.City == city] = float(np.load(path_BRT + city + "_active_modes.npy"))
    df.active_modes_2035_all[df.City == city] = float(np.load(path_all + city + "_active_modes.npy"))
    df.air_pollution_2035_BAU[df.City == city] = float(np.load(path_BAU + city + "_air_pollution.npy"))
    df.air_pollution_2035_CT[df.City == city] = float(np.load(path_carbon_tax + city + "_air_pollution.npy"))
    df.air_pollution_2035_FE[df.City == city] = float(np.load(path_fuel_efficiency + city + "_air_pollution.npy"))
    df.air_pollution_2035_UGB[df.City == city] = float(np.load(path_UGB + city + "_air_pollution.npy"))
    df.air_pollution_2035_BRT[df.City == city] = float(np.load(path_BRT + city + "_air_pollution.npy"))
    df.air_pollution_2035_all[df.City == city] = float(np.load(path_all + city + "_air_pollution.npy"))
    df.car_accidents_2035_BAU[df.City == city] = float(np.load(path_BAU + city + "_car_accidents.npy"))
    df.car_accidents_2035_CT[df.City == city] = float(np.load(path_carbon_tax + city + "_car_accidents.npy"))
    df.car_accidents_2035_FE[df.City == city] = float(np.load(path_fuel_efficiency + city + "_car_accidents.npy"))
    df.car_accidents_2035_UGB[df.City == city] = float(np.load(path_UGB + city + "_car_accidents.npy"))
    df.car_accidents_2035_BRT[df.City == city] = float(np.load(path_BRT + city + "_car_accidents.npy"))
    df.car_accidents_2035_all[df.City == city] = float(np.load(path_all + city + "_car_accidents.npy"))
    df.noise_2035_BAU[df.City == city] = float(np.load(path_BAU + city + "_noise.npy"))
    df.noise_2035_CT[df.City == city] = float(np.load(path_carbon_tax + city + "_noise.npy"))
    df.noise_2035_FE[df.City == city] = float(np.load(path_fuel_efficiency + city + "_noise.npy"))
    df.noise_2035_UGB[df.City == city] = float(np.load(path_UGB + city + "_noise.npy"))
    df.noise_2035_BRT[df.City == city] = float(np.load(path_BRT + city + "_noise.npy"))
    df.noise_2035_all[df.City == city] = float(np.load(path_all + city + "_noise.npy"))
    df.income_2035[df.City == city] = float(np.load(path_BAU + city + "_income.npy")[20])
    df.tax_2035_CT[df.City == city] = float(np.load(path_carbon_tax + city + "_save_tax_per_pers.npy")[20])
    df.tax_2035_all[df.City == city] = float(np.load(path_all + city + "_save_tax_per_pers.npy")[20])
    df.cost_2035_BRT[df.City == city] = float(np.load(path_BRT + city + "_cost_BRT_per_pers.npy")[20])
    df.cost_2035_all[df.City == city] = float(np.load(path_all + city + "_cost_BRT_per_pers.npy")[20])
    if welfare_increasing == True:
        df.tax_2035_all_welfare_increasing[df.City == city] = float(np.load(path_welfare_increasing + city + "_save_tax_per_pers.npy")[20])
        df.cost_2035_all_welfare_increasing[df.City == city] = float(np.load(path_welfare_increasing + city + "_cost_BRT_per_pers.npy")[20])
        df.emissions_2035_all_welfare_increasing[df.City == city] = np.load(path_welfare_increasing + city + "_emissions.npy")[20]
        df.welfare_with_cobenefits_2035_all_welfare_increasing[df.City == city] = np.load(path_welfare_increasing + city + "_total_welfare_with_cobenefits.npy")[20]  
        df.welfare_without_cobenefits_2035_all_welfare_increasing[df.City == city] = np.load(path_welfare_increasing + city + "_total_welfare.npy")[20]  
        df.avg_dist_city_center_all_welfare_increasing[df.City == city] = np.nansum(np.load(path_welfare_increasing + city + "_density.npy")[20] * np.load(path_welfare_increasing + city + "_distance.npy")) / np.nansum(np.load(path_welfare_increasing + city + "_density.npy")[20])
        df.modal_share_cars_all_welfare_increasing[df.City == city] = 100 * np.nansum(np.load(path_welfare_increasing + city + "_density.npy")[20][np.load(path_welfare_increasing + city + "_modal_shares.npy")[20] == 0]) / np.nansum(np.load(path_welfare_increasing + city + "_density.npy")[20]) 
        df.housing_2035_all_welfare_increasing[df.City == city] = np.average(np.load(path_welfare_increasing + city + "_dwelling_size.npy")[20], weights = np.load(path_welfare_increasing + city + "_density.npy")[20])
        df.rent_2035_all_welfare_increasing[df.City == city] = np.average(np.load(path_welfare_increasing + city + "_rent.npy")[20], weights = np.load(path_welfare_increasing + city + "_density.npy")[20])
        tcost_all_welfare_increasing = np.load(path_welfare_increasing + city + "_prix_transport.npy")
        df.tcost_2035_all_welfare_increasing[df.City == city] = np.average(tcost_all_welfare_increasing[~np.isnan(tcost_all_welfare_increasing)], weights = np.load(path_welfare_increasing + city + "_density.npy")[20][~np.isnan(tcost_all_welfare_increasing)])
        df.active_modes_2035_all_welfare_increasing[df.City == city] = float(np.load(path_welfare_increasing + city + "_active_modes.npy"))
        df.air_pollution_2035_all_welfare_increasing[df.City == city] = float(np.load(path_welfare_increasing + city + "_air_pollution.npy"))
        df.car_accidents_2035_all_welfare_increasing[df.City == city] = float(np.load(path_welfare_increasing + city + "_car_accidents.npy"))
        df.noise_2035_all_welfare_increasing[df.City == city] = float(np.load(path_welfare_increasing + city + "_noise.npy"))

        
df.iloc[:,1:102] = df.iloc[:,1:102].astype(float)

df.health_2035_BAU = df.active_modes_2035_BAU - (df.noise_2035_BAU + df.car_accidents_2035_BAU + df.air_pollution_2035_BAU)
df.health_2035_CT = df.active_modes_2035_CT - (df.noise_2035_CT + df.car_accidents_2035_CT + df.air_pollution_2035_CT)
df.health_2035_FE = df.active_modes_2035_FE - (df.noise_2035_FE + df.car_accidents_2035_FE + df.air_pollution_2035_FE)
df.health_2035_UGB = df.active_modes_2035_UGB - (df.noise_2035_UGB + df.car_accidents_2035_UGB + df.air_pollution_2035_UGB)
df.health_2035_BRT = df.active_modes_2035_BRT - (df.noise_2035_BRT + df.car_accidents_2035_BRT + df.air_pollution_2035_BRT)
df.health_2035_all = df.active_modes_2035_all - (df.noise_2035_all + df.car_accidents_2035_all + df.air_pollution_2035_all)
df.health_2035_all_welfare_increasing = df.active_modes_2035_all_welfare_increasing - (df.noise_2035_all_welfare_increasing + df.car_accidents_2035_all_welfare_increasing + df.air_pollution_2035_all_welfare_increasing)
    
# Compute variations in outcomes
df["var_health_CT"] = -100 * (df.health_2035_CT - df.health_2035_BAU) / df.health_2035_BAU
df["var_health_FE"] = -100 * (df.health_2035_FE - df.health_2035_BAU) / df.health_2035_BAU
df["var_health_UGB"] = -100 * (df.health_2035_UGB - df.health_2035_BAU) / df.health_2035_BAU
df["var_health_BRT"] = -100 * (df.health_2035_BRT - df.health_2035_BAU) / df.health_2035_BAU
df["var_health_all"] = -100 * (df.health_2035_all - df.health_2035_BAU) / df.health_2035_BAU
df["var_housing_CT"] = 100 * (df.housing_2035_CT - df.housing_2035_BAU) / df.housing_2035_BAU
df["var_housing_FE"] = 100 * (df.housing_2035_FE - df.housing_2035_BAU) / df.housing_2035_BAU
df["var_housing_UGB"] = 100 * (df.housing_2035_UGB - df.housing_2035_BAU) / df.housing_2035_BAU
df["var_housing_BRT"] = 100 * (df.housing_2035_BRT - df.housing_2035_BAU) / df.housing_2035_BAU
df["var_housing_all"] = 100 * (df.housing_2035_all - df.housing_2035_BAU) / df.housing_2035_BAU
df["var_rent_CT"] = 100 * (df.rent_2035_CT - df.rent_2035_BAU) / df.rent_2035_BAU
df["var_rent_FE"] = 100 * (df.rent_2035_FE - df.rent_2035_BAU) / df.rent_2035_BAU
df["var_rent_UGB"] = 100 * (df.rent_2035_UGB - df.rent_2035_BAU) / df.rent_2035_BAU
df["var_rent_BRT"] = 100 * (df.rent_2035_BRT - df.rent_2035_BAU) / df.rent_2035_BAU
df["var_rent_all"] = 100 * (df.rent_2035_all - df.rent_2035_BAU) / df.rent_2035_BAU
df["var_tcost_CT"] = 100 * (df.tcost_2035_CT - df.tcost_2035_BAU) / df.tcost_2035_BAU
df["var_tcost_FE"] = 100 * (df.tcost_2035_FE - df.tcost_2035_BAU) / df.tcost_2035_BAU
df["var_tcost_UGB"] = 100 * (df.tcost_2035_UGB - df.tcost_2035_BAU) / df.tcost_2035_BAU
df["var_tcost_BRT"] = 100 * (df.tcost_2035_BRT - df.tcost_2035_BAU) / df.tcost_2035_BAU
df["var_tcost_all"] = 100 * (df.tcost_2035_all - df.tcost_2035_BAU) / df.tcost_2035_BAU
df["emissions_2035_BRT_var"] = 100 * (df.emissions_2035_BRT - df.emissions_2035_BAU) / df.emissions_2035_BAU
df["emissions_2035_UGB_var"] = 100 * (df.emissions_2035_UGB - df.emissions_2035_BAU) / df.emissions_2035_BAU
df["emissions_2035_FE_var"] = 100 * (df.emissions_2035_FE - df.emissions_2035_BAU) / df.emissions_2035_BAU
df["emissions_2035_CT_var"] = 100 * (df.emissions_2035_CT - df.emissions_2035_BAU) / df.emissions_2035_BAU
df["emissions_2035_all_var"] = 100 * (df.emissions_2035_all - df.emissions_2035_BAU) / df.emissions_2035_BAU
df["welfare_2035_BRT_var"] = 100 * (df.welfare_with_cobenefits_2035_BRT - df.welfare_with_cobenefits_2035_BAU) / df.welfare_with_cobenefits_2035_BAU
df["welfare_2035_UGB_var"] = 100 * (df.welfare_with_cobenefits_2035_UGB - df.welfare_with_cobenefits_2035_BAU) / df.welfare_with_cobenefits_2035_BAU
df["welfare_2035_FE_var"] = 100 * (df.welfare_with_cobenefits_2035_FE - df.welfare_with_cobenefits_2035_BAU) / df.welfare_with_cobenefits_2035_BAU
df["welfare_2035_CT_var"] = 100 * (df.welfare_with_cobenefits_2035_CT - df.welfare_with_cobenefits_2035_BAU) / df.welfare_with_cobenefits_2035_BAU
df["welfare_2035_all_var"] = 100 * (df.welfare_with_cobenefits_2035_all - df.welfare_with_cobenefits_2035_BAU) / df.welfare_with_cobenefits_2035_BAU
df["welfare_2035_BRT_var_without"] = 100 * (df.welfare_without_cobenefits_2035_BRT - df.welfare_without_cobenefits_2035_BAU) / df.welfare_without_cobenefits_2035_BAU
df["welfare_2035_UGB_var_without"] = 100 * (df.welfare_without_cobenefits_2035_UGB - df.welfare_without_cobenefits_2035_BAU) / df.welfare_without_cobenefits_2035_BAU
df["welfare_2035_FE_var_without"] = 100 * (df.welfare_without_cobenefits_2035_FE - df.welfare_without_cobenefits_2035_BAU) / df.welfare_without_cobenefits_2035_BAU
df["welfare_2035_CT_var_without"] = 100 * (df.welfare_without_cobenefits_2035_CT - df.welfare_without_cobenefits_2035_BAU) / df.welfare_without_cobenefits_2035_BAU
df["welfare_2035_all_var_without"] = 100 * (df.welfare_without_cobenefits_2035_all - df.welfare_without_cobenefits_2035_BAU) / df.welfare_without_cobenefits_2035_BAU
df["var_active_modes_CT"] = np.nan
df["var_active_modes_FE"] = np.nan
df["var_active_modes_UGB"] = np.nan
df["var_active_modes_BRT"] = np.nan
df["var_active_modes_all"] = np.nan
df["var_active_modes_CT"][df.active_modes_2035_BAU >0] = 100 * (df.active_modes_2035_CT[df.active_modes_2035_BAU >0] - df.active_modes_2035_BAU)[df.active_modes_2035_BAU >0] / df.active_modes_2035_BAU[df.active_modes_2035_BAU >0]
df["var_active_modes_FE"][df.active_modes_2035_BAU >0] = 100 * (df.active_modes_2035_FE[df.active_modes_2035_BAU >0] - df.active_modes_2035_BAU)[df.active_modes_2035_BAU >0] / df.active_modes_2035_BAU[df.active_modes_2035_BAU >0]
df["var_active_modes_UGB"][df.active_modes_2035_BAU >0] = 100 * (df.active_modes_2035_UGB[df.active_modes_2035_BAU >0] - df.active_modes_2035_BAU)[df.active_modes_2035_BAU >0] / df.active_modes_2035_BAU[df.active_modes_2035_BAU >0]
df["var_active_modes_BRT"][df.active_modes_2035_BAU >0] = 100 * (df.active_modes_2035_BRT[df.active_modes_2035_BAU >0] - df.active_modes_2035_BAU)[df.active_modes_2035_BAU >0] / df.active_modes_2035_BAU[df.active_modes_2035_BAU >0]
df["var_active_modes_all"][df.active_modes_2035_BAU >0] = 100 * (df.active_modes_2035_all[df.active_modes_2035_BAU >0] - df.active_modes_2035_BAU)[df.active_modes_2035_BAU >0] / df.active_modes_2035_BAU[df.active_modes_2035_BAU >0]
df["var_air_pollution_CT"] = 100 * (df.air_pollution_2035_CT - df.air_pollution_2035_BAU) / df.air_pollution_2035_BAU
df["var_air_pollution_FE"] = 100 * (df.air_pollution_2035_FE - df.air_pollution_2035_BAU) / df.air_pollution_2035_BAU
df["var_air_pollution_UGB"] = 100 * (df.air_pollution_2035_UGB - df.air_pollution_2035_BAU) / df.air_pollution_2035_BAU
df["var_air_pollution_BRT"] = 100 * (df.air_pollution_2035_BRT - df.air_pollution_2035_BAU) / df.air_pollution_2035_BAU
df["var_air_pollution_all"] = 100 * (df.air_pollution_2035_all - df.air_pollution_2035_BAU) / df.air_pollution_2035_BAU
df["var_car_accidents_CT"] = 100 * (df.car_accidents_2035_CT - df.car_accidents_2035_BAU) / df.car_accidents_2035_BAU
df["var_car_accidents_FE"] = 100 * (df.car_accidents_2035_FE - df.car_accidents_2035_BAU) / df.car_accidents_2035_BAU
df["var_car_accidents_UGB"] = 100 * (df.car_accidents_2035_UGB - df.car_accidents_2035_BAU) / df.car_accidents_2035_BAU
df["var_car_accidents_BRT"] = 100 * (df.car_accidents_2035_BRT - df.car_accidents_2035_BAU) / df.car_accidents_2035_BAU
df["var_car_accidents_all"] = 100 * (df.car_accidents_2035_all - df.car_accidents_2035_BAU) / df.car_accidents_2035_BAU
df["var_noise_CT"] = 100 * (df.noise_2035_CT - df.noise_2035_BAU) / df.noise_2035_BAU
df["var_noise_FE"] = 100 * (df.noise_2035_FE - df.noise_2035_BAU) / df.noise_2035_BAU
df["var_noise_UGB"] = 100 * (df.noise_2035_UGB - df.noise_2035_BAU) / df.noise_2035_BAU
df["var_noise_BRT"] = 100 * (df.noise_2035_BRT - df.noise_2035_BAU) / df.noise_2035_BAU
df["var_noise_all"] = 100 * (df.noise_2035_all - df.noise_2035_BAU) / df.noise_2035_BAU
df["disp_income_BAU"] = df.income_2035
df["disp_income_CT"] = df.income_2035 + df.tax_2035_CT
df["disp_income_BRT"] = df.income_2035 - df.cost_2035_BRT
df["disp_income_all"] = df.income_2035 + df.tax_2035_all - df.cost_2035_all
df["var_disp_income_CT"] = 100 * (df.disp_income_CT - df.disp_income_BAU) / df.disp_income_BAU
df["var_disp_income_BRT"] = 100 * (df.disp_income_BRT - df.disp_income_BAU) / df.disp_income_BAU
df["var_disp_income_all"] = 100 * (df.disp_income_all - df.disp_income_BAU) / df.disp_income_BAU
if welfare_increasing == True:
    df["disp_income_all_welfare_increasing"] = df.income_2035 + df.tax_2035_all_welfare_increasing - df.cost_2035_all_welfare_increasing
    df["var_disp_income_all_welfare_increasing"] = 100 * (df.disp_income_all_welfare_increasing - df.disp_income_BAU) / df.disp_income_BAU  
    df["var_health_all_welfare_increasing"] = -100 * (df.health_2035_all_welfare_increasing - df.health_2035_BAU) / df.health_2035_BAU
    df["var_housing_all_welfare_increasing"] = 100 * (df.housing_2035_all_welfare_increasing - df.housing_2035_BAU) / df.housing_2035_BAU
    df["var_rent_all_welfare_increasing"] = 100 * (df.rent_2035_all_welfare_increasing - df.rent_2035_BAU) / df.rent_2035_BAU
    df["var_tcost_all_welfare_increasing"] = 100 * (df.tcost_2035_all_welfare_increasing - df.tcost_2035_BAU) / df.tcost_2035_BAU
    df["emissions_2035_all_welfare_increasing_var"] = 100 * (df.emissions_2035_all_welfare_increasing - df.emissions_2035_BAU) / df.emissions_2035_BAU
    df["welfare_2035_all_welfare_increasing_var"] = 100 * (df.welfare_with_cobenefits_2035_all_welfare_increasing - df.welfare_with_cobenefits_2035_BAU) / df.welfare_with_cobenefits_2035_BAU
    df["welfare_2035_all_welfare_increasing_var_without"] = 100 * (df.welfare_without_cobenefits_2035_all_welfare_increasing - df.welfare_without_cobenefits_2035_BAU) / df.welfare_without_cobenefits_2035_BAU
    df["var_active_modes_all_welfare_increasing"] = np.nan
    df["var_active_modes_all_welfare_increasing"][df.active_modes_2035_BAU >0] = 100 * (df.active_modes_2035_all_welfare_increasing[df.active_modes_2035_BAU >0] - df.active_modes_2035_BAU)[df.active_modes_2035_BAU >0] / df.active_modes_2035_BAU[df.active_modes_2035_BAU >0]
    df["var_air_pollution_all_welfare_increasing"] = 100 * (df.air_pollution_2035_all_welfare_increasing - df.air_pollution_2035_BAU) / df.air_pollution_2035_BAU
    df["var_car_accidents_all_welfare_increasing"] = 100 * (df.car_accidents_2035_all_welfare_increasing - df.car_accidents_2035_BAU) / df.car_accidents_2035_BAU
    df["var_noise_all_welfare_increasing"] = 100 * (df.noise_2035_all_welfare_increasing - df.noise_2035_BAU) / df.noise_2035_BAU

    
#df.to_excel('C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/results_20221218.xlsx')

df["Population"] = np.nan
for city in df.City:
    df.Population[df.City == city] = np.load(path_BAU + city + "_population.npy")[20]
    
city_continent = pd.read_csv(path_data + "CityDatabases/cityDatabase_221NewCities.csv")
city_continent = city_continent.iloc[:, [0, 2]]
city_continent = city_continent.drop_duplicates(subset = "City")
city_continent = city_continent.sort_values('City')
df = df.merge(city_continent, on = "City", how = 'left')
color_tab = {'North_America':'red', 'Europe':'green', 'Asia':'blue', 'Oceania':'yellow', 'South_America': 'brown', 'Africa': 'orange'}

# Health analysis
new_df = pd.DataFrame(index = ['CT', 'FE', 'BRT', 'UGB'], columns = ['active_modes', 'air_pollution', 'car_accidents', 'noise'])    
for i in new_df.columns:
    for j in new_df.index:
        new_df.loc[j,i] = np.average(np.array(df['var_'+i+'_'+j][~np.isnan(df['var_'+i+'_'+j])].astype(float)), weights = np.array(df.Population.astype(float))[~np.isnan(df['var_'+i+'_'+j])])

fig, ax = plt.subplots(1,1, figsize = (10,6))
label =  ['Fuel \n Tax', 'Fuel \n Efficiency', 'Bus Rapid \n Transit', 'Urban Growth \n Boundary']
x = np.arange(len(label))
width = 0.2
rect1 = ax.bar(x - 0.4,
              new_df['active_modes'],
              width = width, 
               label = 'Active modes',
               edgecolor = "black"
              )
rect2 = ax.bar(x -0.2,
              new_df['air_pollution'],
              width = width,
              label = 'Air pollution',
              edgecolor = "black")
rect3 = ax.bar(x,
              new_df['car_accidents'],
              width = width,
              label = 'Car accidents',
              edgecolor = "black")
rect4 = ax.bar(x+0.2,
              new_df['noise'],
              width = width,
              label = 'Noise',
              edgecolor = "black")
ax.set_ylabel("Variations (%)",
              fontsize = 20,
              labelpad = 20)
ax.set_xticks(x)
ax.set_xticklabels(label,
                   fontsize = 20)
ax.legend(fontsize = 16)

# Income analysis
new_df = pd.DataFrame(index = ['CT', 'BRT', 'all', 'all_welfare_increasing'], columns = ['disp_income'])    
for i in new_df.columns:
    for j in new_df.index:
        new_df.loc[j,i] = np.average(np.array(df['var_'+i+'_'+j][~np.isnan(df['var_'+i+'_'+j])].astype(float)), weights = np.array(df.Population.astype(float))[~np.isnan(df['var_'+i+'_'+j])])

fig, ax = plt.subplots(1,1, figsize = (10,6))
label =  ['CT', 'BRT', 'all', 'all_welfare_increasing']
x = np.arange(len(label))
width = 0.5
rect1 = ax.bar(x,
              new_df['disp_income'],
              width = width, 
               label = 'disp_income modes',
               edgecolor = "black"
              )
ax.set_ylabel("Variations (%)",
              fontsize = 20,
              labelpad = 20)
ax.set_xticks(x)
ax.set_xticklabels(label,
                   fontsize = 20)
ax.legend(fontsize = 16)

# Plot the impact of the policies on rents, dwelling sizes, transportation costs, and health cobenefits
new_df = pd.DataFrame(index = ['CT', 'FE', 'BRT', 'UGB'], columns = ['rent', 'health', 'housing', 'tcost'])    
for i in new_df.columns:
    for j in new_df.index:
        new_df.loc[j,i] = np.average(np.array(df['var_'+i+'_'+j].astype(float)), weights = np.array(df.Population.astype(float)))

fig, ax = plt.subplots(1,1, figsize = (10,6))
label =  ['Fuel \n Tax', 'Fuel \n Efficiency', 'Bus Rapid \n Transit', 'Urban Growth \n Boundary']
x = np.arange(len(label))
width = 0.2
rect1 = ax.bar(x - 0.4,
              new_df['rent'],
              width = width, 
               label = 'Avg. rent per m2',
               edgecolor = "black"
              )
rect2 = ax.bar(x -0.2,
              new_df['health'],
              width = width,
              label = 'Avg. health cobenefits',
              edgecolor = "black")
rect3 = ax.bar(x,
              new_df['housing'],
              width = width,
              label = 'Avg. dwelling sizes',
              edgecolor = "black")
rect4 = ax.bar(x+0.2,
              new_df['tcost'],
              width = width,
              label = 'Avg. transportation costs',
              edgecolor = "black")
ax.set_ylabel("Variations (%)",
              fontsize = 20,
              labelpad = 20)
ax.set_xticks(x)
ax.set_xticklabels(label,
                   fontsize = 20)
ax.legend(fontsize = 16)
ax.grid()

# Plot the impact on emissions and welfare
if welfare_increasing == True:
    data = pd.DataFrame(columns = ['policy', 'Average emissions'])
    data['policy'] = ["Fuel tax", "BRT", "Fuel efficiency", "UGB", "all", "welfare_increasing"]
    data['Average emissions'] = [np.nansum(df.emissions_2035_CT_var * df.population_2035) / np.nansum(df.population_2035), np.nansum(df.emissions_2035_BRT_var * df.population_2035) / np.nansum(df.population_2035), np.nansum(df.emissions_2035_FE_var * df.population_2035) / np.nansum(df.population_2035), np.nansum(df.emissions_2035_UGB_var * df.population_2035) / np.nansum(df.population_2035), np.nansum(df.emissions_2035_all_var * df.population_2035) / np.nansum(df.population_2035), np.nansum(df.emissions_2035_all_welfare_increasing_var * df.population_2035) / np.nansum(df.population_2035)]
    data['Average welfare'] = [np.nansum(df.welfare_2035_CT_var * df.population_2035) / np.nansum(df.population_2035), np.nansum(df.welfare_2035_BRT_var * df.population_2035) / np.nansum(df.population_2035), np.nansum(df.welfare_2035_FE_var * df.population_2035) / np.nansum(df.population_2035), np.nansum(df.welfare_2035_UGB_var * df.population_2035) / np.nansum(df.population_2035), np.nansum(df.welfare_2035_all_var * df.population_2035) / np.nansum(df.population_2035), np.nansum(df.welfare_2035_all_welfare_increasing_var * df.population_2035) / np.nansum(df.population_2035)]
    data['Average welfare without cobenefits'] = [np.nansum(df.welfare_2035_CT_var_without * df.population_2035) / np.nansum(df.population_2035), np.nansum(df.welfare_2035_BRT_var_without * df.population_2035) / np.nansum(df.population_2035), np.nansum(df.welfare_2035_FE_var_without * df.population_2035) / np.nansum(df.population_2035), np.nansum(df.welfare_2035_UGB_var_without * df.population_2035) / np.nansum(df.population_2035), np.nansum(df.welfare_2035_all_var_without * df.population_2035) / np.nansum(df.population_2035), np.nansum(df.welfare_2035_all_welfare_increasing_var_without * df.population_2035) / np.nansum(df.population_2035)]

tidy = data.melt(id_vars='policy').rename(columns=str.title)
tidy = tidy[0:12]
plt.rcParams['figure.dpi'] = 360
sns.set(style="whitegrid")
fig, ax = plt.subplots(figsize=(12,4))
sns.barplot(data=tidy, x="Policy", y="Value", hue = "Variable", palette=['#fc8d62', '#66c2a5', '#66c2a5'])#("Set2"))
plt.xlabel('')
plt.ylabel('')
plt.yticks([], [])
plt.xticks(ticks = np.arange(6), labels = ["Fuel Tax", "Bus Rapid Transit", "Fuel Efficiency", "Urban Growth Boundary", "All", "All welfare-increasing"])
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles[0:], labels=["Total transport emissions", "Average welfare"], title = '', bbox_to_anchor=(0.65, 0.3), fontsize = 14, title_fontsize=14, loc=2, borderaxespad=0.)._legend_box.align = "left"
ax.tick_params(axis = 'both', labelsize=14, color='#4f4e4e')
sns.despine(left=True, top = True)
plt.text(x=0.2, y=0.6, s='+' +str(round(data["Average welfare"][0], 1))+ "%", 
                 color='black', fontsize=14, horizontalalignment='center')
plt.text(x=1.2, y=0.6, s='+' +str(round(data["Average welfare"][1], 1))+ "%", 
                 color='black', fontsize=14, horizontalalignment='center')
plt.text(x=2.2, y=0.6, s='+' +str(round(data["Average welfare"][2], 1))+ "%", 
                 color='black', fontsize=14, horizontalalignment='center')
plt.text(x=3.2, y=-1, s=str(round(data["Average welfare"][3], 1))+ "%", 
                 color='white', fontsize=14, horizontalalignment='center')
plt.text(x=4.2, y=-1, s=str(round(data["Average welfare"][4], 1))+ "%", 
                 color='black', fontsize=14, horizontalalignment='center')
plt.text(x=5.2, y=-1, s=str(round(data["Average welfare"][5], 1))+ "%", 
                 color='black', fontsize=14, horizontalalignment='center')
plt.text(x=-0.2, y=-1, s=str(round(data["Average emissions"][0], 1))+ "%", 
                 color='white', fontsize=14, horizontalalignment='center')
plt.text(x=0.8, y=-1, s=str(round(data["Average emissions"][1], 1))+ "%", 
                 color='white', fontsize=14, horizontalalignment='center')
plt.text(x=1.8, y=-1, s=str(round(data["Average emissions"][2], 1))+ "%", 
                 color='white', fontsize=14, horizontalalignment='center')
plt.text(x=2.8, y=-1, s=str(round(data["Average emissions"][3], 1))+ "%", 
                 color='white', fontsize=14, horizontalalignment='center')
plt.text(x=3.8, y=-1, s=str(round(data["Average emissions"][4], 1))+ "%", 
                 color='white', fontsize=14, horizontalalignment='center')
plt.text(x=4.8, y=-1, s=str(round(data["Average emissions"][5], 1))+ "%", 
                 color='white', fontsize=14, horizontalalignment='center')
#plt.savefig('aggregated.png')


# Regressions: emissions/welfare variations on city characteristics
city_characteristics2 = pd.read_excel("C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/city_characteristics_20211027.xlsx")
city_characteristics = pd.read_excel("C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/city_characteristics_20210810.xlsx")
df = df.merge(city_characteristics, left_on = "City", right_on = 'city')
df = df.merge(city_characteristics2.loc[:, ["city", "agricultural_rent"]], on = "city")
compute_density = pd.read_excel("C:/Users/charl/OneDrive/Bureau/Urban sprawl/Urban sprawl and density/scenarios_densities_20211115.xlsx")
compute_density["data_density_2015"] = 0.01 * compute_density.data_pop_2015 / compute_density.ESA_land_cover_2015
df = df.merge(compute_density, on = 'City')
df["log_population"] = np.log(df.population)
df["log_income"] = np.log(df.income)
df["log_pop2035"] = np.log(df.pop_2035)
df["log_inc2035"] = np.log(df.inc_2035)
df["log_agri_rent"] = np.log(df.agricultural_rent)
d_corr_density_scells2 = np.load(path_calibration + "d_corr_density_scells2.npy", allow_pickle = True)
d_corr_density_scells2 = np.array(d_corr_density_scells2, ndmin = 1)[0]
d_corr_rent_scells2 = np.load(path_calibration + "d_corr_rent_scells2.npy", allow_pickle = True)
d_corr_rent_scells2 = np.array(d_corr_rent_scells2, ndmin = 1)[0]
r2density_scells2 = np.load(path_calibration + "r2density_scells2.npy", allow_pickle = True)
r2density_scells2 = np.array(r2density_scells2, ndmin = 1)[0]
r2rent_scells2 = np.load(path_calibration + "r2rent_scells2.npy", allow_pickle = True)
r2rent_scells2 = np.array(r2rent_scells2, ndmin = 1)[0]
df["corr_rent"] = np.array(list(d_corr_rent_scells2.values()))[sample_of_cities.loc[sample_of_cities.City != 'Sfax', :].final_sample == 1, 0]
df["corr_density"] = np.array(list(d_corr_density_scells2.values()))[sample_of_cities.loc[sample_of_cities.City != 'Sfax', :].final_sample == 1, 0]
df["r2_rent"] = np.array(list(r2rent_scells2.values()))[sample_of_cities.loc[sample_of_cities.City != 'Sfax', :].final_sample == 1]
df["r2_density"] = np.array(list(r2density_scells2.values()))[sample_of_cities.loc[sample_of_cities.City != 'Sfax', :].final_sample == 1]
df["pop_growth"] = df.pop_2035 / df.population
df["log_pop_growth"] = np.log(df.pop_growth)
df["inc_growth"] = df.inc_2035 / df.income
df["log_inc_growth"] = np.log(df.inc_growth)

df["urba"] = np.nan
for city in df.City:
    country = list_city.Country[list_city.City == city].iloc[0]
    proj = str(int(list_city.GridEPSG[list_city.City == city].iloc[0]))
    
    land_cover_ESACCI = pd.read_csv(path_data + 'Data/' + country + '/' + city + 
                           '/Land_Cover/grid_ESACCI_LandCover_2015_' + 
                           str.upper(city) + '_' + proj +'.csv')
    
    df.urba[df.City == city] = np.nansum(land_cover_ESACCI.ESACCI190)

df["length_network"] = np.nan
for city in df.City:
    df.length_network[df.City == city] = np.load('C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/capital_costs_25_0_01_50_5/' + city + "_length_network.npy").item()
    
    
df["log_urba"] = np.log(df.urba)
df["density"] = df.urba / df.population
df["log_density"] = np.log(df.density)
df["log_length_network"] = np.log(df.length_network)
df["length_network2"] = df.length_network * df.length_network
df["network_pop"] = df.length_network / df.population
df["network_pop2"] = df["network_pop"] * df["network_pop"]
df["diff_pop"] = df["pop_2035"] - df["population"]
df["diff_inc"] = df["inc_2035"] - df["income"]

informal_housing = import_informal_housing(list_city, path_folder)
df = df.merge(informal_housing.loc[:, ["City", "informal_housing"]], left_on = "City", right_on = "City")
#bus_rapid = df.loc[:, ["City", "Continent_y", "quartiles_BRT", "length_network", "network_pop", "substitution_potential", "modal_share_car"]]

polycentricity = pd.read_excel(path_data + 'Article/Data_Criterias/CBD_Criterias_Table.ods', engine="odf")
polycentricity = polycentricity.iloc[:, [2, 16]]
polycentricity.columns = ['city', 'polycentricity_index']
polycentricity.loc[polycentricity.city == "Addis Ababa", "city"] = "Addis_Ababa"
polycentricity.loc[polycentricity.city == "Belo Horizonte", "city"] = "Belo_Horizonte"
polycentricity.loc[polycentricity.city == "Buenos Aires", "city"] = "Buenos_Aires"
polycentricity.loc[polycentricity.city == "Cape Town", "city"] = "Cape_Town"
polycentricity.loc[polycentricity.city == "Chiang Mai", "city"] = "Chiang_Mai"
polycentricity.loc[polycentricity.city == "Cluj-Napoca", "city"] = "Cluj_Napoca"
polycentricity.loc[polycentricity.city == "Frankfurt am Main", "city"] = "Frankfurt_am_Main"
polycentricity.loc[polycentricity.city == "Goiânia", "city"] = "Goiania"
polycentricity.loc[polycentricity.city == "Hong Kong", "city"] = "Hong_Kong"
polycentricity.loc[polycentricity.city == "Los Angeles", "city"] = "Los_Angeles"
polycentricity.loc[polycentricity.city == "Malmö", "city"] = "Malmo"
polycentricity.loc[polycentricity.city == "Mar del Plata", "city"] = "Mar_del_Plata"
polycentricity.loc[polycentricity.city == "Mexico City", "city"] = "Mexico_City"
polycentricity.loc[polycentricity.city == "New York", "city"] = "New_York"
polycentricity.loc[polycentricity.city == "Nizhny Novgorod", "city"] = "Nizhny_Novgorod"
polycentricity.loc[polycentricity.city == "Porto Alegre", "city"] = "Porto_Alegre"
polycentricity.loc[polycentricity.city == "Rio de Janeiro", "city"] = "Rio_de_Janeiro"
polycentricity.loc[polycentricity.city == "Rostov-on-Don", "city"] = "Rostov_on_Don"
polycentricity.loc[polycentricity.city == "San Diego", "city"] = "San_Diego"
polycentricity.loc[polycentricity.city == "San Fransisco", "city"] = "San_Fransisco"
polycentricity.loc[polycentricity.city == "Sao Paulo", "city"] = "Sao_Paulo"
polycentricity.loc[polycentricity.city == "St Petersburg", "city"] = "St_Petersburg"
polycentricity.loc[polycentricity.city == "The Hague", "city"] = "The_Hague"
polycentricity.loc[polycentricity.city == "Ulan Bator", "city"] = "Ulan_Bator"
polycentricity.loc[polycentricity.city == "Washington DC", "city"] = "Washington_DC"
polycentricity.loc[polycentricity.city == "Zürich", "city"] = "Zurich"
df = df.merge(polycentricity.loc[:, ["city", "polycentricity_index"]], left_on = "City", right_on = "city")
    
second_step = pd.DataFrame(index = list_city.City, columns = ["spatial_data_cover", "market_data_cover"])

for city in list(list_city.City):
    
    country = list_city.Country[list_city.City == city].iloc[0]
    proj = str(int(list_city.GridEPSG[list_city.City == city].iloc[0]))
    
    #Population  
    density = pd.read_csv(path_data + 'Data/' + country + '/' + city +
                          '/Population_Density/grille_GHSL_density_2015_' +
                          str.upper(city) + '.txt', sep = '\s+|,', engine='python')
    density = density.loc[:,density.columns.str.startswith("density")]
    density = np.array(density).squeeze()
    
    (country, proj, density, rents_and_size, land_use, land_cover_ESACCI, driving, transit, grille, 
     centre, distance_cbd, conversion_rate) = import_data(list_city, path_data, city, path_folder)        
    density = density.loc[:,density.columns.str.startswith("density")]
    density = np.array(density).squeeze()
    second_step.loc[second_step.index == city, "spatial_data_cover"] = sum(~np.isnan(rents_and_size.medRent)) / sum((pd.to_numeric(density)) > 0) 
    second_step.loc[second_step.index == city, "market_data_cover"] = np.nansum(density) / np.nansum(rents_and_size.dataCount)
    
df = df.merge(second_step, left_on = "City", right_on = "City")

df.market_data_cover = df.market_data_cover.astype(float)
df.spatial_data_cover = df.spatial_data_cover.astype(float)

city_continent = pd.read_csv(path_folder + "CityDatabases/cityDatabase_221NewCities.csv")
city_continent = city_continent.iloc[:, [0, 2]]
city_continent = city_continent.drop_duplicates(subset = "City")
city_continent = city_continent.sort_values('City')
df = df.merge(city_continent, left_on = "City", right_on = "City")
fixed_effects = pd.get_dummies(df.Continent_y)
df["Asia"] = fixed_effects.Asia
#df["Africa"] = fixed_effects.Africa
df["North_America"] = fixed_effects.North_America
df["Oceania"] = fixed_effects.Oceania
df["South_America"] = fixed_effects.South_America
df["Europe"]= fixed_effects.Europe

s = "log_population + log_income + log_agri_rent + substitution_potential + log_pop_growth + log_inc_growth + r2_density"
s_brt = s + "+ network_pop+ network_pop2"

df["emissions_2035_CT_var"] = df["emissions_2035_CT_var"].astype(float)
df["emissions_2035_FE_var"] = df["emissions_2035_FE_var"].astype(float)
df["emissions_2035_UGB_var"] = df["emissions_2035_UGB_var"].astype(float)
df["emissions_2035_BRT_var"] = df["emissions_2035_BRT_var"].astype(float)

reg1 = ols("emissions_2035_CT_var ~ " + s, data=df).fit(cov_type='HC3')
reg1.summary()

reg2 = ols("emissions_2035_FE_var ~ " + s, data=df).fit(cov_type='HC3')
reg2.summary()

reg3 = ols("emissions_2035_UGB_var ~ " + s, data=df).fit(cov_type='HC3')
reg3.summary()

reg4 = ols("emissions_2035_BRT_var ~ " + s_brt, data=df).fit(cov_type='HC2')
reg4.summary()

df["welfare_2035_CT_var"] = df["welfare_2035_CT_var"].astype(float)
df["welfare_2035_FE_var"] = df["welfare_2035_FE_var"].astype(float)
df["welfare_2035_UGB_var"] = df["welfare_2035_UGB_var"].astype(float)
df["welfare_2035_BRT_var"] = df["welfare_2035_BRT_var"].astype(float)

reg1 = ols("welfare_2035_CT_var ~ " + s, data=df).fit(cov_type='HC3')
reg1.summary()

reg2 = ols("welfare_2035_FE_var ~ " + s, data=df).fit(cov_type='HC3')
reg2.summary()

reg3 = ols("welfare_2035_UGB_var ~ " + s, data=df).fit(cov_type='HC3')
reg3.summary()

reg4 = ols("welfare_2035_BRT_var ~ " + s_brt, data=df).fit(cov_type='HC2')
reg4.summary()

#PCA Analysis
features = ['log_population', 'log_income', 'substitution_potential', 'log_pop_growth', 'log_inc_growth', 'log_density']
x = df.loc[:, features].values
y = df.loc[:,['cost_effectiveness_with_cobenefits_BRT', 'cost_effectiveness_with_cobenefits_UGB', 'cost_effectiveness_with_cobenefits_FE', 'cost_effectiveness_with_cobenefits_CT']].values
x = StandardScaler().fit_transform(x)
pca = PCA(n_components=4)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal_component_1', 'principal_component_2', 'principal_component_3', 'principal_component_4'])

finalDf = pd.concat([principalDf, df[['cost_effectiveness_with_cobenefits_BRT', 'cost_effectiveness_with_cobenefits_UGB', 'cost_effectiveness_with_cobenefits_FE', 'cost_effectiveness_with_cobenefits_CT',"welfare_2035_CT_var","welfare_2035_FE_var","welfare_2035_UGB_var","welfare_2035_BRT_var","emissions_2035_CT_var","emissions_2035_FE_var","emissions_2035_UGB_var","emissions_2035_BRT_var", "City", "r2_density", 'network_pop', 'network_pop2']]], axis = 1)
print(pca.explained_variance_ratio_)
print(pca.components_)

plt.figure(figsize = (12.2, 6))
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 20}

plt.rc('font', **font)
ax = sns.heatmap(pca.components_,
                 cmap='YlGnBu',
                 yticklabels=[ "PCA"+str(x) for x in range(1,pca.n_components_+1)],
                 xticklabels=['log(population)', 'log(income)', 'modal shift \n potential', 'log(pop growth)', 'log(inc growth)', 'log(density)'],
                 cbar_kws={"orientation": "vertical"})
ax.set_xticklabels(labels = ['log(population)', 'log(income)', 'modal shift \n potential', 'log(pop growth)', 'log(inc growth)', 'log(density)'], rotation=0, ha='center')
ax.set_aspect("equal")

s = 'principal_component_1 + principal_component_2 + principal_component_3+ principal_component_4'

reg1 = ols("welfare_2035_CT_var ~ " + s + '+ r2_density', data=finalDf).fit(cov_type='HC3')
reg1.summary()

eg1 = ols("welfare_2035_FE_var ~ " + s + '+ r2_density', data=finalDf).fit(cov_type='HC3')
reg1.summary()

reg1 = ols("welfare_2035_UGB_var ~ " + s + '+ r2_density', data=finalDf).fit(cov_type='HC3')
reg1.summary()

reg1 = ols("welfare_2035_BRT_var ~ " + s + '+ r2_density + network_pop + network_pop2', data=finalDf).fit(cov_type='HC3')
reg1.summary()

reg1 = ols("emissions_2035_CT_var ~ " + s + '+ r2_density', data=finalDf).fit(cov_type='HC3')
reg1.summary()

reg1 = ols("emissions_2035_FE_var ~ " + s + '+ r2_density', data=finalDf).fit(cov_type='HC3')
reg1.summary()

reg1 = ols("emissions_2035_UGB_var ~ " + s + '+ r2_density', data=finalDf).fit(cov_type='HC3')
reg1.summary()

reg1 = ols("emissions_2035_BRT_var ~ " + s + '+ r2_density + network_pop + network_pop2', data=finalDf).fit(cov_type='HC3')
reg1.summary()

#Impact of policies on the distance to the city center and on modal shares
df["var_dist_city_center_CT"] = 100 * (df.avg_dist_city_center_CT - df.avg_dist_city_center_BAU) / df.avg_dist_city_center_BAU
df["var_modal_share_cars_CT"] = 100 * (df.modal_share_cars_CT - df.modal_share_cars_BAU) / df.modal_share_cars_BAU

colors = ['#fc8d62', '#fc8d62', '#fc8d62']
colors1 = dict(color=colors[0])
colors2 = dict(color=colors[1])
colors3 = dict(color=colors[2])

fig, ax = plt.subplots(figsize = (6, 4))
ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.yaxis.set_ticks_position('none')
ax.tick_params(axis = 'both', labelsize=14, color='#4f4e4e')
ax.grid(color='grey', axis='x', linestyle='-', linewidth=0.25, alpha=0.5)
bp1 = sns.boxplot(y = "var_dist_city_center_CT", x = "Continent", data = df, order = ['Asia', 'Europe', 'North_America', 'Oceania', 'South_America'], boxprops = dict(linewidth=2, facecolor=(0,0,0,0), edgecolor="#fc8d62"), medianprops = dict(linewidth=2, color="#fc8d62"), whiskerprops = dict(linewidth=2, color="#fc8d62"), capprops = dict(linewidth=2, color="#fc8d62"), flierprops = colors3, showfliers=False, saturation = 1)
bp2 = sns.swarmplot(y = df.var_dist_city_center_CT, x = df.Continent, palette = ["#404040"], order = ['Asia', 'Europe', 'North_America', 'Oceania', 'South_America'])
plt.xticks(ticks = np.arange(0, 5), labels = ['Asia', 'Europe', 'North \n America', 'Oceania', 'South \n America'])
plt.ylabel("Variations (%)", fontsize = 14)
plt.xlabel("", fontsize = 1)
plt.grid(visible = True, axis = 'y')
plt.grid(visible = False, axis = 'x')
plt.savefig("dist_center_ct")


fig, ax = plt.subplots(figsize = (6, 4))
ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.yaxis.set_ticks_position('none')
ax.tick_params(axis = 'both', labelsize=14, color='#4f4e4e')
ax.grid(color='grey', axis='x', linestyle='-', linewidth=0.25, alpha=0.5)
bp1 = sns.boxplot(y = "var_modal_share_cars_CT", x = "Continent", data = df, order = ['Asia', 'Europe', 'North_America', 'Oceania', 'South_America'], boxprops = dict(linewidth=2, facecolor=(0,0,0,0), edgecolor="#fc8d62"), medianprops = dict(linewidth=2, color="#fc8d62"), whiskerprops = dict(linewidth=2, color="#fc8d62"), capprops = dict(linewidth=2, color="#fc8d62"), flierprops = colors3, showfliers=False, saturation = 1)
bp2 = sns.swarmplot(y = df.var_modal_share_cars_CT, x = df.Continent, palette = ["#404040"], order = ['Asia', 'Europe', 'North_America', 'Oceania', 'South_America'])
plt.xticks(ticks = np.arange(0, 5), labels = ['Asia', 'Europe', 'North \n America', 'Oceania', 'South \n America'])
plt.ylabel("Variations (%)", fontsize = 14)
plt.xlabel("", fontsize = 1)
plt.grid(visible = True, axis = 'y')
plt.grid(visible = False, axis = 'x')
plt.savefig("mod_share_ct")

df["var_dist_city_center_FE"] = 100 * (df.avg_dist_city_center_FE - df.avg_dist_city_center_BAU) / df.avg_dist_city_center_BAU
df["var_modal_share_cars_FE"] = 100 * (df.modal_share_cars_FE - df.modal_share_cars_BAU) / df.modal_share_cars_BAU


fig, ax = plt.subplots(figsize = (6, 4))
ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.yaxis.set_ticks_position('none')
ax.tick_params(axis = 'both', labelsize=14, color='#4f4e4e')
ax.grid(color='grey', axis='x', linestyle='-', linewidth=0.25, alpha=0.5)
bp1 = sns.boxplot(y = "var_dist_city_center_FE", x = "Continent", data = df, order = ['Asia', 'Europe', 'North_America', 'Oceania', 'South_America'], boxprops = dict(linewidth=2, facecolor=(0,0,0,0), edgecolor="#fc8d62"), medianprops = dict(linewidth=2, color="#fc8d62"), whiskerprops = dict(linewidth=2, color="#fc8d62"), capprops = dict(linewidth=2, color="#fc8d62"), flierprops = colors3, showfliers=False, saturation = 1)
bp2 = sns.swarmplot(y = df.var_dist_city_center_FE, x = df.Continent, palette = ["#404040"], order = ['Asia', 'Europe', 'North_America', 'Oceania', 'South_America'])
plt.xticks(ticks = np.arange(0, 5), labels = ['Asia', 'Europe', 'North \n America', 'Oceania', 'South \n America'])
plt.ylabel("Variations (%)", fontsize = 14)
plt.xlabel("", fontsize = 1)
plt.grid(visible = True, axis = 'y')
plt.grid(visible = False, axis = 'x')



fig, ax = plt.subplots(figsize = (6, 4))
ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.yaxis.set_ticks_position('none')
ax.tick_params(axis = 'both', labelsize=14, color='#4f4e4e')
ax.grid(color='grey', axis='x', linestyle='-', linewidth=0.25, alpha=0.5)
bp1 = sns.boxplot(y = "var_modal_share_cars_FE", x = "Continent", data = df, order = ['Asia', 'Europe', 'North_America', 'Oceania', 'South_America'], boxprops = dict(linewidth=2, facecolor=(0,0,0,0), edgecolor="#fc8d62"), medianprops = dict(linewidth=2, color="#fc8d62"), whiskerprops = dict(linewidth=2, color="#fc8d62"), capprops = dict(linewidth=2, color="#fc8d62"), flierprops = colors3, showfliers=False, saturation = 1)
bp2 = sns.swarmplot(y = df.var_modal_share_cars_FE, x = df.Continent, palette = ["#404040"], order = ['Asia', 'Europe', 'North_America', 'Oceania', 'South_America'])
plt.xticks(ticks = np.arange(0, 5), labels = ['Asia', 'Europe', 'North \n America', 'Oceania', 'South \n America'])
plt.ylabel("Variations (%)", fontsize = 14)
plt.xlabel("", fontsize = 1)
plt.grid(visible = True, axis = 'y')
plt.grid(visible = False, axis = 'x')

df["var_dist_city_center_BRT"] = 100 * (df.avg_dist_city_center_BRT - df.avg_dist_city_center_BAU) / df.avg_dist_city_center_BAU
df["var_modal_share_cars_BRT"] = 100 * (df.modal_share_cars_BRT - df.modal_share_cars_BAU) / df.modal_share_cars_BAU

fig, ax = plt.subplots(figsize = (6, 4))
ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.yaxis.set_ticks_position('none')
ax.tick_params(axis = 'both', labelsize=14, color='#4f4e4e')
ax.grid(color='grey', axis='x', linestyle='-', linewidth=0.25, alpha=0.5)
bp1 = sns.boxplot(y = "var_dist_city_center_BRT", x = "Continent", data = df, order = ['Asia', 'Europe', 'North_America', 'Oceania', 'South_America'], boxprops = dict(linewidth=2, facecolor=(0,0,0,0), edgecolor="#fc8d62"), medianprops = dict(linewidth=2, color="#fc8d62"), whiskerprops = dict(linewidth=2, color="#fc8d62"), capprops = dict(linewidth=2, color="#fc8d62"), flierprops = colors3, showfliers=False, saturation = 1)
bp2 = sns.swarmplot(y = df.var_dist_city_center_BRT, x = df.Continent, palette = ["#404040"], order = ['Asia', 'Europe', 'North_America', 'Oceania', 'South_America'])
plt.xticks(ticks = np.arange(0, 5), labels = ['Asia', 'Europe', 'North \n America', 'Oceania', 'South \n America'])
plt.ylabel("Variations (%)", fontsize = 14)
plt.xlabel("", fontsize = 1)
plt.grid(visible = True, axis = 'y')
plt.grid(visible = False, axis = 'x')


fig, ax = plt.subplots(figsize = (6, 4))
ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.yaxis.set_ticks_position('none')
ax.tick_params(axis = 'both', labelsize=14, color='#4f4e4e')
ax.grid(color='grey', axis='x', linestyle='-', linewidth=0.25, alpha=0.5)
bp1 = sns.boxplot(y = "var_modal_share_cars_BRT", x = "Continent", data = df, order = ['Asia', 'Europe', 'North_America', 'Oceania', 'South_America'], boxprops = dict(linewidth=2, facecolor=(0,0,0,0), edgecolor="#fc8d62"), medianprops = dict(linewidth=2, color="#fc8d62"), whiskerprops = dict(linewidth=2, color="#fc8d62"), capprops = dict(linewidth=2, color="#fc8d62"), flierprops = colors3, showfliers=False, saturation = 1)
bp2 = sns.swarmplot(y = df.var_modal_share_cars_BRT, x = df.Continent, palette = ["#404040"], order = ['Asia', 'Europe', 'North_America', 'Oceania', 'South_America'])
plt.xticks(ticks = np.arange(0, 5), labels = ['Asia', 'Europe', 'North \n America', 'Oceania', 'South \n America'])
plt.ylabel("Variations (%)", fontsize = 14)
plt.xlabel("", fontsize = 1)
plt.grid(visible = True, axis = 'y')
plt.grid(visible = False, axis = 'x')

df["var_dist_city_center_UGB"] = 100 * (df.avg_dist_city_center_UGB - df.avg_dist_city_center_BAU) / df.avg_dist_city_center_BAU
df["var_modal_share_cars_UGB"] = 100 * (df.modal_share_cars_UGB - df.modal_share_cars_BAU) / df.modal_share_cars_BAU


fig, ax = plt.subplots(figsize = (6, 4))
ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.yaxis.set_ticks_position('none')
ax.tick_params(axis = 'both', labelsize=14, color='#4f4e4e')
ax.grid(color='grey', axis='x', linestyle='-', linewidth=0.25, alpha=0.5)
bp1 = sns.boxplot(y = "var_dist_city_center_UGB", x = "Continent", data = df, order = ['Asia', 'Europe', 'North_America', 'Oceania', 'South_America'], boxprops = dict(linewidth=2, facecolor=(0,0,0,0), edgecolor="#fc8d62"), medianprops = dict(linewidth=2, color="#fc8d62"), whiskerprops = dict(linewidth=2, color="#fc8d62"), capprops = dict(linewidth=2, color="#fc8d62"), flierprops = colors3, showfliers=False, saturation = 1)
bp2 = sns.swarmplot(y = df.var_dist_city_center_UGB, x = df.Continent, palette = ["#404040"], order = ['Asia', 'Europe', 'North_America', 'Oceania', 'South_America'])
plt.xticks(ticks = np.arange(0, 5), labels = ['Asia', 'Europe', 'North \n America', 'Oceania', 'South \n America'])
plt.ylabel("Variations (%)", fontsize = 14)
plt.xlabel("", fontsize = 1)
plt.grid(visible = True, axis = 'y')
plt.grid(visible = False, axis = 'x')



fig, ax = plt.subplots(figsize = (6, 4))
ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.yaxis.set_ticks_position('none')
ax.tick_params(axis = 'both', labelsize=14, color='#4f4e4e')
ax.grid(color='grey', axis='x', linestyle='-', linewidth=0.25, alpha=0.5)
bp1 = sns.boxplot(y = "var_modal_share_cars_UGB", x = "Continent", data = df, order = ['Asia', 'Europe', 'North_America', 'Oceania', 'South_America'], boxprops = dict(linewidth=2, facecolor=(0,0,0,0), edgecolor="#fc8d62"), medianprops = dict(linewidth=2, color="#fc8d62"), whiskerprops = dict(linewidth=2, color="#fc8d62"), capprops = dict(linewidth=2, color="#fc8d62"), flierprops = colors3, showfliers=False, saturation = 1)
bp2 = sns.swarmplot(y = df.var_modal_share_cars_UGB, x = df.Continent, palette = ["#404040"], order = ['Asia', 'Europe', 'North_America', 'Oceania', 'South_America'])
plt.xticks(ticks = np.arange(0, 5), labels = ['Asia', 'Europe', 'North \n America', 'Oceania', 'South \n America'])
plt.ylabel("Variations (%)", fontsize = 14)
plt.xlabel("", fontsize = 1)
plt.grid(visible = True, axis = 'y')
plt.grid(visible = False, axis = 'x')


