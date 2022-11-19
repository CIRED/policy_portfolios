# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 09:44:37 2022

@author: charl
"""

import pandas as pd
import numpy as np
import copy
import pickle
import scipy as sc
from scipy import optimize
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import seaborn as sns

from calibration.calibration import *
from calibration.validation import *
from inputs.data import *
from inputs.land_use import *
from model.model import *
from outputs.outputs import *
from inputs.parameters import *
from inputs.transport import *
from density_functions import *

path_data = "C:/Users/charl/OneDrive/Bureau/City_dataStudy/"
path_folder = "C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Data/"
#path_outputs = "C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/calibration_20210415/"
list_city = list_of_cities_and_databases(path_data,'cityDatabase')#cityDatabase_intermediaire

import_data_option = 0

if import_data_option == 1:
    df = import_data_urbanized_area_reg(list_city)
else:
    df = pd.read_excel("C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/dataset_model_density.xlsx")

df["constant"] = np.ones(192)
df = df.iloc[:,1:10]

city_continent = pd.read_csv(path_folder + "CityDatabases/cityDatabase_221NewCities.csv")
city_continent = city_continent.iloc[:, [0, 2]]
city_continent = city_continent.drop_duplicates(subset = "City")
city_continent = city_continent.sort_values('City')
df = df.merge(city_continent, left_on = "city", right_on = "City")
fixed_effects = pd.get_dummies(df.Continent)
df["Asia"] = fixed_effects.Asia
df["Africa"] = fixed_effects.Africa
df["North_America"] = fixed_effects.North_America
df["Oceania"] = fixed_effects.Oceania
df["South_America"] = fixed_effects.South_America

df["log_urbanized_area"] = np.log(df.urbanized_area)
df["log_population"] = np.log(df.population)
df["log_income"] = np.log(df.income)
df["log_land_prices"] = np.log(df.land_prices)
df["log_commuting_price"] = np.log(df.commuting_price)
df["log_commuting_time"] = np.log(df.commuting_time)

informal_housing = import_informal_housing(list_city, path_folder)
df = df.merge(informal_housing.iloc[:, [0, 2]], left_on = "city", right_on = 'City')

ssp = load_ssp("OECD")

df = df.merge(ssp, left_on = "city", right_on = 'City')

results = pd.DataFrame(index = df.City, columns = ["urbanized_area_2015", "urbanized_area_reg1_2015", "urbanized_area_reg1b_2015"])
results.urbanized_area_2015 = np.array(df.urbanized_area)

### MODEL 1: POP AND INCOME

y = np.array((df.urbanized_area))
X = pd.DataFrame(df.iloc[:, [2, 3, 8]])
ols = sm.OLS(y, X)
reg1 = ols.fit(cov_type='HC1') #HC3 censé être mieux
reg1.summary()

#In-sample prediction
results.urbanized_area_reg1_2015 = np.array(reg1.predict(X))

#Out_of_sample prediction
for year in ['2050', '2100']:
    for scenario in ['SSP1', 'SSP2', 'SSP3', 'SSP4', 'SSP5']:
        X = np.column_stack((df.population * df["pop_growth_rate_" + year + "_" + scenario], df.income * df["income_growth_rate_" + year + "_" + scenario], np.ones(192)))
        results["urbanized_area_reg1_" + year + "_" + scenario] = np.array(reg1.predict(X))

### MODEL 1b: POP AND INCOME (log)

y = np.array((df.log_urbanized_area))
X = pd.DataFrame(df.iloc[:, [17, 18, 8]])
ols = sm.OLS(y, X)
reg1b = ols.fit(cov_type='HC1') #HC3 censé être mieux
reg1b.summary()

#In-sample prediction
results.urbanized_area_reg1b_2015 = np.exp(np.array(reg1b.predict(X)))

#Out_of_sample prediction
for year in ['2050', '2100']:
    for scenario in ['SSP1', 'SSP2', 'SSP3', 'SSP4', 'SSP5']:
        X = np.column_stack((np.log(df.population * df["pop_growth_rate_" + year + "_" + scenario]), np.log(df.income * df["income_growth_rate_" + year + "_" + scenario]), np.ones(192)))
        results["urbanized_area_reg1b_" + year + "_" + scenario] = np.exp(np.array(reg1b.predict(X)))


#y = np.array((df.urbanized_area))
#X = pd.DataFrame(df.iloc[:, [2, 3, 11, 12, 13, 14, 15, 8]])
#ols = sm.OLS(y, X)
#reg1_fixed_effects = ols.fit(cov_type='HC1') #HC3 censé être mieux
#reg1_fixed_effects.summary()

#y = np.array((df.log_urbanized_area))
#X = pd.DataFrame(df.iloc[:, [17, 18, 11, 12, 13, 14, 15, 8]])
#ols = sm.OLS(y, X)
#reg1b_fixed_effects = ols.fit(cov_type='HC1') #HC3 censé être mieux
#reg1b_fixed_effects.summary()

### MODEL 2: POP, INCOME, AGRI RENT, TRANSPORTATION COSTS, POLYCENTRICITY

y = np.array((df.urbanized_area))
X = pd.DataFrame(df.iloc[:, [2, 3, 4, 5, 6, 7, 8]])
ols = sm.OLS(y, X)
reg2 = ols.fit(cov_type='HC1') #HC3 censé être mieux
reg2.summary()

#In-sample prediction
results["urbanized_area_reg2_2015"] = np.array(reg2.predict(X))

#Out_of_sample prediction
for year in ['2050', '2100']:
    for scenario in ['SSP1', 'SSP2', 'SSP3', 'SSP4', 'SSP5']:
        X = np.column_stack((df.population * df["pop_growth_rate_" + year + "_" + scenario], df.income * df["income_growth_rate_" + year + "_" + scenario], df.land_prices, df.commuting_price, df.commuting_time, df.polycentricity, np.ones(192)))
        results["urbanized_area_reg2_" + year + "_" + scenario] = np.array(reg2.predict(X))

### MODEL 2b: POP, INCOME, AGRI RENT, TRANSPORTATION COSTS, POLYCENTRICITY (log)

y = np.array((df.log_urbanized_area))
X = pd.DataFrame(df.iloc[:, [17, 18, 19, 20, 21, 7, 8]])
ols = sm.OLS(y, X)
reg2b = ols.fit(cov_type='HC1') #HC3 censé être mieux
reg2b.summary()

#In-sample prediction
results["urbanized_area_reg2b_2015"] = np.exp(np.array(reg2b.predict(X)))

#Out_of_sample prediction
for year in ['2050', '2100']:
    for scenario in ['SSP1', 'SSP2', 'SSP3', 'SSP4', 'SSP5']:
        X = np.column_stack((np.log(df.population * df["pop_growth_rate_" + year + "_" + scenario]), np.log(df.income * df["income_growth_rate_" + year + "_" + scenario]), np.log(df.land_prices), np.log(df.commuting_price), np.log(df.commuting_time), df.polycentricity, np.ones(192)))
        results["urbanized_area_reg2b_" + year + "_" + scenario] = np.exp(np.array(reg2b.predict(X)))

### MODEL 3: POP, INCOME, AGRI RENT, TRANSPORTATION COSTS, POLYCENTRICITY, INFORMALITY

y = np.array((df.urbanized_area))
X = pd.DataFrame(df.iloc[:, [2, 3, 4, 5, 6, 7, 23, 8]])
ols = sm.OLS(y, X)
reg3 = ols.fit(cov_type='HC1') #HC3 censé être mieux
reg3.summary()

#In-sample prediction
results["urbanized_area_reg3_2015"] = np.array(reg3.predict(X))

#Out_of_sample prediction
for year in ['2050', '2100']:
    for scenario in ['SSP1', 'SSP2', 'SSP3', 'SSP4', 'SSP5']:
        X = np.column_stack((df.population * df["pop_growth_rate_" + year + "_" + scenario], df.income * df["income_growth_rate_" + year + "_" + scenario], df.land_prices, df.commuting_price, df.commuting_time, df.polycentricity, df.informal_housing, np.ones(192)))
        results["urbanized_area_reg3_" + year + "_" + scenario] = np.array(reg3.predict(X))

### MODEL 3b: POP, INCOME, AGRI RENT, TRANSPORTATION COSTS, POLYCENTRICITY, INFORMALITY (log)

y = np.array((df.log_urbanized_area))
X = pd.DataFrame(df.iloc[:, [17, 18, 19, 20, 21, 7, 23, 8]])
ols = sm.OLS(y, X)
reg3b = ols.fit(cov_type='HC1') #HC3 censé être mieux
reg3b.summary()

#In-sample prediction
results["urbanized_area_reg3b_2015"] = np.exp(np.array(reg3b.predict(X)))

#Out_of_sample prediction
for year in ['2050', '2100']:
    for scenario in ['SSP1', 'SSP2', 'SSP3', 'SSP4', 'SSP5']:
        X = np.column_stack((np.log(df.population * df["pop_growth_rate_" + year + "_" + scenario]), np.log(df.income * df["income_growth_rate_" + year + "_" + scenario]), np.log(df.land_prices), np.log(df.commuting_price), np.log(df.commuting_time), df.polycentricity, df.informal_housing, np.ones(192)))
        results["urbanized_area_reg3b_" + year + "_" + scenario] = np.exp(np.array(reg3b.predict(X)))

#Model 4
results["urbanized_area_reg4_2015"] = np.nan
results["urbanized_area_reg4_2050_SSP1"] = np.nan
results["urbanized_area_reg4_2050_SSP2"] = np.nan
results["urbanized_area_reg4_2050_SSP3"] = np.nan
results["urbanized_area_reg4_2050_SSP4"] = np.nan
results["urbanized_area_reg4_2050_SSP5"] = np.nan
results["urbanized_area_reg4_2100_SSP1"] = np.nan
results["urbanized_area_reg4_2100_SSP2"] = np.nan
results["urbanized_area_reg4_2100_SSP3"] = np.nan
results["urbanized_area_reg4_2100_SSP4"] = np.nan
results["urbanized_area_reg4_2100_SSP5"] = np.nan
path_ssp1 = "C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/ssp1_20220216/"
path_ssp2 = "C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/ssp2_20220216/"
path_ssp3 = "C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/ssp3_20220216/"
path_ssp4 = "C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/ssp4_20220216/"
path_ssp5 = "C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/ssp5_20220216/"

for city in np.delete(np.unique(results.index), 153):
    results.loc[results.index == city, 'urbanized_area_reg4_2015'] = np.load(path_ssp1 + city + "_urbanized_area.npy")[0]
    results.loc[results.index == city, 'urbanized_area_reg4_2050_SSP1'] = np.load(path_ssp1 + city + "_urbanized_area.npy")[35]
    results.loc[results.index == city, 'urbanized_area_reg4_2050_SSP2'] = np.load(path_ssp2 + city + "_urbanized_area.npy")[35]
    results.loc[results.index == city, 'urbanized_area_reg4_2050_SSP3'] = np.load(path_ssp3 + city + "_urbanized_area.npy")[35]
    results.loc[results.index == city, 'urbanized_area_reg4_2050_SSP4'] = np.load(path_ssp4 + city + "_urbanized_area.npy")[35]
    results.loc[results.index == city, 'urbanized_area_reg4_2050_SSP5'] = np.load(path_ssp5 + city + "_urbanized_area.npy")[35]
    results.loc[results.index == city, 'urbanized_area_reg4_2100_SSP1'] = np.load(path_ssp1 + city + "_urbanized_area.npy")[85]
    results.loc[results.index == city, 'urbanized_area_reg4_2100_SSP2'] = np.load(path_ssp2 + city + "_urbanized_area.npy")[85]
    results.loc[results.index == city, 'urbanized_area_reg4_2100_SSP3'] = np.load(path_ssp3 + city + "_urbanized_area.npy")[85]
    results.loc[results.index == city, 'urbanized_area_reg4_2100_SSP4'] = np.load(path_ssp4 + city + "_urbanized_area.npy")[85]
    results.loc[results.index == city, 'urbanized_area_reg4_2100_SSP5'] = np.load(path_ssp5 + city + "_urbanized_area.npy")[85]
    
sample_of_cities = pd.DataFrame(columns = ['City', 'criterion1', 'criterion2', 'criterion3', 'final_sample'], index = np.unique(list_city.City))
sample_of_cities.City = sample_of_cities.index
path_data = "C:/Users/charl/OneDrive/Bureau/City_dataStudy/"
path_folder = "C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Data/"
path_calibration = "C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/calibration_20211124/"
path_BAU = "C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/BAU_20211124/"
list_city = list_of_cities_and_databases(path_data,'cityDatabase')#cityDatabase_intermediaire

#Criterion 1: selected cells
selected_cells = np.load(path_calibration + "d_selected_cells.npy", allow_pickle = True)
selected_cells = np.array(selected_cells, ndmin = 1)[0]
for city in list(np.delete(sample_of_cities.index, 153)):
    if (selected_cells[city] > 1):
        sample_of_cities.loc[city, "criterion1"] = 1
    elif (selected_cells[city] == 1):
        sample_of_cities.loc[city, "criterion1"] = 0
print("Number of cities excluded because of criterion 1:", sum(sample_of_cities.criterion1 == 0))

#Criterion 2: housing budget exceeds income

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
    size[size > 1000] = np.nan
    income = pd.read_excel(path_folder + "income.xlsx").income[pd.read_excel(path_folder + "income.xlsx").city == city].squeeze()   
    share_housing = rent * size / income
    if weighted_percentile(np.array(share_housing)[~np.isnan(density) & ~np.isnan(share_housing)], 80, weights=density[~np.isnan(density) & ~np.isnan(share_housing)]) > 1:
        sample_of_cities.loc[city, ['criterion2']] = 0
    else:
        sample_of_cities.loc[city, ['criterion2']] = 1

print("Number of cities excluded because of criterion 2:", sum(sample_of_cities.criterion2 == 0))

#Criterion 3: bad fit on density and rents

d_corr_density_scells2 = np.load(path_calibration + "d_corr_density_scells2.npy", allow_pickle = True)
d_corr_density_scells2 = np.array(d_corr_density_scells2, ndmin = 1)[0]
d_corr_rent_scells2 = np.load(path_calibration + "d_corr_rent_scells2.npy", allow_pickle = True)
d_corr_rent_scells2 = np.array(d_corr_rent_scells2, ndmin = 1)[0]

for city in list(sample_of_cities.index[sample_of_cities.criterion1 == 1 ]):
    if (d_corr_density_scells2[city][0] < 0) | (d_corr_rent_scells2[city][0] < 0):
        sample_of_cities.loc[city, "criterion3"] = 0
    else:
        sample_of_cities.loc[city, "criterion3"] = 1
        
print("Number of cities excluded because of criterion 3:", sum((sample_of_cities.criterion3 == 0) & (sample_of_cities.criterion2 == 1)))

sample_of_cities.final_sample[(sample_of_cities.criterion1 == 0) | (sample_of_cities.criterion2 == 0) | (sample_of_cities.criterion3 == 0)] = 0
sample_of_cities.final_sample[(sample_of_cities.criterion1 == 1) & (sample_of_cities.criterion2 == 1) & (sample_of_cities.criterion3 == 1)] = 1

print("Number of cities kept in the end:", sum(sample_of_cities.final_sample == 1))

results = results.merge(sample_of_cities.loc[:, ["City", "final_sample"]], left_index = True, right_on = "City")
results["urbanized_area_reg4_2015"].loc[results.final_sample == 0] = np.nan
results["urbanized_area_reg4_2050_SSP1"].loc[results.final_sample == 0] = np.nan
results["urbanized_area_reg4_2050_SSP2"].loc[results.final_sample == 0] = np.nan
results["urbanized_area_reg4_2050_SSP3"].loc[results.final_sample == 0] = np.nan
results["urbanized_area_reg4_2050_SSP4"].loc[results.final_sample == 0] = np.nan
results["urbanized_area_reg4_2050_SSP5"].loc[results.final_sample == 0] = np.nan
results["urbanized_area_reg4_2100_SSP1"].loc[results.final_sample == 0] = np.nan
results["urbanized_area_reg4_2100_SSP2"].loc[results.final_sample == 0] = np.nan
results["urbanized_area_reg4_2100_SSP3"].loc[results.final_sample == 0] = np.nan
results["urbanized_area_reg4_2100_SSP4"].loc[results.final_sample == 0] = np.nan
results["urbanized_area_reg4_2100_SSP5"].loc[results.final_sample == 0] = np.nan

results["urbanized_area_reg2b_2015"].loc[results.final_sample == 0] = np.nan
results["urbanized_area_reg2b_2050_SSP1"].loc[results.final_sample == 0] = np.nan
results["urbanized_area_reg2b_2050_SSP2"].loc[results.final_sample == 0] = np.nan
results["urbanized_area_reg2b_2050_SSP3"].loc[results.final_sample == 0] = np.nan
results["urbanized_area_reg2b_2050_SSP4"].loc[results.final_sample == 0] = np.nan
results["urbanized_area_reg2b_2050_SSP5"].loc[results.final_sample == 0] = np.nan
results["urbanized_area_reg2b_2100_SSP1"].loc[results.final_sample == 0] = np.nan
results["urbanized_area_reg2b_2100_SSP2"].loc[results.final_sample == 0] = np.nan
results["urbanized_area_reg2b_2100_SSP3"].loc[results.final_sample == 0] = np.nan
results["urbanized_area_reg2b_2100_SSP4"].loc[results.final_sample == 0] = np.nan
results["urbanized_area_reg2b_2100_SSP5"].loc[results.final_sample == 0] = np.nan

### CHARTS

plt.rcParams['figure.dpi'] = 360
sns.set(style="whitegrid")
fig, ax = plt.subplots(figsize=(6,4))
plt.xlabel('')
plt.ylabel('')
#plt.title("Reg 1", fontsize=12, color='#4f4e4e')
sns.despine(left=True, right = True, top = True)
sns.barplot(np.array([0, 1, 2, 3, 4]), np.repeat(sum(results.urbanized_area_reg1_2015), 5), palette=['#838B8B'], label = "Data - 2015")
sns.barplot(np.array([0, 1, 2, 3, 4]), np.array([sum(results.urbanized_area_reg1_2050_SSP1), sum(results.urbanized_area_reg1_2050_SSP2), sum(results.urbanized_area_reg1_2050_SSP3), sum(results.urbanized_area_reg1_2050_SSP4), sum(results.urbanized_area_reg1_2050_SSP5)]) - np.repeat(sum(results.urbanized_area_reg1_2015), 5), bottom = np.repeat(sum(results.urbanized_area_reg1_2015), 5), palette = ['#FF0000'], label = "Simulations - 2050")
sns.barplot(np.array([0, 1, 2, 3, 4]), np.array([sum(results.urbanized_area_reg1_2100_SSP1), sum(results.urbanized_area_reg1_2100_SSP2), sum(results.urbanized_area_reg1_2100_SSP3), sum(results.urbanized_area_reg1_2100_SSP4), sum(results.urbanized_area_reg1_2100_SSP5)]) - np.array([sum(results.urbanized_area_reg1_2050_SSP1), sum(results.urbanized_area_reg1_2050_SSP2), sum(results.urbanized_area_reg1_2050_SSP3), sum(results.urbanized_area_reg1_2050_SSP4), sum(results.urbanized_area_reg1_2050_SSP5)]), bottom = np.array([sum(results.urbanized_area_reg1_2050_SSP1), sum(results.urbanized_area_reg1_2050_SSP2), sum(results.urbanized_area_reg1_2050_SSP3), sum(results.urbanized_area_reg1_2050_SSP4), sum(results.urbanized_area_reg1_2050_SSP5)]), palette = ['#CD0000'], label = "Simulations - 2100")
plt.xticks(np.array([0, 1, 2, 3, 4]), ["SSP1", "SSP2", "SSP3", "SSP4", "SSP5"])
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles[0:], labels=labels[0:], title = '', bbox_to_anchor=(0.3, 0.95), fontsize = 12, loc=2, borderaxespad=0.)
ax.tick_params(axis = 'both', labelsize=12, color='#4f4e4e')

plt.rcParams['figure.dpi'] = 360
sns.set(style="whitegrid")
fig, ax = plt.subplots(figsize=(6,4))
plt.xlabel('')
plt.ylabel('')
#plt.title("Reg 1b", fontsize=12, color='#4f4e4e')
sns.despine(left=True, right = True, top = True)
sns.barplot(np.array([0, 1, 2, 3, 4]), np.repeat(sum(results.urbanized_area_reg1b_2015), 5), palette=['#838B8B'], label = "Data - 2015")
sns.barplot(np.array([0, 1, 2, 3, 4]), np.array([sum(results.urbanized_area_reg1b_2050_SSP1), sum(results.urbanized_area_reg1b_2050_SSP2), sum(results.urbanized_area_reg1b_2050_SSP3), sum(results.urbanized_area_reg1b_2050_SSP4), sum(results.urbanized_area_reg1b_2050_SSP5)]) - np.repeat(sum(results.urbanized_area_reg1b_2015), 5), bottom = np.repeat(sum(results.urbanized_area_reg1b_2015), 5), palette = ['#FF0000'], label = "Simulations - 2050")
sns.barplot(np.array([0, 1, 2, 3, 4]), np.array([sum(results.urbanized_area_reg1b_2100_SSP1), sum(results.urbanized_area_reg1b_2100_SSP2), sum(results.urbanized_area_reg1b_2100_SSP3), sum(results.urbanized_area_reg1b_2100_SSP4), sum(results.urbanized_area_reg1b_2100_SSP5)]) - np.array([sum(results.urbanized_area_reg1b_2050_SSP1), sum(results.urbanized_area_reg1b_2050_SSP2), sum(results.urbanized_area_reg1b_2050_SSP3), sum(results.urbanized_area_reg1b_2050_SSP4), sum(results.urbanized_area_reg1b_2050_SSP5)]), bottom = np.array([sum(results.urbanized_area_reg1b_2050_SSP1), sum(results.urbanized_area_reg1b_2050_SSP2), sum(results.urbanized_area_reg1b_2050_SSP3), sum(results.urbanized_area_reg1b_2050_SSP4), sum(results.urbanized_area_reg1b_2050_SSP5)]), palette = ['#CD0000'], label = "Simulations - 2100")
plt.xticks(np.array([0, 1, 2, 3, 4]), ["SSP1", "SSP2", "SSP3", "SSP4", "SSP5"])
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles[0:], labels=labels[0:], title = '', bbox_to_anchor=(0.3, 0.95), fontsize = 12, loc=2, borderaxespad=0.)
ax.tick_params(axis = 'both', labelsize=12, color='#4f4e4e')


plt.rcParams['figure.dpi'] = 360
sns.set(style="whitegrid")
fig, ax = plt.subplots(figsize=(6,4))
plt.xlabel('')
plt.ylabel('')
#plt.title("Reg 2", fontsize=12, color='#4f4e4e')
sns.despine(left=True, right = True, top = True)
sns.barplot(np.array([0, 1, 2, 3, 4]), np.repeat(sum(results.urbanized_area_reg2_2015), 5), palette=['#838B8B'], label = "Data - 2015")
sns.barplot(np.array([0, 1, 2, 3, 4]), np.array([sum(results.urbanized_area_reg2_2050_SSP1), sum(results.urbanized_area_reg2_2050_SSP2), sum(results.urbanized_area_reg2_2050_SSP3), sum(results.urbanized_area_reg2_2050_SSP4), sum(results.urbanized_area_reg2_2050_SSP5)]) - np.repeat(sum(results.urbanized_area_reg2_2015), 5), bottom = np.repeat(sum(results.urbanized_area_reg2_2015), 5), palette = ['#FF0000'], label = "Simulations - 2050")
sns.barplot(np.array([0, 1, 2, 3, 4]), np.array([sum(results.urbanized_area_reg2_2100_SSP1), sum(results.urbanized_area_reg2_2100_SSP2), sum(results.urbanized_area_reg2_2100_SSP3), sum(results.urbanized_area_reg2_2100_SSP4), sum(results.urbanized_area_reg2_2100_SSP5)]) - np.array([sum(results.urbanized_area_reg2_2050_SSP1), sum(results.urbanized_area_reg2_2050_SSP2), sum(results.urbanized_area_reg2_2050_SSP3), sum(results.urbanized_area_reg2_2050_SSP4), sum(results.urbanized_area_reg2_2050_SSP5)]), bottom = np.array([sum(results.urbanized_area_reg2_2050_SSP1), sum(results.urbanized_area_reg2_2050_SSP2), sum(results.urbanized_area_reg2_2050_SSP3), sum(results.urbanized_area_reg2_2050_SSP4), sum(results.urbanized_area_reg2_2050_SSP5)]), palette = ['#CD0000'], label = "Simulations - 2100")
plt.xticks(np.array([0, 1, 2, 3, 4]), ["SSP1", "SSP2", "SSP3", "SSP4", "SSP5"])
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles[0:], labels=labels[0:], title = '', bbox_to_anchor=(0.3, 0.95), fontsize = 12, loc=2, borderaxespad=0.)
ax.tick_params(axis = 'both', labelsize=12, color='#4f4e4e')

plt.rcParams['figure.dpi'] = 360
sns.set(style="whitegrid")
fig, ax = plt.subplots(figsize=(6,4))
plt.xlabel('')
plt.ylabel('')
#plt.title("Reg 2b", fontsize=12, color='#4f4e4e')
sns.despine(left=True, right = True, top = True)
sns.barplot(np.array([0, 1, 2, 3, 4]), np.repeat(np.nansum(results.urbanized_area_reg2b_2015), 5), palette=['#838B8B'], label = "Data - 2015")
sns.barplot(np.array([0, 1, 2, 3, 4]), np.array([np.nansum(results.urbanized_area_reg2b_2050_SSP1), np.nansum(results.urbanized_area_reg2b_2050_SSP2), np.nansum(results.urbanized_area_reg2b_2050_SSP3), np.nansum(results.urbanized_area_reg2b_2050_SSP4), np.nansum(results.urbanized_area_reg2b_2050_SSP5)]) - np.repeat(np.nansum(results.urbanized_area_reg2b_2015), 5), bottom = np.repeat(np.nansum(results.urbanized_area_reg2b_2015), 5), palette = ['#FF0000'], label = "Simulations - 2050")
sns.barplot(np.array([0, 1, 2, 3, 4]), np.array([np.nansum(results.urbanized_area_reg2b_2100_SSP1), np.nansum(results.urbanized_area_reg2b_2100_SSP2), np.nansum(results.urbanized_area_reg2b_2100_SSP3), np.nansum(results.urbanized_area_reg2b_2100_SSP4), np.nansum(results.urbanized_area_reg2b_2100_SSP5)]) - np.array([np.nansum(results.urbanized_area_reg2b_2050_SSP1), np.nansum(results.urbanized_area_reg2b_2050_SSP2), np.nansum(results.urbanized_area_reg2b_2050_SSP3), np.nansum(results.urbanized_area_reg2b_2050_SSP4), np.nansum(results.urbanized_area_reg2b_2050_SSP5)]), bottom = np.array([np.nansum(results.urbanized_area_reg2b_2050_SSP1), np.nansum(results.urbanized_area_reg2b_2050_SSP2), np.nansum(results.urbanized_area_reg2b_2050_SSP3), np.nansum(results.urbanized_area_reg2b_2050_SSP4), np.nansum(results.urbanized_area_reg2b_2050_SSP5)]), palette = ['#CD0000'], label = "Simulations - 2100")
plt.xticks(np.array([0, 1, 2, 3, 4]), ["SSP1", "SSP2", "SSP3", "SSP4", "SSP5"])
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles[0:], labels=labels[0:], title = '', bbox_to_anchor=(0.3, 0.95), fontsize = 12, loc=2, borderaxespad=0.)
ax.tick_params(axis = 'both', labelsize=12, color='#4f4e4e')

plt.rcParams['figure.dpi'] = 360
sns.set(style="whitegrid")
fig, ax = plt.subplots(figsize=(6,4))
plt.xlabel('')
plt.ylabel('')
#plt.title("Reg 3", fontsize=12, color='#4f4e4e')
sns.despine(left=True, right = True, top = True)
sns.barplot(np.array([0, 1, 2, 3, 4]), np.repeat(sum(results.urbanized_area_reg3_2015), 5), palette=['#838B8B'], label = "Data - 2015")
sns.barplot(np.array([0, 1, 2, 3, 4]), np.array([sum(results.urbanized_area_reg3_2050_SSP1), sum(results.urbanized_area_reg3_2050_SSP2), sum(results.urbanized_area_reg3_2050_SSP3), sum(results.urbanized_area_reg3_2050_SSP4), sum(results.urbanized_area_reg3_2050_SSP5)]) - np.repeat(sum(results.urbanized_area_reg3_2015), 5), bottom = np.repeat(sum(results.urbanized_area_reg3_2015), 5), palette = ['#FF0000'], label = "Simulations - 2050")
sns.barplot(np.array([0, 1, 2, 3, 4]), np.array([sum(results.urbanized_area_reg3_2100_SSP1), sum(results.urbanized_area_reg3_2100_SSP2), sum(results.urbanized_area_reg3_2100_SSP3), sum(results.urbanized_area_reg3_2100_SSP4), sum(results.urbanized_area_reg3_2100_SSP5)]) - np.array([sum(results.urbanized_area_reg3_2050_SSP1), sum(results.urbanized_area_reg3_2050_SSP2), sum(results.urbanized_area_reg3_2050_SSP3), sum(results.urbanized_area_reg3_2050_SSP4), sum(results.urbanized_area_reg3_2050_SSP5)]), bottom = np.array([sum(results.urbanized_area_reg3_2050_SSP1), sum(results.urbanized_area_reg3_2050_SSP2), sum(results.urbanized_area_reg3_2050_SSP3), sum(results.urbanized_area_reg3_2050_SSP4), sum(results.urbanized_area_reg3_2050_SSP5)]), palette = ['#CD0000'], label = "Simulations - 2100")
plt.xticks(np.array([0, 1, 2, 3, 4]), ["SSP1", "SSP2", "SSP3", "SSP4", "SSP5"])
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles[0:], labels=labels[0:], title = '', bbox_to_anchor=(0.3, 0.95), fontsize = 12, loc=2, borderaxespad=0.)
ax.tick_params(axis = 'both', labelsize=12, color='#4f4e4e')

plt.rcParams['figure.dpi'] = 360
sns.set(style="whitegrid")
fig, ax = plt.subplots(figsize=(6,4))
plt.xlabel('')
plt.ylabel('')
#plt.title("Reg 3b", fontsize=12, color='#4f4e4e')
sns.despine(left=True, right = True, top = True)
sns.barplot(np.array([0, 1, 2, 3, 4]), np.repeat(sum(results.urbanized_area_reg3b_2015), 5), palette=['#838B8B'], label = "Data - 2015")
sns.barplot(np.array([0, 1, 2, 3, 4]), np.array([sum(results.urbanized_area_reg3b_2050_SSP1), sum(results.urbanized_area_reg3b_2050_SSP2), sum(results.urbanized_area_reg3b_2050_SSP3), sum(results.urbanized_area_reg3b_2050_SSP4), sum(results.urbanized_area_reg3b_2050_SSP5)]) - np.repeat(sum(results.urbanized_area_reg3b_2015), 5), bottom = np.repeat(sum(results.urbanized_area_reg3b_2015), 5), palette = ['#FF0000'], label = "Simulations - 2050")
sns.barplot(np.array([0, 1, 2, 3, 4]), np.array([sum(results.urbanized_area_reg3b_2100_SSP1), sum(results.urbanized_area_reg3b_2100_SSP2), sum(results.urbanized_area_reg3b_2100_SSP3), sum(results.urbanized_area_reg3b_2100_SSP4), sum(results.urbanized_area_reg3b_2100_SSP5)]) - np.array([sum(results.urbanized_area_reg3b_2050_SSP1), sum(results.urbanized_area_reg3b_2050_SSP2), sum(results.urbanized_area_reg3b_2050_SSP3), sum(results.urbanized_area_reg3b_2050_SSP4), sum(results.urbanized_area_reg3b_2050_SSP5)]), bottom = np.array([sum(results.urbanized_area_reg3b_2050_SSP1), sum(results.urbanized_area_reg3b_2050_SSP2), sum(results.urbanized_area_reg3b_2050_SSP3), sum(results.urbanized_area_reg3b_2050_SSP4), sum(results.urbanized_area_reg3b_2050_SSP5)]), palette = ['#CD0000'], label = "Simulations - 2100")
plt.xticks(np.array([0, 1, 2, 3, 4]), ["SSP1", "SSP2", "SSP3", "SSP4", "SSP5"])
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles[0:], labels=labels[0:], title = '', bbox_to_anchor=(0.3, 0.95), fontsize = 12, loc=2, borderaxespad=0.)
ax.tick_params(axis = 'both', labelsize=12, color='#4f4e4e')

plt.rcParams['figure.dpi'] = 360
sns.set(style="whitegrid")
fig, ax = plt.subplots(figsize=(6,4))
plt.xlabel('')
plt.ylabel('')
#plt.title("Reg 4", fontsize=12, color='#4f4e4e')
sns.despine(left=True, right = True, top = True)
sns.barplot(np.array([0, 1, 2, 3, 4]), np.repeat(np.nansum(results.urbanized_area_reg4_2015), 5), palette=['#838B8B'], label = "Data - 2015")
sns.barplot(np.array([0, 1, 2, 3, 4]), np.array([np.nansum(results.urbanized_area_reg4_2050_SSP1), np.nansum(results.urbanized_area_reg4_2050_SSP2), np.nansum(results.urbanized_area_reg4_2050_SSP3), np.nansum(results.urbanized_area_reg4_2050_SSP4), np.nansum(results.urbanized_area_reg4_2050_SSP5)]) - np.repeat(np.nansum(results.urbanized_area_reg4_2015), 5), bottom = np.repeat(np.nansum(results.urbanized_area_reg4_2015), 5), palette = ['#FF0000'], label = "Simulations - 2050")
sns.barplot(np.array([0, 1, 2, 3, 4]), np.array([np.nansum(results.urbanized_area_reg4_2100_SSP1), np.nansum(results.urbanized_area_reg4_2100_SSP2), np.nansum(results.urbanized_area_reg4_2100_SSP3), np.nansum(results.urbanized_area_reg4_2100_SSP4), np.nansum(results.urbanized_area_reg4_2100_SSP5)]) - np.array([np.nansum(results.urbanized_area_reg4_2050_SSP1), np.nansum(results.urbanized_area_reg4_2050_SSP2), np.nansum(results.urbanized_area_reg4_2050_SSP3), np.nansum(results.urbanized_area_reg4_2050_SSP4), np.nansum(results.urbanized_area_reg4_2050_SSP5)]), bottom = np.array([np.nansum(results.urbanized_area_reg4_2050_SSP1), np.nansum(results.urbanized_area_reg4_2050_SSP2), np.nansum(results.urbanized_area_reg4_2050_SSP3), np.nansum(results.urbanized_area_reg4_2050_SSP4), np.nansum(results.urbanized_area_reg4_2050_SSP5)]), palette = ['#CD0000'], label = "Simulations - 2100")
plt.xticks(np.array([0, 1, 2, 3, 4]), ["SSP1", "SSP2", "SSP3", "SSP4", "SSP5"])
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles[0:], labels=labels[0:], title = '', bbox_to_anchor=(0.3, 0.95), fontsize = 12, loc=2, borderaxespad=0.)
ax.tick_params(axis = 'both', labelsize=12, color='#4f4e4e')

### STATS DES BY CONTINENT

city_continent = pd.read_csv(path_folder + "CityDatabases/cityDatabase_221NewCities.csv")
city_continent = city_continent.iloc[:, [0, 2]]
city_continent = city_continent.drop_duplicates(subset = "City")
city_continent = city_continent.sort_values('City')
results = results.merge(city_continent, left_index = True, right_on = "City")

summary_by_continent = results.groupby('Continent').sum()

summary_by_continent.urbanized_area_reg4_2050_SSP1 / summary_by_continent.urbanized_area_reg4_2015
summary_by_continent.urbanized_area_reg4_2050_SSP2 / summary_by_continent.urbanized_area_reg4_2015
summary_by_continent.urbanized_area_reg4_2050_SSP3 / summary_by_continent.urbanized_area_reg4_2015
summary_by_continent.urbanized_area_reg4_2050_SSP4 / summary_by_continent.urbanized_area_reg4_2015
summary_by_continent.urbanized_area_reg4_2050_SSP5 / summary_by_continent.urbanized_area_reg4_2015

summary_by_continent.urbanized_area_reg2b_2050_SSP1 / summary_by_continent.urbanized_area_reg2b_2015
summary_by_continent.urbanized_area_reg2b_2050_SSP2 / summary_by_continent.urbanized_area_reg2b_2015
summary_by_continent.urbanized_area_reg2b_2050_SSP3 / summary_by_continent.urbanized_area_reg2b_2015
summary_by_continent.urbanized_area_reg2b_2050_SSP4 / summary_by_continent.urbanized_area_reg2b_2015
summary_by_continent.urbanized_area_reg2b_2050_SSP5 / summary_by_continent.urbanized_area_reg2b_2015

### COMPARISON WITH OTHER DATABASES

#Cumulé - Gao

gao_ssp1 = pd.read_csv("C:/Users/charl/OneDrive/Bureau/Urban sprawl/gao_2020/NationalTotalUrbanArea_SSP1_2000-2100_v1.csv")
gao_ssp2 = pd.read_csv("C:/Users/charl/OneDrive/Bureau/Urban sprawl/gao_2020/NationalTotalUrbanArea_SSP2_2000-2100_v1.csv")
gao_ssp3 = pd.read_csv("C:/Users/charl/OneDrive/Bureau/Urban sprawl/gao_2020/NationalTotalUrbanArea_SSP3_2000-2100_v1.csv")
gao_ssp4 = pd.read_csv("C:/Users/charl/OneDrive/Bureau/Urban sprawl/gao_2020/NationalTotalUrbanArea_SSP4_2000-2100_v1.csv")
gao_ssp5 = pd.read_csv("C:/Users/charl/OneDrive/Bureau/Urban sprawl/gao_2020/NationalTotalUrbanArea_SSP5_2000-2100_v1.csv")

plt.rcParams['figure.dpi'] = 360
sns.set(style="whitegrid")
fig, ax = plt.subplots(figsize=(6,4))
plt.xlabel('')
plt.ylabel('')
#plt.title("Gao and O'Neill (2020)", fontsize=12, color='#4f4e4e')
sns.despine(left=True, right = True, top = True)
sns.barplot(np.array([0, 1, 2, 3, 4]), np.repeat(sum(gao_ssp1.UrbAmt2010), 5), palette=['#838B8B'], label = "Data - 2010")
sns.barplot(np.array([0, 1, 2, 3, 4]), np.array([sum(gao_ssp1.UrbAmt2050), sum(gao_ssp2.UrbAmt2050), sum(gao_ssp3.UrbAmt2050), sum(gao_ssp4.UrbAmt2050), sum(gao_ssp5.UrbAmt2050)]) - np.repeat(sum(gao_ssp1.UrbAmt2010), 5), bottom = np.repeat(sum(gao_ssp1.UrbAmt2010), 5), palette = ['#FF0000'], label = "Simulations - 2050")
sns.barplot(np.array([0, 1, 2, 3, 4]), np.array([sum(gao_ssp1.UrbAmt2100), sum(gao_ssp2.UrbAmt2100), sum(gao_ssp3.UrbAmt2100), sum(gao_ssp4.UrbAmt2100), sum(gao_ssp5.UrbAmt2100)]) - np.array([sum(gao_ssp1.UrbAmt2050), sum(gao_ssp2.UrbAmt2050), sum(gao_ssp3.UrbAmt2050), sum(gao_ssp4.UrbAmt2050), sum(gao_ssp5.UrbAmt2050)]), bottom = np.array([sum(gao_ssp1.UrbAmt2050), sum(gao_ssp2.UrbAmt2050), sum(gao_ssp3.UrbAmt2050), sum(gao_ssp4.UrbAmt2050), sum(gao_ssp5.UrbAmt2050)]), palette = ['#CD0000'], label = "Simulations - 2100")
plt.xticks(np.array([0, 1, 2, 3, 4]), ["SSP1", "SSP2", "SSP3", "SSP4", "SSP5"])
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles[0:], labels=labels[0:], title = '', bbox_to_anchor=(0.3, 0.95), fontsize = 12, loc=2, borderaxespad=0.)
ax.tick_params(axis = 'both', labelsize=12, color='#4f4e4e')

#Stats des by continent
iso_continent = pd.read_excel("C:/Users/charl/OneDrive/Bureau/Urban sprawl/iso_continent.xlsx")

gap_ssp1_continent = gao_ssp1.merge(iso_continent, on = "ISO3v10").groupby('Continent').sum()
gap_ssp1_continent["UrbAmt2050"] / gap_ssp1_continent["UrbAmt2020"]

gap_ssp2_continent = gao_ssp2.merge(iso_continent, on = "ISO3v10").groupby('Continent').sum()
gap_ssp2_continent["UrbAmt2050"] / gap_ssp2_continent["UrbAmt2020"]

gap_ssp3_continent = gao_ssp3.merge(iso_continent, on = "ISO3v10").groupby('Continent').sum()
gap_ssp3_continent["UrbAmt2050"] / gap_ssp3_continent["UrbAmt2020"]

gap_ssp4_continent = gao_ssp4.merge(iso_continent, on = "ISO3v10").groupby('Continent').sum()
gap_ssp4_continent["UrbAmt2050"] / gap_ssp4_continent["UrbAmt2020"]

gap_ssp5_continent = gao_ssp5.merge(iso_continent, on = "ISO3v10").groupby('Continent').sum()
gap_ssp5_continent["UrbAmt2050"] / gap_ssp5_continent["UrbAmt2020"]

#Cumulé - Li

li_ssp1 = pd.read_excel("C:/Users/charl/OneDrive/Bureau/Urban sprawl/li_2020.xlsx", sheet_name = 'SSP1')
li_ssp2 = pd.read_excel("C:/Users/charl/OneDrive/Bureau/Urban sprawl/li_2020.xlsx", sheet_name = 'SSP2')
li_ssp3 = pd.read_excel("C:/Users/charl/OneDrive/Bureau/Urban sprawl/li_2020.xlsx", sheet_name = 'SSP3')
li_ssp4 = pd.read_excel("C:/Users/charl/OneDrive/Bureau/Urban sprawl/li_2020.xlsx", sheet_name = 'SSP4')
li_ssp5 = pd.read_excel("C:/Users/charl/OneDrive/Bureau/Urban sprawl/li_2020.xlsx", sheet_name = 'SSP5')

plt.rcParams['figure.dpi'] = 360
sns.set(style="whitegrid")
fig, ax = plt.subplots(figsize=(6,4))
plt.xlabel('')
plt.ylabel('')
#plt.title("Li et al. (2020)", fontsize=12, color='#4f4e4e')
sns.despine(left=True, right = True, top = True)
sns.barplot(np.array([0, 1, 2, 3, 4]), np.repeat(sum(li_ssp1.y2020), 5), palette=['#838B8B'], label = "Data - 2020")
sns.barplot(np.array([0, 1, 2, 3, 4]), np.array([sum(li_ssp1.y2050), sum(li_ssp2.y2050), sum(li_ssp3.y2050), sum(li_ssp4.y2050), sum(li_ssp5.y2050)]) - np.repeat(sum(li_ssp1.y2020), 5), bottom = np.repeat(sum(li_ssp1.y2020), 5), palette = ['#FF0000'], label = "Simulations - 2050")
sns.barplot(np.array([0, 1, 2, 3, 4]), np.array([sum(li_ssp1.y2100), sum(li_ssp2.y2100), sum(li_ssp3.y2100), sum(li_ssp4.y2100), sum(li_ssp5.y2100)]) - np.array([sum(li_ssp1.y2050), sum(li_ssp2.y2050), sum(li_ssp3.y2050), sum(li_ssp4.y2050), sum(li_ssp5.y2050)]), bottom = np.array([sum(li_ssp1.y2050), sum(li_ssp2.y2050), sum(li_ssp3.y2050), sum(li_ssp4.y2050), sum(li_ssp5.y2050)]), palette = ['#CD0000'], label = "Simulations - 2100")
plt.xticks(np.array([0, 1, 2, 3, 4]), ["SSP1", "SSP2", "SSP3", "SSP4", "SSP5"])
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles[0:], labels=labels[0:], title = '', bbox_to_anchor=(0.3, 0.95), fontsize = 12, loc=2, borderaxespad=0.)
ax.tick_params(axis = 'both', labelsize=12, color='#4f4e4e')

#Stats des by continent
iso_continent = pd.read_excel("C:/Users/charl/OneDrive/Bureau/Urban sprawl/iso_continent.xlsx")

li_ssp1_continent = li_ssp1.merge(iso_continent, left_on = "cntryCode", right_on = "ISO3v10").groupby('Continent').sum()
li_ssp1_continent["y2100"] / li_ssp1_continent["y2020"]

li_ssp2_continent = li_ssp2.merge(iso_continent, left_on = "cntryCode", right_on = "ISO3v10").groupby('Continent').sum()
li_ssp2_continent["y2100"] / li_ssp2_continent["y2020"]

li_ssp3_continent = li_ssp3.merge(iso_continent, left_on = "cntryCode", right_on = "ISO3v10").groupby('Continent').sum()
li_ssp3_continent["y2050"] / li_ssp3_continent["y2020"]

li_ssp4_continent = li_ssp4.merge(iso_continent, left_on = "cntryCode", right_on = "ISO3v10").groupby('Continent').sum()
li_ssp4_continent["y2050"] / li_ssp4_continent["y2020"]

li_ssp5_continent = li_ssp5.merge(iso_continent, left_on = "cntryCode", right_on = "ISO3v10").groupby('Continent').sum()
li_ssp5_continent["y2050"] / li_ssp5_continent["y2020"]

#Cumulé - Chen

chen = pd.read_excel('C:/Users/charl/OneDrive/Bureau/Urban sprawl/chen_2020/global_sprawl.xlsx')

plt.rcParams['figure.dpi'] = 360
sns.set(style="whitegrid")
plt.ylabel('')
fig, ax = plt.subplots(figsize=(6,4))
plt.xlabel('')
plt.ylabel('')
#plt.title("Li et al. (2020)", fontsize=12, color='#4f4e4e')
sns.despine(left=True, right = True, top = True)
sns.barplot(np.array([0, 1, 2, 3, 4]), chen["chen_2020"], palette=['#838B8B'], label = "2020")
sns.barplot(np.array([0, 1, 2, 3, 4]), chen["chen_2050"] - chen["chen_2020"], bottom = chen["chen_2020"], palette = ['#FF0000'], label = "Simulations - 2050")
sns.barplot(np.array([0, 1, 2, 3, 4]), chen["chen_2100"] - chen["chen_2050"], bottom = chen["chen_2050"], palette = ['#CD0000'], label = "Simulations - 2100")
plt.xticks(np.array([0, 1, 2, 3, 4]), ["SSP1", "SSP2", "SSP3", "SSP4", "SSP5"])
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles[0:], labels=labels[0:], title = '', bbox_to_anchor=(0.3, 0.95), fontsize = 12, loc=2, borderaxespad=0.)
ax.tick_params(axis = 'both', labelsize=12, color='#4f4e4e')
plt.ylabel('')

#Cumulé - Huang

huang = pd.read_excel('C:/Users/charl/OneDrive/Bureau/Urban sprawl/huang_seto.xlsx', sheet_name = 'huang_2019')

plt.rcParams['figure.dpi'] = 360
sns.set(style="whitegrid")
plt.ylabel('')
fig, ax = plt.subplots(figsize=(6,4))
plt.xlabel('')
plt.ylabel('')
#plt.title("Li et al. (2020)", fontsize=12, color='#4f4e4e')
sns.despine(left=True, right = True, top = True)
sns.barplot(np.array([0, 1, 2, 3, 4]), np.repeat(sum(huang.urban_land_2015), 5), palette=['#838B8B'], label = "2015")
sns.barplot(np.array([0, 1, 2, 3, 4]), np.array([sum(huang.urban_land_2050_ssp1), sum(huang.urban_land_2050_ssp2), sum(huang.urban_land_2050_ssp3), sum(huang.urban_land_2050_ssp4), sum(huang.urban_land_2050_ssp5)]) - np.repeat(sum(huang.urban_land_2015), 5), bottom = np.repeat(sum(huang.urban_land_2015), 5), palette = ['#FF0000'], label = "Simulations - 2050")
plt.xticks(np.array([0, 1, 2, 3, 4]), ["SSP1", "SSP2", "SSP3", "SSP4", "SSP5"])
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles[0:], labels=labels[0:], title = '', bbox_to_anchor=(0.3, 0.95), fontsize = 12, loc=2, borderaxespad=0.)
ax.tick_params(axis = 'both', labelsize=12, color='#4f4e4e')
plt.ylabel('')

#Stats des by continent
huang.index = huang["Continent"]
huang["urban_land_2050_ssp1"] / huang["urban_land_2015"]
huang["urban_land_2050_ssp2"] / huang["urban_land_2015"]
huang["urban_land_2050_ssp3"] / huang["urban_land_2015"]
huang["urban_land_2050_ssp4"] / huang["urban_land_2015"]
huang["urban_land_2050_ssp5"] / huang["urban_land_2015"]

#Cumulé - Seto 2011 and 2012

seto_2012 = pd.read_excel('C:/Users/charl/OneDrive/Bureau/Urban sprawl/huang_seto.xlsx', sheet_name = 'seto_2012')
seto_2011 = pd.read_excel('C:/Users/charl/OneDrive/Bureau/Urban sprawl/huang_seto.xlsx', sheet_name = 'seto_2011')

plt.rcParams['figure.dpi'] = 360
sns.set(style="whitegrid")
plt.ylabel('')
fig, ax = plt.subplots(figsize=(12,4))
plt.xlabel('')
plt.ylabel('')
#plt.title("Li et al. (2020)", fontsize=12, color='#4f4e4e')
sns.despine(left=True, right = True, top = True)
sns.barplot(np.arange(13), np.concatenate([np.repeat(sum(seto_2012.urban_land_2000), 1), np.repeat(np.array([seto_2011.urban_land_2000]), 4).reshape(4,3,order='F').ravel()]), palette=['#838B8B'], label = "2000")
sns.barplot(np.arange(13), np.concatenate([np.repeat(sum(seto_2012.urban_land_2030), 1), seto_2011.urban_land_2030_A1, seto_2011.urban_land_2030_A2, seto_2011.urban_land_2030_B1, seto_2011.urban_land_2030_B2]) - np.concatenate([np.repeat(sum(seto_2012.urban_land_2000), 1), np.repeat(np.array([seto_2011.urban_land_2000]), 4).reshape(4,3,order='F').ravel()]), bottom =  np.concatenate([np.repeat(sum(seto_2012.urban_land_2000), 1), np.repeat(np.array([seto_2011.urban_land_2000]), 4).reshape(4,3,order='F').ravel()]), palette = ['#FF0000'], label = "Simulations - 2050")
plt.xticks(np.arange(13), ["Seto et al., 2012", "MODIS - A1", "GRUMP - A1", "GLC00 - A1", "MODIS - A2", "GRUMP - A2", "GLC00 - A2", "MODIS - B1", "GRUMP - B1", "GLC00 - B1", "MODIS - B2", "GRUMP - B2", "GLC00 - B2"], fontsize = 6)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles[0:], labels=labels[0:], title = '', bbox_to_anchor=(0.3, 0.95), fontsize = 12, loc=2, borderaxespad=0.)
ax.tick_params(axis = 'both', labelsize=8, color='#4f4e4e')
plt.ylabel('')
plt.xlabel('')

seto_2012.index = seto_2012.Continent
seto_2012.urban_land_2030 / seto_2012.urban_land_2000

seto_2011.urban_land_2030_A1 / seto_2011.urban_land_2000
seto_2011.urban_land_2030_A2 / seto_2011.urban_land_2000
seto_2011.urban_land_2030_B1 / seto_2011.urban_land_2000
seto_2011.urban_land_2030_B2 / seto_2011.urban_land_2000

#Angel 2011

angel_2011 = pd.read_excel('C:/Users/charl/OneDrive/Bureau/Urban sprawl/huang_seto.xlsx', sheet_name = 'angel_2011')
angel_2011 = angel_2011.loc[angel_2011.Region == 'World', :]

plt.rcParams['figure.dpi'] = 360
sns.set(style="whitegrid")
plt.ylabel('')
fig, ax = plt.subplots(figsize=(6,4))
plt.xlabel('')
plt.ylabel('')
#plt.title("Li et al. (2020)", fontsize=12, color='#4f4e4e')
sns.despine(left=True, right = True, top = True)
sns.barplot(np.arange(3), angel_2011.Urban_land_2000, palette=['#838B8B'], label = "2000")
sns.barplot(np.arange(3), angel_2011.Urban_land_2050 - angel_2011.Urban_land_2000, bottom = angel_2011.Urban_land_2000, palette = ['#FF0000'], label = "Simulations - 2050")
plt.xticks(np.arange(3), ["Density decline 0%", "Density decline 1%", "Density decline 2%"], fontsize = 6)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles[0:], labels=labels[0:], title = '', bbox_to_anchor=(0.3, 0.95), fontsize = 12, loc=2, borderaxespad=0.)
ax.tick_params(axis = 'both', labelsize=8, color='#4f4e4e')
plt.ylabel('')
plt.xlabel('')

#stats des by contient

iso_code = pd.read_excel("C:/Users/charl/OneDrive/Bureau/Urban sprawl/ISO_code.xlsx")
results = results.merge(iso_code, on = "City")

summary_by_continent_angel = results.groupby('Angel').sum()

summary_by_continent_angel.urbanized_area_reg4_2050_SSP1 / summary_by_continent_angel.urbanized_area_reg4_2015
summary_by_continent_angel.urbanized_area_reg4_2050_SSP2 / summary_by_continent_angel.urbanized_area_reg4_2015
summary_by_continent_angel.urbanized_area_reg4_2050_SSP3 / summary_by_continent_angel.urbanized_area_reg4_2015
summary_by_continent_angel.urbanized_area_reg4_2050_SSP4 / summary_by_continent_angel.urbanized_area_reg4_2015
summary_by_continent_angel.urbanized_area_reg4_2050_SSP5 / summary_by_continent_angel.urbanized_area_reg4_2015

summary_by_continent_angel.urbanized_area_reg2b_2050_SSP1 / summary_by_continent_angel.urbanized_area_reg2b_2015
summary_by_continent_angel.urbanized_area_reg2b_2050_SSP2 / summary_by_continent_angel.urbanized_area_reg2b_2015
summary_by_continent_angel.urbanized_area_reg2b_2050_SSP3 / summary_by_continent_angel.urbanized_area_reg2b_2015
summary_by_continent_angel.urbanized_area_reg2b_2050_SSP4 / summary_by_continent_angel.urbanized_area_reg2b_2015
summary_by_continent_angel.urbanized_area_reg2b_2050_SSP5 / summary_by_continent_angel.urbanized_area_reg2b_2015

angel_2011_v2 = pd.read_excel('C:/Users/charl/OneDrive/Bureau/Urban sprawl/huang_seto.xlsx', sheet_name = 'angel_2011')
angel_2011_density0 = angel_2011_v2.loc[angel_2011_v2['Annual density decline (%)'] == 0, :]
angel_2011_density1 = angel_2011_v2.loc[angel_2011_v2['Annual density decline (%)'] == 1, :]
angel_2011_density2 = angel_2011_v2.loc[angel_2011_v2['Annual density decline (%)'] == 2, :]

angel_2011_density0.index = angel_2011_density0.Region
angel_2011_density1.index = angel_2011_density1.Region
angel_2011_density2.index = angel_2011_density2.Region

angel_2011_density0.Urban_land_2050 / angel_2011_density0.Urban_land_2010
angel_2011_density1.Urban_land_2050 / angel_2011_density1.Urban_land_2010
angel_2011_density2.Urban_land_2050 / angel_2011_density2.Urban_land_2010

#atlas of urban expansion

data_density = pd.read_excel("C:/Users/charl/OneDrive/Bureau/Urban sprawl and density/atlas_of_urban_expansion.xlsx")
np.nansum(data_density.urban_extent_t3_ha) / np.nansum(data_density.urban_extent_t1_ha)

plt.rcParams['figure.dpi'] = 360
sns.set(style="whitegrid")
plt.ylabel('')
fig, ax = plt.subplots(figsize=(6,4))
plt.xlabel('')
plt.ylabel('')
#plt.title("Li et al. (2020)", fontsize=12, color='#4f4e4e')
sns.despine(left=True, right = True, top = True)
sns.barplot(np.array([0]), np.array([np.nansum(data_density.urban_extent_t1_ha)]), palette=['#838B8B'], label = "t1")
sns.barplot(np.array([0]), np.array([np.nansum(data_density.urban_extent_t2_ha) - np.nansum(data_density.urban_extent_t1_ha)]), bottom = np.array([np.nansum(data_density.urban_extent_t1_ha)]), palette = ['#FF0000'], label = "t2")
sns.barplot(np.array([0]), np.array([np.nansum(data_density.urban_extent_t3_ha) - np.nansum(data_density.urban_extent_t2_ha)]), bottom = np.array([np.nansum(data_density.urban_extent_t2_ha)]), palette = ['#CD0000'], label = "t3")
plt.xticks(np.array([0]), ["Atlas of Urban Expansion"])
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles[0:], labels=labels[0:], title = '', bbox_to_anchor=(0.3, 0.95), fontsize = 12, loc=2, borderaxespad=0.)
ax.tick_params(axis = 'both', labelsize=12, color='#4f4e4e')
plt.ylabel('')

#Güneralp 2020

guneralp_2020_city = pd.read_excel("C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Data/guneralp_2020/Input_ranked-by-LocationName_WUP300K.xlsx").loc[:, ['Region', 'Area1970', 'Area1990', 'Area2010']]
guneralp_2020_city = guneralp_2020_city.loc[guneralp_2020_city.Area2010 != " ", :]
guneralp_2020_city = guneralp_2020_city.loc[guneralp_2020_city.Area1990 != " ", :]
guneralp_2020_city = guneralp_2020_city.loc[guneralp_2020_city.Area1970 != " ", :]

plt.rcParams['figure.dpi'] = 360
sns.set(style="whitegrid")
plt.ylabel('')
fig, ax = plt.subplots(figsize=(6,4))
plt.xlabel('')
plt.ylabel('')
#plt.title("Li et al. (2020)", fontsize=12, color='#4f4e4e')
sns.despine(left=True, right = True, top = True)
sns.barplot(np.array([0]), np.array([np.nansum(guneralp_2020_city.Area1970)]), palette=['#838B8B'], label = "1970")
sns.barplot(np.array([0]), np.array([np.nansum(guneralp_2020_city.Area1990) - np.nansum(guneralp_2020_city.Area1970)]), bottom = np.array([np.nansum(guneralp_2020_city.Area1970)]), palette = ['#FF0000'], label = "1990")
sns.barplot(np.array([0]), np.array([np.nansum(guneralp_2020_city.Area2010) - np.nansum(guneralp_2020_city.Area1990)]), bottom = np.array([np.nansum(guneralp_2020_city.Area1990)]), palette = ['#CD0000'], label = "2010")
plt.xticks(np.array([0]), ["Güneralp et al. (2020)"])
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles[0:], labels=labels[0:], title = '', bbox_to_anchor=(0.3, 0.95), fontsize = 12, loc=2, borderaxespad=0.)
ax.tick_params(axis = 'both', labelsize=12, color='#4f4e4e')
plt.ylabel('')

guneralp_by_region = guneralp_2020_city.groupby('Region').sum()
guneralp_by_region["Area2010"] / guneralp_by_region["Area1970"]

### CATPLOT REG2B

city_continent = pd.read_csv(path_folder + "CityDatabases/cityDatabase_221NewCities.csv")
city_continent = city_continent.iloc[:, [0, 2]]
city_continent = city_continent.drop_duplicates(subset = "City")
city_continent = city_continent.sort_values('City')
results = results.merge(city_continent, left_index = True, right_on = "City")

results["pop_2015"] = np.nan
results["pop_2050_SSP1"] = np.nan
results["pop_2050_SSP2"] = np.nan
results["pop_2050_SSP3"] = np.nan
results["pop_2050_SSP4"] = np.nan
results["pop_2050_SSP5"] = np.nan

results.index = results.City
for city in np.delete(results.index, 153):
    results.loc[results.index == city, 'pop_2015'] = np.nansum(np.load(path_ssp1 + city + "_density.npy")[0])
    results.loc[results.index == city, 'pop_2050_SSP1'] = np.nansum(np.load(path_ssp1 + city + "_density.npy")[35])
    results.loc[results.index == city, 'pop_2050_SSP2'] = np.nansum(np.load(path_ssp2 + city + "_density.npy")[35])
    results.loc[results.index == city, 'pop_2050_SSP3'] = np.nansum(np.load(path_ssp3 + city + "_density.npy")[35])
    results.loc[results.index == city, 'pop_2050_SSP4'] = np.nansum(np.load(path_ssp4 + city + "_density.npy")[35])
    results.loc[results.index == city, 'pop_2050_SSP5'] = np.nansum(np.load(path_ssp5 + city + "_density.npy")[35])
    

results["inc_2015"] = np.nan
results["inc_2050_SSP1"] = np.nan
results["inc_2050_SSP2"] = np.nan
results["inc_2050_SSP3"] = np.nan
results["inc_2050_SSP4"] = np.nan
results["inc_2050_SSP5"] = np.nan

results.index = results.City
for city in np.delete(results.index, 153):
    results.loc[results.index == city, 'inc_2015'] = np.nansum(np.load(path_ssp1 + city + "_income.npy")[0])
    results.loc[results.index == city, 'inc_2050_SSP1'] = np.nansum(np.load(path_ssp1 + city + "_income.npy")[35])
    results.loc[results.index == city, 'inc_2050_SSP2'] = np.nansum(np.load(path_ssp2 + city + "_income.npy")[35])
    results.loc[results.index == city, 'inc_2050_SSP3'] = np.nansum(np.load(path_ssp3 + city + "_income.npy")[35])
    results.loc[results.index == city, 'inc_2050_SSP4'] = np.nansum(np.load(path_ssp4 + city + "_income.npy")[35])
    results.loc[results.index == city, 'inc_2050_SSP5'] = np.nansum(np.load(path_ssp5 + city + "_income.npy")[35])
    
    
y = np.array(((results["urbanized_area_reg4_2050_SSP2"] / results["urbanized_area_reg4_2015"]).loc[results.final_sample ==1]))
X = pd.DataFrame(np.array([(results["inc_2050_SSP2"]/results["inc_2015"]).loc[results.final_sample ==1], (results["pop_2050_SSP2"]/results["pop_2015"]).loc[results.final_sample ==1]])).transpose()
ols = sm.OLS(y, X)
reg1 = ols.fit(cov_type='HC1') #HC3 censé être mieux
reg1.summary()   
    
    
results["inc_2015"] = results.inc_2015 * results.pop_2015
results["inc_2050_SSP1"] = results.inc_2050_SSP1 * results.pop_2015
results["inc_2050_SSP2"] = results.inc_2050_SSP2 * results.pop_2015
results["inc_2050_SSP3"] = results.inc_2050_SSP3 * results.pop_2015
results["inc_2050_SSP4"] = results.inc_2050_SSP4 * results.pop_2015
results["inc_2050_SSP5"] = results.inc_2050_SSP5 * results.pop_2015


summary_by_continent = results.groupby('Continent').sum()
summary_by_continent["Continent"] = summary_by_continent.index

summary_by_continent.urbanized_area_reg2b_2100_SSP1 = summary_by_continent.urbanized_area_reg2b_2050_SSP1 / summary_by_continent.urbanized_area_reg2b_2015
summary_by_continent.urbanized_area_reg2b_2100_SSP2 = summary_by_continent.urbanized_area_reg2b_2050_SSP2 / summary_by_continent.urbanized_area_reg2b_2015
summary_by_continent.urbanized_area_reg2b_2100_SSP3 = summary_by_continent.urbanized_area_reg2b_2050_SSP3 / summary_by_continent.urbanized_area_reg2b_2015
summary_by_continent.urbanized_area_reg2b_2100_SSP4 = summary_by_continent.urbanized_area_reg2b_2050_SSP4 / summary_by_continent.urbanized_area_reg2b_2015
summary_by_continent.urbanized_area_reg2b_2100_SSP5 = summary_by_continent.urbanized_area_reg2b_2050_SSP5 / summary_by_continent.urbanized_area_reg2b_2015

summary_by_continent.urbanized_area_reg4_2100_SSP1 = summary_by_continent.urbanized_area_reg4_2050_SSP1 / summary_by_continent.urbanized_area_reg4_2015
summary_by_continent.urbanized_area_reg4_2100_SSP2 = summary_by_continent.urbanized_area_reg4_2050_SSP2 / summary_by_continent.urbanized_area_reg4_2015
summary_by_continent.urbanized_area_reg4_2100_SSP3 = summary_by_continent.urbanized_area_reg4_2050_SSP3 / summary_by_continent.urbanized_area_reg4_2015
summary_by_continent.urbanized_area_reg4_2100_SSP4 = summary_by_continent.urbanized_area_reg4_2050_SSP4 / summary_by_continent.urbanized_area_reg4_2015
summary_by_continent.urbanized_area_reg4_2100_SSP5 = summary_by_continent.urbanized_area_reg4_2050_SSP5 / summary_by_continent.urbanized_area_reg4_2015

summary_by_continent.pop_2050_SSP1 = summary_by_continent.pop_2050_SSP1 / summary_by_continent.pop_2015
summary_by_continent.pop_2050_SSP2 = summary_by_continent.pop_2050_SSP2 / summary_by_continent.pop_2015
summary_by_continent.pop_2050_SSP3 = summary_by_continent.pop_2050_SSP3 / summary_by_continent.pop_2015
summary_by_continent.pop_2050_SSP4 = summary_by_continent.pop_2050_SSP4 / summary_by_continent.pop_2015
summary_by_continent.pop_2050_SSP5 = summary_by_continent.pop_2050_SSP5 / summary_by_continent.pop_2015

summary_by_continent.inc_2050_SSP1 = summary_by_continent.inc_2050_SSP1 / summary_by_continent.inc_2015
summary_by_continent.inc_2050_SSP2 = summary_by_continent.inc_2050_SSP2 / summary_by_continent.inc_2015
summary_by_continent.inc_2050_SSP3 = summary_by_continent.inc_2050_SSP3 / summary_by_continent.inc_2015
summary_by_continent.inc_2050_SSP4 = summary_by_continent.inc_2050_SSP4 / summary_by_continent.inc_2015
summary_by_continent.inc_2050_SSP5 = summary_by_continent.inc_2050_SSP5 / summary_by_continent.inc_2015


df_ssp1 = pd.melt(summary_by_continent.loc[:, ['urbanized_area_reg2b_2100_SSP1', 'urbanized_area_reg4_2100_SSP1', 'pop_2050_SSP1', 'inc_2050_SSP1', 'Continent']], ["Continent"], var_name="method", value_name="urbanized_area")
df_ssp1["SSP"] = "SSP1"

df_ssp2= pd.melt(summary_by_continent.loc[:, ['urbanized_area_reg2b_2100_SSP2', 'urbanized_area_reg4_2100_SSP2', 'pop_2050_SSP2', 'inc_2050_SSP2', 'Continent']], ["Continent"], var_name="method", value_name="urbanized_area")
df_ssp2["SSP"] = "SSP2"

df_ssp3 = pd.melt(summary_by_continent.loc[:, ['urbanized_area_reg2b_2100_SSP3', 'urbanized_area_reg4_2100_SSP3', 'pop_2050_SSP3', 'inc_2050_SSP3', 'Continent']], ["Continent"], var_name="method", value_name="urbanized_area")
df_ssp3["SSP"] = "SSP3"

df_ssp4 = pd.melt(summary_by_continent.loc[:, ['urbanized_area_reg2b_2100_SSP4', 'urbanized_area_reg4_2100_SSP4', 'pop_2050_SSP4', 'inc_2050_SSP4', 'Continent']], ["Continent"], var_name="method", value_name="urbanized_area")
df_ssp4["SSP"] = "SSP4"

df_ssp5 = pd.melt(summary_by_continent.loc[:, ['urbanized_area_reg2b_2100_SSP5', 'urbanized_area_reg4_2100_SSP5', 'pop_2050_SSP5', 'inc_2050_SSP5', 'Continent']], ["Continent"], var_name="method", value_name="urbanized_area")
df_ssp5["SSP"] = "SSP5"

df_all_ssp = df_ssp1.append(df_ssp2).append(df_ssp3).append(df_ssp4).append(df_ssp5)
df_all_ssp.method[(df_all_ssp.method == 'urbanized_area_reg2b_2100_SSP1')|(df_all_ssp.method == 'urbanized_area_reg2b_2100_SSP2')|(df_all_ssp.method == 'urbanized_area_reg2b_2100_SSP3')|(df_all_ssp.method == 'urbanized_area_reg2b_2100_SSP4')|(df_all_ssp.method == 'urbanized_area_reg2b_2100_SSP5')] = 'Urban econ'
df_all_ssp.method[(df_all_ssp.method == 'urbanized_area_reg4_2100_SSP1')|(df_all_ssp.method == 'urbanized_area_reg4_2100_SSP2')|(df_all_ssp.method == 'urbanized_area_reg4_2100_SSP3')|(df_all_ssp.method == 'urbanized_area_reg4_2100_SSP4')|(df_all_ssp.method == 'urbanized_area_reg4_2100_SSP5')] = 'NEDUM'
df_all_ssp.method[(df_all_ssp.method == 'pop_2050_SSP1')|(df_all_ssp.method == 'pop_2050_SSP2')|(df_all_ssp.method == 'pop_2050_SSP3')|(df_all_ssp.method == 'pop_2050_SSP4')|(df_all_ssp.method == 'pop_2050_SSP5')] = 'pop'
df_all_ssp.method[(df_all_ssp.method == 'inc_2050_SSP1')|(df_all_ssp.method == 'inc_2050_SSP2')|(df_all_ssp.method == 'inc_2050_SSP3')|(df_all_ssp.method == 'inc_2050_SSP4')|(df_all_ssp.method == 'inc_2050_SSP5')] = 'income'


plt.rcParams['figure.dpi'] = 400
sns.set(font_scale=4, style = "whitegrid", rc={'figure.figsize':(15, 10)})
g = sns.catplot(x = "Continent", y = "urbanized_area", hue = "method", col = "SSP", data = df_all_ssp, kind="bar", col_wrap=2, height=10, aspect=15/10, palette = sns.color_palette("Set2"))
(g.set_axis_labels("", "Urbanized area")
  .set_titles("{col_name}")  # 
  .set_xticklabels(["Africa", "Asia", "Europe", "North \n America", "Oeania", "South \n America"])
  .despine(left=True, right = True, top = True)) 
g._legend.set_bbox_to_anchor((.7, 0.15))
g._legend.set_title(None)

#CATPLOT GAO

gao_ssp1 = pd.read_csv("C:/Users/charl/OneDrive/Bureau/Urban sprawl/gao_2020/NationalTotalUrbanArea_SSP1_2000-2100_v1.csv")
gao_ssp2 = pd.read_csv("C:/Users/charl/OneDrive/Bureau/Urban sprawl/gao_2020/NationalTotalUrbanArea_SSP2_2000-2100_v1.csv")
gao_ssp3 = pd.read_csv("C:/Users/charl/OneDrive/Bureau/Urban sprawl/gao_2020/NationalTotalUrbanArea_SSP3_2000-2100_v1.csv")
gao_ssp4 = pd.read_csv("C:/Users/charl/OneDrive/Bureau/Urban sprawl/gao_2020/NationalTotalUrbanArea_SSP4_2000-2100_v1.csv")
gao_ssp5 = pd.read_csv("C:/Users/charl/OneDrive/Bureau/Urban sprawl/gao_2020/NationalTotalUrbanArea_SSP5_2000-2100_v1.csv")
iso_continent = pd.read_excel("C:/Users/charl/OneDrive/Bureau/Urban sprawl/iso_continent.xlsx")
gap_ssp1_continent = gao_ssp1.merge(iso_continent, on = "ISO3v10").groupby('Continent').sum()
gap_ssp2_continent = gao_ssp2.merge(iso_continent, on = "ISO3v10").groupby('Continent').sum()
gap_ssp3_continent = gao_ssp3.merge(iso_continent, on = "ISO3v10").groupby('Continent').sum()
gap_ssp4_continent = gao_ssp4.merge(iso_continent, on = "ISO3v10").groupby('Continent').sum()
gap_ssp5_continent = gao_ssp5.merge(iso_continent, on = "ISO3v10").groupby('Continent').sum()

summary_by_continent = results.groupby('Continent').sum()
summary_by_continent["Continent"] = summary_by_continent.index

gap_ssp1_continent = gap_ssp1_continent.loc[:, ['UrbAmt2010', 'UrbAmt2050']]
gap_ssp1_continent.columns = ["Gao_2010_SSP1", "Gao_2100_SSP1"]
gap_ssp1_continent["Gao_2100_SSP1"] = gap_ssp1_continent.Gao_2100_SSP1 / gap_ssp1_continent.Gao_2010_SSP1

gap_ssp2_continent = gap_ssp2_continent.loc[:, ['UrbAmt2010', 'UrbAmt2050']]
gap_ssp2_continent.columns = ["Gao_2010_SSP2", "Gao_2100_SSP2"]
gap_ssp2_continent["Gao_2100_SSP2"] = gap_ssp2_continent.Gao_2100_SSP2 / gap_ssp2_continent.Gao_2010_SSP2

gap_ssp3_continent = gap_ssp3_continent.loc[:, ['UrbAmt2010', 'UrbAmt2050']]
gap_ssp3_continent.columns = ["Gao_2010_SSP3", "Gao_2100_SSP3"]
gap_ssp3_continent["Gao_2100_SSP3"] = gap_ssp3_continent.Gao_2100_SSP3 / gap_ssp3_continent.Gao_2010_SSP3

gap_ssp4_continent = gap_ssp4_continent.loc[:, ['UrbAmt2010', 'UrbAmt2050']]
gap_ssp4_continent.columns = ["Gao_2010_SSP4", "Gao_2100_SSP4"]
gap_ssp4_continent["Gao_2100_SSP4"] = gap_ssp4_continent.Gao_2100_SSP4 / gap_ssp4_continent.Gao_2010_SSP4

gap_ssp5_continent = gap_ssp5_continent.loc[:, ['UrbAmt2010', 'UrbAmt2050']]
gap_ssp5_continent.columns = ["Gao_2010_SSP5", "Gao_2100_SSP5"]
gap_ssp5_continent["Gao_2100_SSP5"] = gap_ssp5_continent.Gao_2100_SSP5 / gap_ssp5_continent.Gao_2010_SSP5

summary_by_continent.urbanized_area_reg4_2100_SSP1 = summary_by_continent.urbanized_area_reg4_2050_SSP1 / summary_by_continent.urbanized_area_reg4_2015
summary_by_continent.urbanized_area_reg4_2100_SSP2 = summary_by_continent.urbanized_area_reg4_2050_SSP2 / summary_by_continent.urbanized_area_reg4_2015
summary_by_continent.urbanized_area_reg4_2100_SSP3 = summary_by_continent.urbanized_area_reg4_2050_SSP3 / summary_by_continent.urbanized_area_reg4_2015
summary_by_continent.urbanized_area_reg4_2100_SSP4 = summary_by_continent.urbanized_area_reg4_2050_SSP4 / summary_by_continent.urbanized_area_reg4_2015
summary_by_continent.urbanized_area_reg4_2100_SSP5 = summary_by_continent.urbanized_area_reg4_2050_SSP5 / summary_by_continent.urbanized_area_reg4_2015


summary_by_continent = summary_by_continent.merge(gap_ssp1_continent, left_index = True, right_index = True).merge(gap_ssp2_continent, left_index = True, right_index = True).merge(gap_ssp3_continent, left_index = True, right_index = True).merge(gap_ssp4_continent, left_index = True, right_index = True).merge(gap_ssp5_continent, left_index = True, right_index = True)

df_ssp1 = pd.melt(summary_by_continent.loc[:, ['Gao_2100_SSP1', 'urbanized_area_reg4_2100_SSP1', 'Continent']], ["Continent"], var_name="method", value_name="urbanized_area")
df_ssp1["SSP"] = "SSP1"

df_ssp2= pd.melt(summary_by_continent.loc[:, ['Gao_2100_SSP2', 'urbanized_area_reg4_2100_SSP2', 'Continent']], ["Continent"], var_name="method", value_name="urbanized_area")
df_ssp2["SSP"] = "SSP2"

df_ssp3 = pd.melt(summary_by_continent.loc[:, ['Gao_2100_SSP3', 'urbanized_area_reg4_2100_SSP3', 'Continent']], ["Continent"], var_name="method", value_name="urbanized_area")
df_ssp3["SSP"] = "SSP3"

df_ssp4 = pd.melt(summary_by_continent.loc[:, ['Gao_2100_SSP4', 'urbanized_area_reg4_2100_SSP4', 'Continent']], ["Continent"], var_name="method", value_name="urbanized_area")
df_ssp4["SSP"] = "SSP4"

df_ssp5 = pd.melt(summary_by_continent.loc[:, ['Gao_2100_SSP5', 'urbanized_area_reg4_2100_SSP5', 'Continent']], ["Continent"], var_name="method", value_name="urbanized_area")
df_ssp5["SSP"] = "SSP5"

df_all_ssp = df_ssp1.append(df_ssp2).append(df_ssp3).append(df_ssp4).append(df_ssp5)
df_all_ssp.method[(df_all_ssp.method == 'Gao_2100_SSP1')|(df_all_ssp.method == 'Gao_2100_SSP2')|(df_all_ssp.method == 'Gao_2100_SSP3')|(df_all_ssp.method == 'Gao_2100_SSP4')|(df_all_ssp.method == 'Gao_2100_SSP5')] = 'Gao'
df_all_ssp.method[(df_all_ssp.method == 'urbanized_area_reg4_2100_SSP1')|(df_all_ssp.method == 'urbanized_area_reg4_2100_SSP2')|(df_all_ssp.method == 'urbanized_area_reg4_2100_SSP3')|(df_all_ssp.method == 'urbanized_area_reg4_2100_SSP4')|(df_all_ssp.method == 'urbanized_area_reg4_2100_SSP5')] = 'NEDUM'

plt.rcParams['figure.dpi'] = 400
sns.set(font_scale=4, style = "whitegrid", rc={'figure.figsize':(15, 10)})
g = sns.catplot(x = "Continent", y = "urbanized_area", hue = "method", col = "SSP", data = df_all_ssp, kind="bar", col_wrap=2, height=10, aspect=15/10, palette = sns.color_palette("Set2"))
(g.set_axis_labels("", "Urbanized area")
  .set_titles("{col_name}")  # 
  .set_xticklabels(["Africa", "Asia", "Europe", "North \n America", "Oeania", "South \n America"])
  .despine(left=True, right = True, top = True)) 
g._legend.set_bbox_to_anchor((.7, 0.15))
g._legend.set_title(None)

### CATPLOT LI

li_ssp1 = pd.read_excel("C:/Users/charl/OneDrive/Bureau/Urban sprawl/li_2020.xlsx", sheet_name = 'SSP1')
li_ssp2 = pd.read_excel("C:/Users/charl/OneDrive/Bureau/Urban sprawl/li_2020.xlsx", sheet_name = 'SSP2')
li_ssp3 = pd.read_excel("C:/Users/charl/OneDrive/Bureau/Urban sprawl/li_2020.xlsx", sheet_name = 'SSP3')
li_ssp4 = pd.read_excel("C:/Users/charl/OneDrive/Bureau/Urban sprawl/li_2020.xlsx", sheet_name = 'SSP4')
li_ssp5 = pd.read_excel("C:/Users/charl/OneDrive/Bureau/Urban sprawl/li_2020.xlsx", sheet_name = 'SSP5')

iso_continent = pd.read_excel("C:/Users/charl/OneDrive/Bureau/Urban sprawl/iso_continent.xlsx")

li_ssp1_continent = li_ssp1.merge(iso_continent, left_on = "cntryCode", right_on = "ISO3v10").groupby('Continent').sum()
li_ssp2_continent = li_ssp2.merge(iso_continent, left_on = "cntryCode", right_on = "ISO3v10").groupby('Continent').sum()
li_ssp3_continent = li_ssp3.merge(iso_continent, left_on = "cntryCode", right_on = "ISO3v10").groupby('Continent').sum()
li_ssp4_continent = li_ssp4.merge(iso_continent, left_on = "cntryCode", right_on = "ISO3v10").groupby('Continent').sum()
li_ssp5_continent = li_ssp5.merge(iso_continent, left_on = "cntryCode", right_on = "ISO3v10").groupby('Continent').sum()

summary_by_continent = results.groupby('Continent').sum()
summary_by_continent["Continent"] = summary_by_continent.index

li_ssp1_continent = li_ssp1_continent.loc[:, ['y2020', 'y2050']]
li_ssp1_continent.columns = ["Li_2020_SSP1", "Li_2100_SSP1"]
li_ssp1_continent["Li_2100_SSP1"] = li_ssp1_continent.Li_2100_SSP1 / li_ssp1_continent.Li_2020_SSP1

li_ssp2_continent = li_ssp2_continent.loc[:, ['y2020', 'y2050']]
li_ssp2_continent.columns = ["Li_2020_SSP2", "Li_2100_SSP2"]
li_ssp2_continent["Li_2100_SSP2"] = li_ssp2_continent.Li_2100_SSP2 / li_ssp2_continent.Li_2020_SSP2

li_ssp3_continent = li_ssp3_continent.loc[:, ['y2020', 'y2050']]
li_ssp3_continent.columns = ["Li_2020_SSP3", "Li_2100_SSP3"]
li_ssp3_continent["Li_2100_SSP3"] = li_ssp3_continent.Li_2100_SSP3 / li_ssp3_continent.Li_2020_SSP3

li_ssp4_continent = li_ssp4_continent.loc[:, ['y2020', 'y2050']]
li_ssp4_continent.columns = ["Li_2020_SSP4", "Li_2100_SSP4"]
li_ssp4_continent["Li_2100_SSP4"] = li_ssp4_continent.Li_2100_SSP4 / li_ssp4_continent.Li_2020_SSP4

li_ssp5_continent = li_ssp5_continent.loc[:, ['y2020', 'y2050']]
li_ssp5_continent.columns = ["Li_2020_SSP5", "Li_2100_SSP5"]
li_ssp5_continent["Li_2100_SSP5"] = li_ssp5_continent.Li_2100_SSP5 / li_ssp5_continent.Li_2020_SSP5

summary_by_continent.urbanized_area_reg4_2100_SSP1 = summary_by_continent.urbanized_area_reg4_2050_SSP1 / summary_by_continent.urbanized_area_reg4_2015
summary_by_continent.urbanized_area_reg4_2100_SSP2 = summary_by_continent.urbanized_area_reg4_2050_SSP2 / summary_by_continent.urbanized_area_reg4_2015
summary_by_continent.urbanized_area_reg4_2100_SSP3 = summary_by_continent.urbanized_area_reg4_2050_SSP3 / summary_by_continent.urbanized_area_reg4_2015
summary_by_continent.urbanized_area_reg4_2100_SSP4 = summary_by_continent.urbanized_area_reg4_2050_SSP4 / summary_by_continent.urbanized_area_reg4_2015
summary_by_continent.urbanized_area_reg4_2100_SSP5 = summary_by_continent.urbanized_area_reg4_2050_SSP5 / summary_by_continent.urbanized_area_reg4_2015


summary_by_continent = summary_by_continent.merge(li_ssp1_continent, left_index = True, right_index = True).merge(li_ssp2_continent, left_index = True, right_index = True).merge(li_ssp3_continent, left_index = True, right_index = True).merge(li_ssp4_continent, left_index = True, right_index = True).merge(li_ssp5_continent, left_index = True, right_index = True)

df_ssp1 = pd.melt(summary_by_continent.loc[:, ['Li_2100_SSP1', 'urbanized_area_reg4_2100_SSP1', 'Continent']], ["Continent"], var_name="method", value_name="urbanized_area")
df_ssp1["SSP"] = "SSP1"

df_ssp2= pd.melt(summary_by_continent.loc[:, ['Li_2100_SSP2', 'urbanized_area_reg4_2100_SSP2', 'Continent']], ["Continent"], var_name="method", value_name="urbanized_area")
df_ssp2["SSP"] = "SSP2"

df_ssp3 = pd.melt(summary_by_continent.loc[:, ['Li_2100_SSP3', 'urbanized_area_reg4_2100_SSP3', 'Continent']], ["Continent"], var_name="method", value_name="urbanized_area")
df_ssp3["SSP"] = "SSP3"

df_ssp4 = pd.melt(summary_by_continent.loc[:, ['Li_2100_SSP4', 'urbanized_area_reg4_2100_SSP4', 'Continent']], ["Continent"], var_name="method", value_name="urbanized_area")
df_ssp4["SSP"] = "SSP4"

df_ssp5 = pd.melt(summary_by_continent.loc[:, ['Li_2100_SSP5', 'urbanized_area_reg4_2100_SSP5', 'Continent']], ["Continent"], var_name="method", value_name="urbanized_area")
df_ssp5["SSP"] = "SSP5"

df_all_ssp = df_ssp1.append(df_ssp2).append(df_ssp3).append(df_ssp4).append(df_ssp5)
df_all_ssp.method[(df_all_ssp.method == 'Li_2100_SSP1')|(df_all_ssp.method == 'Li_2100_SSP2')|(df_all_ssp.method == 'Li_2100_SSP3')|(df_all_ssp.method == 'Li_2100_SSP4')|(df_all_ssp.method == 'Li_2100_SSP5')] = 'Li'
df_all_ssp.method[(df_all_ssp.method == 'urbanized_area_reg4_2100_SSP1')|(df_all_ssp.method == 'urbanized_area_reg4_2100_SSP2')|(df_all_ssp.method == 'urbanized_area_reg4_2100_SSP3')|(df_all_ssp.method == 'urbanized_area_reg4_2100_SSP4')|(df_all_ssp.method == 'urbanized_area_reg4_2100_SSP5')] = 'NEDUM'

plt.rcParams['figure.dpi'] = 400
sns.set(font_scale=4, style = "whitegrid", rc={'figure.figsize':(15, 10)})
g = sns.catplot(x = "Continent", y = "urbanized_area", hue = "method", col = "SSP", data = df_all_ssp, kind="bar", col_wrap=2, height=10, aspect=15/10, palette = sns.color_palette("Set2"))
(g.set_axis_labels("", "Urbanized area")
  .set_titles("{col_name}")  # 
  .set_xticklabels(["Africa", "Asia", "Europe", "North \n America", "Oeania", "South \n America"])
  .despine(left=True, right = True, top = True)) 
g._legend.set_bbox_to_anchor((.7, 0.15))
g._legend.set_title(None)

### CATPLOT HUANG

huang = pd.read_excel('C:/Users/charl/OneDrive/Bureau/Urban sprawl/huang_seto.xlsx', sheet_name = 'huang_2019')
huang.index = huang.Continent

results2 = copy.deepcopy(results)
results2["Continent"][(results2.Continent == 'North_America')|(results2.Continent == 'South_America')] = "America"
summary_by_continent = results2.groupby('Continent').sum()
summary_by_continent["Continent"] = summary_by_continent.index

huang_ssp1_continent = huang.loc[:, ['urban_land_2015', 'urban_land_2050_ssp1']]
huang_ssp1_continent.columns = ["Huang_2015_SSP1", "Huang_2050_SSP1"]
huang_ssp1_continent["Huang_2050_SSP1"] = huang_ssp1_continent.Huang_2050_SSP1 / huang_ssp1_continent.Huang_2015_SSP1

huang_ssp2_continent = huang.loc[:, ['urban_land_2015', 'urban_land_2050_ssp2']]
huang_ssp2_continent.columns = ["Huang_2015_SSP2", "Huang_2050_SSP2"]
huang_ssp2_continent["Huang_2050_SSP2"] = huang_ssp2_continent.Huang_2050_SSP2 / huang_ssp2_continent.Huang_2015_SSP2

huang_ssp3_continent = huang.loc[:, ['urban_land_2015', 'urban_land_2050_ssp3']]
huang_ssp3_continent.columns = ["Huang_2015_SSP3", "Huang_2050_SSP3"]
huang_ssp3_continent["Huang_2050_SSP3"] = huang_ssp3_continent.Huang_2050_SSP3 / huang_ssp3_continent.Huang_2015_SSP3

huang_ssp4_continent = huang.loc[:, ['urban_land_2015', 'urban_land_2050_ssp4']]
huang_ssp4_continent.columns = ["Huang_2015_SSP4", "Huang_2050_SSP4"]
huang_ssp4_continent["Huang_2050_SSP4"] = huang_ssp4_continent.Huang_2050_SSP4 / huang_ssp4_continent.Huang_2015_SSP4

huang_ssp5_continent = huang.loc[:, ['urban_land_2015', 'urban_land_2050_ssp5']]
huang_ssp5_continent.columns = ["Huang_2015_SSP5", "Huang_2050_SSP5"]
huang_ssp5_continent["Huang_2050_SSP5"] = huang_ssp5_continent.Huang_2050_SSP5 / huang_ssp5_continent.Huang_2015_SSP5

summary_by_continent.urbanized_area_reg4_2100_SSP1 = summary_by_continent.urbanized_area_reg4_2050_SSP1 / summary_by_continent.urbanized_area_reg4_2015
summary_by_continent.urbanized_area_reg4_2100_SSP2 = summary_by_continent.urbanized_area_reg4_2050_SSP2 / summary_by_continent.urbanized_area_reg4_2015
summary_by_continent.urbanized_area_reg4_2100_SSP3 = summary_by_continent.urbanized_area_reg4_2050_SSP3 / summary_by_continent.urbanized_area_reg4_2015
summary_by_continent.urbanized_area_reg4_2100_SSP4 = summary_by_continent.urbanized_area_reg4_2050_SSP4 / summary_by_continent.urbanized_area_reg4_2015
summary_by_continent.urbanized_area_reg4_2100_SSP5 = summary_by_continent.urbanized_area_reg4_2050_SSP5 / summary_by_continent.urbanized_area_reg4_2015


summary_by_continent = summary_by_continent.merge(huang_ssp1_continent, left_index = True, right_index = True).merge(huang_ssp2_continent, left_index = True, right_index = True).merge(huang_ssp3_continent, left_index = True, right_index = True).merge(huang_ssp4_continent, left_index = True, right_index = True).merge(huang_ssp5_continent, left_index = True, right_index = True)

df_ssp1 = pd.melt(summary_by_continent.loc[:, ['Huang_2050_SSP1', 'urbanized_area_reg4_2100_SSP1', 'Continent']], ["Continent"], var_name="method", value_name="urbanized_area")
df_ssp1["SSP"] = "SSP1"

df_ssp2= pd.melt(summary_by_continent.loc[:, ['Huang_2050_SSP2', 'urbanized_area_reg4_2100_SSP2', 'Continent']], ["Continent"], var_name="method", value_name="urbanized_area")
df_ssp2["SSP"] = "SSP2"

df_ssp3 = pd.melt(summary_by_continent.loc[:, ['Huang_2050_SSP3', 'urbanized_area_reg4_2100_SSP3', 'Continent']], ["Continent"], var_name="method", value_name="urbanized_area")
df_ssp3["SSP"] = "SSP3"

df_ssp4 = pd.melt(summary_by_continent.loc[:, ['Huang_2050_SSP4', 'urbanized_area_reg4_2100_SSP4', 'Continent']], ["Continent"], var_name="method", value_name="urbanized_area")
df_ssp4["SSP"] = "SSP4"

df_ssp5 = pd.melt(summary_by_continent.loc[:, ['Huang_2050_SSP5', 'urbanized_area_reg4_2100_SSP5', 'Continent']], ["Continent"], var_name="method", value_name="urbanized_area")
df_ssp5["SSP"] = "SSP5"

df_all_ssp = df_ssp1.append(df_ssp2).append(df_ssp3).append(df_ssp4).append(df_ssp5)
df_all_ssp.method[(df_all_ssp.method == 'Huang_2050_SSP1')|(df_all_ssp.method == 'Huang_2050_SSP2')|(df_all_ssp.method == 'Huang_2050_SSP3')|(df_all_ssp.method == 'Huang_2050_SSP4')|(df_all_ssp.method == 'Huang_2050_SSP5')] = 'Huang'
df_all_ssp.method[(df_all_ssp.method == 'urbanized_area_reg4_2100_SSP1')|(df_all_ssp.method == 'urbanized_area_reg4_2100_SSP2')|(df_all_ssp.method == 'urbanized_area_reg4_2100_SSP3')|(df_all_ssp.method == 'urbanized_area_reg4_2100_SSP4')|(df_all_ssp.method == 'urbanized_area_reg4_2100_SSP5')] = 'NEDUM'

plt.rcParams['figure.dpi'] = 400
sns.set(font_scale=4, style = "whitegrid", rc={'figure.figsize':(15, 10)})
g = sns.catplot(x = "Continent", y = "urbanized_area", hue = "method", col = "SSP", data = df_all_ssp, kind="bar", col_wrap=2, height=10, aspect=15/10, palette = sns.color_palette("Set2"))
(g.set_axis_labels("", "Urbanized area")
  .set_titles("{col_name}")  # 
  .set_xticklabels(["Africa", "America", "Asia", "Europe", "Oeania"])
  .despine(left=True, right = True, top = True)) 
g._legend.set_bbox_to_anchor((.7, 0.15))
g._legend.set_title(None)

### CATPLOT SETO

seto_2012 = pd.read_excel('C:/Users/charl/OneDrive/Bureau/Urban sprawl/huang_seto.xlsx', sheet_name = 'seto_2012')
seto_2012 = seto_2012.iloc[0:5, :]
seto_2012.index = seto_2012.Continent

seto_continent = seto_2012.loc[:, ['urban_land_2000', 'urban_land_2030', 'Continent']]
seto_continent.columns = ["Seto_2000", "Seto_2030", 'Continent']
seto_continent["Seto_2030"] = seto_continent.Seto_2030 / seto_continent.Seto_2000

plt.rcParams['figure.dpi'] = 400
sns.set(font_scale=4, style = "whitegrid", rc={'figure.figsize':(15, 10)})
g = sns.catplot(x = "Continent", y = "Seto_2030", data = seto_continent, kind="bar", height=10, aspect=15/10, palette = sns.color_palette("Set2"))
(g.set_axis_labels("", "Urbanized area")
  .set_titles("{col_name}")  # 
  .set_xticklabels(["Africa", "America", "Asia", "Europe", "Oeania"])
  .despine(left=True, right = True, top = True)) 
#g._legend.set_bbox_to_anchor((.7, 0.15))
#g._legend.set_title(None)

### CATPLOT ANGEL

angel_2011 = pd.read_excel('C:/Users/charl/OneDrive/Bureau/Urban sprawl/huang_seto.xlsx', sheet_name = 'angel_2011')
angel_2011 = angel_2011.loc[0:26, ["Region", 'Annual density decline (%)', "Urban_land_2010", "Urban_land_2050"]]

iso_code = pd.read_excel("C:/Users/charl/OneDrive/Bureau/Urban sprawl/ISO_code.xlsx", sheet_name = 'Feuil1')
results = results.merge(iso_code, on = "City")

summary_by_continent_angel = results.groupby('Angel').sum()

summary_by_continent_angel.urbanized_area_reg4_2100_SSP1 = summary_by_continent_angel.urbanized_area_reg4_2050_SSP1 / summary_by_continent_angel.urbanized_area_reg4_2015
summary_by_continent_angel.urbanized_area_reg4_2100_SSP2 = summary_by_continent_angel.urbanized_area_reg4_2050_SSP2 / summary_by_continent_angel.urbanized_area_reg4_2015
summary_by_continent_angel.urbanized_area_reg4_2100_SSP3 = summary_by_continent_angel.urbanized_area_reg4_2050_SSP3 / summary_by_continent_angel.urbanized_area_reg4_2015
summary_by_continent_angel.urbanized_area_reg4_2100_SSP4 = summary_by_continent_angel.urbanized_area_reg4_2050_SSP4 / summary_by_continent_angel.urbanized_area_reg4_2015
summary_by_continent_angel.urbanized_area_reg4_2100_SSP5 = summary_by_continent_angel.urbanized_area_reg4_2050_SSP5 / summary_by_continent_angel.urbanized_area_reg4_2015

angel_2011_density0 = angel_2011.loc[angel_2011['Annual density decline (%)'] == 0, :]
angel_2011_density1 = angel_2011.loc[angel_2011['Annual density decline (%)'] == 1, :]
angel_2011_density2 = angel_2011.loc[angel_2011['Annual density decline (%)'] == 2, :]

angel_2011_density0.index = angel_2011_density0.Region
angel_2011_density1.index = angel_2011_density1.Region
angel_2011_density2.index = angel_2011_density2.Region

angel_2011_density0["Urban_land_2050_0"] = angel_2011_density0["Urban_land_2050"]  / angel_2011_density0.Urban_land_2010
angel_2011_density1["Urban_land_2050_1"] = angel_2011_density1["Urban_land_2050"] / angel_2011_density1.Urban_land_2010
angel_2011_density2["Urban_land_2050_2"] = angel_2011_density2["Urban_land_2050"] / angel_2011_density2.Urban_land_2010

summary_by_continent_angel = summary_by_continent_angel.merge(angel_2011_density0, left_index = True, right_index = True).merge(angel_2011_density1, left_index = True, right_index = True).merge(angel_2011_density2, left_index = True, right_index = True)

summary_by_continent_angel["Region"] = summary_by_continent_angel.index

df_ssp1= pd.melt(summary_by_continent_angel.loc[:, ['urbanized_area_reg4_2100_SSP1', 'Region']], ["Region"], var_name="method", value_name="urbanized_area")
df_ssp1["SSP"] = "SSP1"

df_ssp2= pd.melt(summary_by_continent_angel.loc[:, ['urbanized_area_reg4_2100_SSP2', 'Region']], ["Region"], var_name="method", value_name="urbanized_area")
df_ssp2["SSP"] = "SSP2"

df_ssp3= pd.melt(summary_by_continent_angel.loc[:, ['urbanized_area_reg4_2100_SSP3', 'Region']], ["Region"], var_name="method", value_name="urbanized_area")
df_ssp3["SSP"] = "SSP3"

df_ssp4= pd.melt(summary_by_continent_angel.loc[:, ['urbanized_area_reg4_2100_SSP4', 'Region']], ["Region"], var_name="method", value_name="urbanized_area")
df_ssp4["SSP"] = "SSP4"

df_ssp5= pd.melt(summary_by_continent_angel.loc[:, ['urbanized_area_reg4_2100_SSP5', 'Region']], ["Region"], var_name="method", value_name="urbanized_area")
df_ssp5["SSP"] = "SSP5"

df_density0= pd.melt(summary_by_continent_angel.loc[:, ['Urban_land_2050_0', 'Region']], ["Region"], var_name="method", value_name="urbanized_area")
df_density0["SSP"] = "density0"

df_density1= pd.melt(summary_by_continent_angel.loc[:, ['Urban_land_2050_1', 'Region']], ["Region"], var_name="method", value_name="urbanized_area")
df_density1["SSP"] = "density1"

df_density2= pd.melt(summary_by_continent_angel.loc[:, ['Urban_land_2050_2', 'Region']], ["Region"], var_name="method", value_name="urbanized_area")
df_density2["SSP"] = "density2"



df_all_ssp = df_ssp1.append(df_ssp2).append(df_ssp3).append(df_ssp4).append(df_ssp5).append(df_density0).append(df_density1).append(df_density2)
#df_all_ssp.method[(df_all_ssp.method == 'Huang_2050_SSP1')|(df_all_ssp.method == 'Huang_2050_SSP2')|(df_all_ssp.method == 'Huang_2050_SSP3')|(df_all_ssp.method == 'Huang_2050_SSP4')|(df_all_ssp.method == 'Huang_2050_SSP5')] = 'Huang'
#df_all_ssp.method[(df_all_ssp.method == 'urbanized_area_reg4_2100_SSP1')|(df_all_ssp.method == 'urbanized_area_reg4_2100_SSP2')|(df_all_ssp.method == 'urbanized_area_reg4_2100_SSP3')|(df_all_ssp.method == 'urbanized_area_reg4_2100_SSP4')|(df_all_ssp.method == 'urbanized_area_reg4_2100_SSP5')] = 'NEDUM'

plt.rcParams['figure.dpi'] = 400
sns.set(font_scale=5, style = "whitegrid", rc={'figure.figsize':(15, 10)})
g = sns.catplot(x = "Region", y = "urbanized_area", col = "SSP", data = df_all_ssp, kind="bar", col_wrap=2, height=10, aspect=15/10, palette = sns.color_palette("Set2"))
(g.set_axis_labels("", "Urbanized area")
  .set_titles("{col_name}")  # 
  .set_xticklabels(["East Asia", "Europe and Japan", "Land rich developed countries", "Latin America", "Northern Africa", "South and Central Asia", "Southeast Asia", "Sub-Saharan Africa", "Western Asia"], rotation=90)
  .despine(left=True, right = True, top = True)) 


### CATPLOT GUNERALP

results["pop_2015"] = np.nan
results["pop_2050_SSP1"] = np.nan
results["pop_2050_SSP2"] = np.nan
results["pop_2050_SSP3"] = np.nan
results["pop_2050_SSP4"] = np.nan
results["pop_2050_SSP5"] = np.nan

results.index = results.City
for city in np.delete(results.index, 153):
    results.loc[results.index == city, 'pop_2015'] = np.nansum(np.load(path_ssp1 + city + "_density.npy")[0])
    results.loc[results.index == city, 'pop_2050_SSP1'] = np.nansum(np.load(path_ssp1 + city + "_density.npy")[35])
    results.loc[results.index == city, 'pop_2050_SSP2'] = np.nansum(np.load(path_ssp2 + city + "_density.npy")[35])
    results.loc[results.index == city, 'pop_2050_SSP3'] = np.nansum(np.load(path_ssp3 + city + "_density.npy")[35])
    results.loc[results.index == city, 'pop_2050_SSP4'] = np.nansum(np.load(path_ssp4 + city + "_density.npy")[35])
    results.loc[results.index == city, 'pop_2050_SSP5'] = np.nansum(np.load(path_ssp5 + city + "_density.npy")[35])
    
results["density_reg4_2015"] = results["pop_2015"] / results["urbanized_area_reg4_2015"]
results["density_reg4_2050_SSP1"] = results["pop_2050_SSP1"] / results["urbanized_area_reg4_2050_SSP1"]
results["density_reg4_2050_SSP2"] = results["pop_2050_SSP2"] / results["urbanized_area_reg4_2050_SSP2"]
results["density_reg4_2050_SSP3"] = results["pop_2050_SSP3"] / results["urbanized_area_reg4_2050_SSP3"]
results["density_reg4_2050_SSP4"] = results["pop_2050_SSP4"] / results["urbanized_area_reg4_2050_SSP4"]
results["density_reg4_2050_SSP5"] = results["pop_2050_SSP5"] / results["urbanized_area_reg4_2050_SSP5"]
    
density_scenarios = results.loc[:, ['Country', 'density_reg4_2015', 'density_reg4_2050_SSP1', 'density_reg4_2050_SSP2', 'density_reg4_2050_SSP3', 'density_reg4_2050_SSP4', 'density_reg4_2050_SSP5']]

density_scenarios["region_guneralp"] = np.nan
density_scenarios["region_guneralp"][(density_scenarios.Country == 'Japan') | (density_scenarios.Country == 'Australia') | (density_scenarios.Country == 'New_Zealand')] = 'POECD'
density_scenarios["region_guneralp"][(density_scenarios.Country == 'USA')] = 'NAM'
density_scenarios["region_guneralp"][(density_scenarios.Country == 'India')|(density_scenarios.Country == 'Pakistan')] = 'SAS'
density_scenarios["region_guneralp"][(density_scenarios.Country == 'Mongolia') | (density_scenarios.Country == 'China')| (density_scenarios.Country == 'Vietnam')| (density_scenarios.Country == 'Hong_Kong')] = 'CPA'
density_scenarios["region_guneralp"][(density_scenarios.Country == 'Indonesia') | (density_scenarios.Country == 'Singapore') | (density_scenarios.Country == 'Thailand')] = 'PAS'
density_scenarios["region_guneralp"][(density_scenarios.Country == 'Argentina') |(density_scenarios.Country == 'Brazil') |(density_scenarios.Country == 'Chile') |(density_scenarios.Country == 'Peru')|(density_scenarios.Country == 'Colombia') |(density_scenarios.Country == 'Mexico')|(density_scenarios.Country == 'Uruguay') ] = 'LAC'
density_scenarios["region_guneralp"][(density_scenarios.Country == 'Turkey')|(density_scenarios.Country == 'Portugal')|(density_scenarios.Country == 'Norway')|(density_scenarios.Country == 'Netherlands')|(density_scenarios.Country == 'Italy')|(density_scenarios.Country == 'Switzerland')|(density_scenarios.Country == 'Finland')|(density_scenarios.Country == 'UK')|(density_scenarios.Country == 'Spain')|(density_scenarios.Country == 'Greece')|(density_scenarios.Country == 'Germany')|(density_scenarios.Country == 'France')|(density_scenarios.Country == 'Sweden')|(density_scenarios.Country == 'Ireland')|(density_scenarios.Country == 'Belgium')] = 'WEU'
density_scenarios["region_guneralp"][(density_scenarios.Country == 'South_Africa')|(density_scenarios.Country == 'Ethiopia')|(density_scenarios.Country == 'Ivory_Coast')] = 'SSA'
density_scenarios["region_guneralp"][(density_scenarios.Country == 'Slovenia')|(density_scenarios.Country == 'Romania')|(density_scenarios.Country == 'Poland')|(density_scenarios.Country == 'Croatia')|(density_scenarios.Country == 'Bulgaria')|(density_scenarios.Country == 'Hungary')|(density_scenarios.Country == 'Czech_Republic')] = 'EEU'
density_scenarios["region_guneralp"][(density_scenarios.Country == 'Armenia')|(density_scenarios.Country == 'Latvia')|(density_scenarios.Country == 'Russia')] = 'FSU'
density_scenarios["region_guneralp"][(density_scenarios.Country == 'Tunisia')|(density_scenarios.Country == 'Morocco')|(density_scenarios.Country == 'Iran')] = 'MNA'

plt.scatter(density_scenarios.density_reg4_2015, density_scenarios.density_reg4_2050_SSP1)
plt.scatter(density_scenarios.density_reg4_2015, density_scenarios.density_reg4_2050_SSP2)
plt.scatter(density_scenarios.density_reg4_2015, density_scenarios.density_reg4_2050_SSP3)
plt.scatter(density_scenarios.density_reg4_2015, density_scenarios.density_reg4_2050_SSP4)
plt.scatter(density_scenarios.density_reg4_2015, density_scenarios.density_reg4_2050_SSP5)

guneralp_S25 = pd.DataFrame(index = ['CPA', 'EEU', 'FSU', 'LAC', 'MNA', 'NAM', 'PAS', 'POECD', 'SAS', 'SSA', 'WEU'], columns = ['guneralp_2015_S25', 'guneralp_2050_S25'])
guneralp_S50 = pd.DataFrame(index = ['CPA', 'EEU', 'FSU', 'LAC', 'MNA', 'NAM', 'PAS', 'POECD', 'SAS', 'SSA', 'WEU'], columns = ['guneralp_2015_S50', 'guneralp_2050_S50'])
guneralp_S75 = pd.DataFrame(index = ['CPA', 'EEU', 'FSU', 'LAC', 'MNA', 'NAM', 'PAS', 'POECD', 'SAS', 'SSA', 'WEU'], columns = ['guneralp_2015_S75', 'guneralp_2050_S75'])
  
guneralp_S25["guneralp_2015_S25"] = [48.17, 40.78, 54.48, 86.27, 99.73, 18.09, 89.54, 77.62, 141.16, 69.82, 39.11]
guneralp_S50["guneralp_2015_S50"] = [80.05, 44.66, 61.78, 98.71, 120.66, 20.11, 118.55, 96.53, 172.77, 90.79, 48.46]
guneralp_S75["guneralp_2015_S75"] = [125.80, 50.50, 73.03, 112.65, 144.32, 22.46, 155.21, 116.71, 209.59, 108.87, 55.69] 
    
guneralp_S25["guneralp_2050_S25"] = [10.28, 18.71, 31.28, 53.08, 46.71, 10.21, 30.76, 30.39, 47.73, 46.85, 14.16]
guneralp_S50["guneralp_2050_S50"] = [55.86, 25.32, 48.13, 81.12, 89.36, 14.55, 77.70, 59.93, 91.56, 113.50, 28.64]
guneralp_S75["guneralp_2050_S75"] = [252.14, 38.16, 85.45, 128.92, 173.10, 21.04, 185.31, 104.91, 175.73, 208.86, 46.68]
      

#aggregated_by_region = density_scenarios.loc[:, ['region_guneralp','density_reg4_2015', 'density_reg4_2050_SSP1', 'density_reg4_2050_SSP2', 'density_reg4_2050_SSP3', 'density_reg4_2050_SSP4', 'density_reg4_2050_SSP5']].groupby('region_guneralp').mean()
density_scenarios['predicted_density_2015'] = density_scenarios['predicted_density_2015'].astype(float)
density_scenarios['predicted_density_2035'] = density_scenarios['predicted_density_2035'].astype(float)
density_scenarios['predicted_density_2015_corrected'] = density_scenarios['predicted_density_2015_corrected'].astype(float)
density_scenarios['predicted_density_2035_corrected'] = density_scenarios['predicted_density_2035_corrected'].astype(float)
aggregated_by_region = density_scenarios.loc[:, ['region_guneralp','predicted_density_2015_corrected', 'predicted_density_2035_corrected']].groupby('region_guneralp').mean()

      
aggregated_by_region = aggregated_by_region.merge(guneralp_S25, left_index = True, right_index = True).merge(guneralp_S50, left_index = True, right_index = True).merge(guneralp_S75, left_index = True, right_index = True)

aggregated_by_region["guneralp_S25"] = aggregated_by_region["guneralp_2050_S25"] / aggregated_by_region["guneralp_2015_S25"]
aggregated_by_region["guneralp_S50"] = aggregated_by_region["guneralp_2050_S50"] / aggregated_by_region["guneralp_2015_S50"]
aggregated_by_region["guneralp_S75"] = aggregated_by_region["guneralp_2050_S75"] / aggregated_by_region["guneralp_2015_S75"]

aggregated_by_region["predicted_density_2035"] = aggregated_by_region["predicted_density_2035"] / aggregated_by_region["predicted_density_2015"]
aggregated_by_region["density_reg4_2050_SSP2"] = aggregated_by_region["density_reg4_2050_SSP2"] / aggregated_by_region["density_reg4_2015"]
aggregated_by_region["density_reg4_2050_SSP3"] = aggregated_by_region["density_reg4_2050_SSP3"] / aggregated_by_region["density_reg4_2015"]
aggregated_by_region["density_reg4_2050_SSP4"] = aggregated_by_region["density_reg4_2050_SSP4"] / aggregated_by_region["density_reg4_2015"]
aggregated_by_region["density_reg4_2050_SSP5"] = aggregated_by_region["density_reg4_2050_SSP5"] / aggregated_by_region["density_reg4_2015"]

aggregated_by_region["region_guneralp"] = aggregated_by_region.index

df_ssp1= pd.melt(aggregated_by_region.loc[:, ['predicted_density_2035', 'region_guneralp']], ["region_guneralp"], var_name="method", value_name="density")
df_ssp1["SSP"] = "SSP1"

df_ssp1= pd.melt(aggregated_by_region.loc[:, ['density_reg4_2050_SSP1', 'region_guneralp']], ["region_guneralp"], var_name="method", value_name="density")
df_ssp1["SSP"] = "SSP1"

df_ssp2= pd.melt(aggregated_by_region.loc[:, ['density_reg4_2050_SSP2', 'region_guneralp']], ["region_guneralp"], var_name="method", value_name="density")
df_ssp2["SSP"] = "SSP2"

df_ssp3= pd.melt(aggregated_by_region.loc[:, ['density_reg4_2050_SSP3', 'region_guneralp']], ["region_guneralp"], var_name="method", value_name="density")
df_ssp3["SSP"] = "SSP3"

df_ssp4= pd.melt(aggregated_by_region.loc[:, ['density_reg4_2050_SSP4', 'region_guneralp']], ["region_guneralp"], var_name="method", value_name="density")
df_ssp4["SSP"] = "SSP4"

df_ssp5= pd.melt(aggregated_by_region.loc[:, ['density_reg4_2050_SSP5', 'region_guneralp']], ["region_guneralp"], var_name="method", value_name="density")
df_ssp5["SSP"] = "SSP5"

df_S25= pd.melt(aggregated_by_region.loc[:, ['guneralp_S25', 'region_guneralp']], ["region_guneralp"], var_name="method", value_name="density")
df_S25["SSP"] = "S25"

df_S50= pd.melt(aggregated_by_region.loc[:, ['guneralp_S50', 'region_guneralp']], ["region_guneralp"], var_name="method", value_name="density")
df_S50["SSP"] = "S50"

df_S75= pd.melt(aggregated_by_region.loc[:, ['guneralp_S75', 'region_guneralp']], ["region_guneralp"], var_name="method", value_name="density")
df_S75["SSP"] = "S75"



df_all_ssp = df_ssp1.append(df_ssp2).append(df_ssp3).append(df_ssp4).append(df_ssp5).append(df_S25).append(df_S50).append(df_S75)
#df_all_ssp.method[(df_all_ssp.method == 'Huang_2050_SSP1')|(df_all_ssp.method == 'Huang_2050_SSP2')|(df_all_ssp.method == 'Huang_2050_SSP3')|(df_all_ssp.method == 'Huang_2050_SSP4')|(df_all_ssp.method == 'Huang_2050_SSP5')] = 'Huang'
#df_all_ssp.method[(df_all_ssp.method == 'urbanized_area_reg4_2100_SSP1')|(df_all_ssp.method == 'urbanized_area_reg4_2100_SSP2')|(df_all_ssp.method == 'urbanized_area_reg4_2100_SSP3')|(df_all_ssp.method == 'urbanized_area_reg4_2100_SSP4')|(df_all_ssp.method == 'urbanized_area_reg4_2100_SSP5')] = 'NEDUM'

df_all_ssp = df_ssp1.append(df_S25).append(df_S50).append(df_S75)

plt.bar(np.arange(-0.3, 10, 1), df_all_ssp.density[df_all_ssp.method == "guneralp_S25"], width = 0.2)
plt.bar(np.arange(-0.1, 10, 1), df_all_ssp.density[df_all_ssp.method == "guneralp_S50"], width = 0.2)
plt.bar(np.arange(0.1, 11, 1), df_all_ssp.density[df_all_ssp.method == "guneralp_S75"], width = 0.2)
plt.bar(np.arange(0.3, 11, 1), df_all_ssp.density[df_all_ssp.method == "predicted_density_2035"], width = 0.2)
plt.legend()

plt.rcParams['figure.dpi'] = 400
sns.set(font_scale=5, style = "whitegrid", rc={'figure.figsize':(15, 10)})
g = sns.catplot(x = "region_guneralp", y = "density", col = "SSP", data = df_all_ssp, kind="bar", col_wrap=2, height=10, aspect=15/10, palette = sns.color_palette("Set2"))
(g.set_axis_labels("", "Urbanized area")
  .set_titles("{col_name}")  # 
  #.set_xticklabels(["East Asia", "Europe and Japan", "Land rich developed countries", "Latin America", "Northern Africa", "South and Central Asia", "Southeast Asia", "Sub-Saharan Africa", "Western Asia"], rotation=90)
  .despine(left=True, right = True, top = True)) 


y = np.array(((results["urbanized_area_reg4_2050_SSP2"] / results["urbanized_area_reg4_2015"]).loc[results.final_sample ==1]))
X = pd.DataFrame((results["pop_2050_SSP2"]/results["pop_2015"]).loc[results.final_sample ==1])
ols = sm.OLS(y, X)
reg1 = ols.fit(cov_type='HC1') #HC3 censé être mieux
reg1.summary()





