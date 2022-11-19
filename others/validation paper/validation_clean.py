# -*- coding: utf-8 -*-
"""
Created on Fri May 21 12:04:23 2021

@author: charl
"""

#### PAPIER VALIDATION ####



import pandas as pd
import numpy as np
import copy
import pickle
import scipy as sc
from scipy import optimize
import matplotlib.pyplot as plt
import os
import math

path_data = "C:/Users/charl/OneDrive/Bureau/City_dataStudy/"
path_folder = "C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Data/"


#On commence par importer les résultats des équations 1, 2 et 3
path_outputs = "C:/Users/charl/OneDrive/Bureau/"
df = pd.read_excel(path_outputs + 'mitigation_policies_city_characteristics/Sorties/robustness/resultsavg.xlsx')

#Stats des sur les R2
percentiles_R2 = pd.DataFrame(columns = ["reg1", "reg1b", "reg2", "reg3"], index = ["10", "25", "50", "75", "90"])
percentiles_R2["reg1"] = [np.nanpercentile(df.reg1_r2, 10), np.nanpercentile(df.reg1_r2, 25), np.nanpercentile(df.reg1_r2, 50), np.nanpercentile(df.reg1_r2, 75), np.nanpercentile(df.reg1_r2, 90)]
percentiles_R2["reg1b"] = [np.nanpercentile(df.reg1b_r2, 10), np.nanpercentile(df.reg1b_r2, 25), np.nanpercentile(df.reg1b_r2, 50), np.nanpercentile(df.reg1b_r2, 75), np.nanpercentile(df.reg1b_r2, 90)]
percentiles_R2["reg2"] = [np.nanpercentile(df.reg2_r2, 10), np.nanpercentile(df.reg2_r2, 25), np.nanpercentile(df.reg2_r2, 50), np.nanpercentile(df.reg2_r2, 75), np.nanpercentile(df.reg2_r2, 90)]
percentiles_R2["reg3"] = [np.nanpercentile(df.reg3_r2, 10), np.nanpercentile(df.reg3_r2, 25), np.nanpercentile(df.reg3_r2, 50), np.nanpercentile(df.reg3_r2, 75), np.nanpercentile(df.reg3_r2, 90)]
print(percentiles_R2.to_latex())
print(np.nanmin(df.reg1_r2))

#Analyses villes
print(sum(df.reg1_pvalues_X > 0.05))
print(sum(df.reg1b_pvalues_X > 0.05))
print(sum(df.reg2_pvalues_X > 0.05))
print(sum(df.reg3_pvalues_X > 0.05))

print(sum((df.reg1_pvalues_X < 0.05) & (df.reg1_params_X < 0)))
print(sum((df.reg1b_pvalues_X < 0.05) & (df.reg1b_params_X < 0)))
print(sum((df.reg2_pvalues_X < 0.05) & (df.reg2_params_X < 0)))
print(sum((df.reg3_pvalues_X < 0.05) & (df.reg3_params_X < 0)))

print(sum((df.reg1_pvalues_X < 0.05) & (df.reg1_params_X > 0)))
print(sum((df.reg1b_pvalues_X < 0.05) & (df.reg1b_params_X > 0)))
print(sum((df.reg2_pvalues_X < 0.05) & (df.reg2_params_X > 0)))
print(sum((df.reg3_pvalues_X < 0.05) & (df.reg3_params_X > 0)))


data = pd.DataFrame(columns = ['City', 'density', 'real_estate', 'housing_production'])
data.City = df.city
data.density = "positive"
data.density[df.reg1_pvalues_X > 0.05] = "non_significant"
data.density[((df.reg1_pvalues_X < 0.05) & (df.reg1_params_X < 0))] = "negative"
data.real_estate = "positive"
data.real_estate[df.reg3_pvalues_X > 0.05] = "non_significant"
data.real_estate[((df.reg3_pvalues_X < 0.05) & (df.reg3_params_X < 0))] = "negative"
data.housing_production = "positive"
data.housing_production[df.reg2_pvalues_X > 0.05] = "non_significant"
data.housing_production[((df.reg2_pvalues_X < 0.05) & (df.reg2_params_X < 0))] = "negative"
data.housing_production[data.City == 'Sfax'] = np.nan
data.real_estate[data.City == 'Sfax'] = np.nan
data.density[data.City == 'Sfax'] = np.nan
data.to_excel("C:/Users/charl/OneDrive/Bureau/maps_validation_paper.xlsx")

#SECOND STAGE
second_stage_reg = df.loc[:, ["city", "reg1_r2", "reg1_params_X", "reg1_pvalues_X", "reg2_r2", "reg2_params_X", "reg2_pvalues_X", "reg3_r2","reg3_params_X", "reg3_pvalues_X"]]

city_continent = pd.read_csv(path_folder + "CityDatabases/cityDatabase_221NewCities.csv")
city_continent = city_continent.iloc[:, 0:2]
city_continent = city_continent.drop_duplicates(subset = "City")
city_continent = city_continent.sort_values('City')
second_stage_reg = second_stage_reg.merge(city_continent, left_on = "city", right_on = "City")

#Gini index
gini = pd.read_csv(path_folder + "API_SI.POV.GINI_DS2_en_csv_v2_2252167.csv", header = 2)

gini["Country Name"][gini["Country Name"] == "Cote d\'Ivoire"] = "Ivory_Coast"
gini["Country Name"][gini["Country Name"] == "United States"] = "USA"
gini["Country Name"][gini["Country Name"] == "New Zealand"] = "New_Zealand"
gini["Country Name"][gini["Country Name"] == "United Kingdom"] = "UK"
gini["Country Name"][gini["Country Name"] == "South Africa"] = "South_Africa"
gini["Country Name"][gini["Country Name"] == "Russian Federation"] = "Russia"
gini["Country Name"][gini["Country Name"] == "Hong Kong SAR, China"] = "Hong_Kong"
gini["Country Name"][gini["Country Name"] == "Iran, Islamic Rep."] = "Iran"
gini["Country Name"][gini["Country Name"] == "Czech Republic"] = "Czech_Republic"
gini["2019"][np.isnan(gini["2019"])] = gini["2018"]
gini["2019"][np.isnan(gini["2019"])] = gini["2017"]
gini["2019"][np.isnan(gini["2019"])] = gini["2016"]
gini["2019"][np.isnan(gini["2019"])] = gini["2015"]
gini["2019"][np.isnan(gini["2019"])] = gini["2014"]
gini["2019"][np.isnan(gini["2019"])] = gini["2013"]
gini["2019"][np.isnan(gini["2019"])] = gini["2012"]
gini["2019"][np.isnan(gini["2019"])] = gini["2011"]
gini["2019"][np.isnan(gini["2019"])] = gini["2010"]
gini = gini[["Country Name", "2019"]]
gini.columns = ["Country", "gini"]
second_stage_reg = second_stage_reg.merge(gini, on = "Country")


#Polycentrism (+ population et income ?)
df = pd.read_excel("C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/dataset_regressions_v2.xlsx")
df["constant"] = np.ones(192)
df = df.iloc[:,[1, 3, 4, 5, 6, 7,  8, 9]]
second_stage_reg = second_stage_reg.merge(df, on = "city")

#Informal housing
informal_housing = pd.read_csv(path_folder + "API_EN.POP.SLUM.UR.ZS_DS2_en_csv_v2_2257750.csv", header = 2)
informal_housing["Country Name"][informal_housing["Country Name"] == "Cote d\'Ivoire"] = "Ivory_Coast"
informal_housing["Country Name"][informal_housing["Country Name"] == "United States"] = "USA"
informal_housing["Country Name"][informal_housing["Country Name"] == "New Zealand"] = "New_Zealand"
informal_housing["Country Name"][informal_housing["Country Name"] == "United Kingdom"] = "UK"
informal_housing["Country Name"][informal_housing["Country Name"] == "South Africa"] = "South_Africa"
informal_housing["Country Name"][informal_housing["Country Name"] == "Russian Federation"] = "Russia"
informal_housing["Country Name"][informal_housing["Country Name"] == "Hong Kong SAR, China"] = "Hong_Kong"
informal_housing["Country Name"][informal_housing["Country Name"] == "Iran, Islamic Rep."] = "Iran"
informal_housing["Country Name"][informal_housing["Country Name"] == "Czech Republic"] = "Czech_Republic"
informal_housing["2019"][np.isnan(informal_housing["2019"])] = informal_housing["2018"]
informal_housing["2019"][np.isnan(informal_housing["2019"])] = informal_housing["2017"]
informal_housing["2019"][np.isnan(informal_housing["2019"])] = informal_housing["2016"]
informal_housing["2019"][np.isnan(informal_housing["2019"])] = informal_housing["2015"]
informal_housing["2019"][np.isnan(informal_housing["2019"])] = informal_housing["2014"]
informal_housing["2019"][np.isnan(informal_housing["2019"])] = informal_housing["2013"]
informal_housing["2019"][np.isnan(informal_housing["2019"])] = informal_housing["2012"]
informal_housing["2019"][np.isnan(informal_housing["2019"])] = informal_housing["2011"]
informal_housing["2019"][np.isnan(informal_housing["2019"])] = informal_housing["2010"]
informal_housing = informal_housing[["Country Name", "2019"]]
informal_housing.columns = ["Country", "informal_housing"]
informal_housing.informal_housing[np.isnan(informal_housing.informal_housing)] = 0
second_stage_reg = second_stage_reg.merge(informal_housing, on = "Country")

#Informal jobs
informal_jobs = pd.read_csv(path_folder + "SDG_0831_SEX_ECO_RT_A-filtered-2021-04-28.csv")
informal_jobs = informal_jobs.loc[:,["ref_area.label", "time", "obs_value"]]
for country in np.unique(informal_jobs["ref_area.label"]):        
        most_recent_data = max(informal_jobs.time[informal_jobs["ref_area.label"] == country])
        i = informal_jobs[((informal_jobs["ref_area.label"] == country) & 
                       (informal_jobs.time < most_recent_data))].index
        informal_jobs = informal_jobs.drop(i)
informal_jobs = informal_jobs.loc[:, ["ref_area.label", "obs_value"]]
informal_jobs.columns = ["Country", "informal_jobs"]
informal_jobs.Country[informal_jobs.Country == "South Africa"] = "South_Africa"
informal_jobs.Country[informal_jobs.Country == "Côte d'Ivoire"] = "Ivory_Coast"
second_stage_reg = second_stage_reg.merge(informal_jobs, on = "Country", how = "left")
second_stage_reg.informal_jobs[np.isnan(second_stage_reg.informal_jobs)] = 0

## Number of data points for rents
robustness = pd.DataFrame(columns = ["spatial_data_cover", "market_data_cover"], index = second_stage_reg.city)

for city in np.unique(list_city.City):
    
    (country, proj, density, rents_and_size, land_use, land_cover_ESACCI, driving, transit, grille, 
     centre, distance_cbd, conversion_rate) = import_data(list_city, path_data, city, path_folder)        
    density = density.loc[:,density.columns.str.startswith("density")]
    density = np.array(density).squeeze()
    #robustness["nb_of_cells_rents"].loc[city] = sum(~np.isnan(rents_and_size.medRent))
    robustness["spatial_data_cover"].loc[city] = sum(~np.isnan(rents_and_size.medRent)) / sum((pd.to_numeric(density)) > 0) 
    #robustness["avg_data"].loc[city] = np.nanmean(rents_and_size.dataCount)
    robustness["market_data_cover"].loc[city] = np.nansum(density) / np.nansum(rents_and_size.dataCount)
    
robustness = robustness.apply(pd.to_numeric)
second_stage_reg = second_stage_reg.merge(robustness, on = "city")

sea_regulation = pd.read_excel(path_folder + "sea_planification.xlsx", header = 0)
#sea_regulation["planification"] = sea_regulation.wiki
sea_regulation["planification"] = 2 * sea_regulation.strong_regulation + sea_regulation.low_regulation
sea_regulation["planification"][np.isnan(sea_regulation["planification"])] = sea_regulation.wiki
#sea_regulation["planification"].loc[sea_regulation.city == "Singapore"] = 1
sea_regulation = sea_regulation.loc[:, ["city", "planification", "sea"]]
second_stage_reg = second_stage_reg.merge(sea_regulation, on = "city")

#second_stage_reg.gini[np.isnan(second_stage_reg.gini)] = np.nanmean(second_stage_reg.gini)
second_stage_reg.gini[second_stage_reg.Country == "New_Zealand"] = 36.9
second_stage_reg.gini[second_stage_reg.Country == "Hong_Kong"] = 53.9
second_stage_reg.gini[second_stage_reg.Country == "Singapore"] = 45.9


second_stage_reg["dummy_reg2"] = 'non_significant'
second_stage_reg["dummy_reg2"][(second_stage_reg.reg2_pvalues_X < 0.05) & (second_stage_reg.reg2_params_X < 0)] = 'negative'
second_stage_reg["dummy_reg2"][(second_stage_reg.reg2_pvalues_X < 0.05) & (second_stage_reg.reg2_params_X > 0)] = 'apositive'


y = np.array(second_stage_reg["dummy_reg2"])
X = pd.DataFrame(second_stage_reg.iloc[:, [12, 13, 14, 18, 19, 20, 22, 23, 24, 25]])
ols = sm.MNLogit(y, X)
reg5 = ols.fit(cov_type='HC0') #HC3 censé être mieux
reg5.summary() #Avec ou sans Gini ? Plutôt sans.
print(reg5.summary().as_latex())

second_stage_reg["dummy_reg3"] = 'non_significant'
second_stage_reg["dummy_reg3"][(second_stage_reg.reg3_pvalues_X < 0.05) & (second_stage_reg.reg3_params_X < 0)] = 'negative'
second_stage_reg["dummy_reg3"][(second_stage_reg.reg3_pvalues_X < 0.05) & (second_stage_reg.reg3_params_X > 0)] = 'apositive'


y = np.array(second_stage_reg.reg3_pvalues_X[(second_stage_reg.dummy_reg3 == "non_significant") |(second_stage_reg.dummy_reg3 == "apositive")] > 0.05)
X = pd.DataFrame(second_stage_reg[(second_stage_reg.dummy_reg3 == "non_significant") |(second_stage_reg.dummy_reg3 == "apositive")].iloc[:, [12, 13, 14, 18, 19, 20, 22, 23, 24, 25]])
ols = sm.OLS(y, X)
reg6 = ols.fit(cov_type='HC1') #HC3 censé être mieux
reg6.summary()
from stargazer.stargazer import Stargazer
print(Stargazer([reg6]).render_latex())
