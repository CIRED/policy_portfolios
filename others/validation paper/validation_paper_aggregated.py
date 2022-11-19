# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 14:07:02 2021

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
from stargazer.stargazer import Stargazer

from functions import *

path_data = "C:/Users/charl/OneDrive/Bureau/City_dataStudy/"
path_folder = "C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Data/"
path_outputs = "C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/calibration_20210415/"
list_city = list_of_cities_and_databases(path_data,'cityDatabase')#cityDatabase_intermediaire

d_urbanized_area = {}
d_population = {}
d_income = {}
d_land_price = {}
d_commuting_price = {}
d_commuting_time = {}
d_polycentricity = {}

for city in np.unique(list_city.City):
    country = list_city.Country[list_city.City == city].iloc[0]
    proj = str(int(list_city.GridEPSG[list_city.City == city].iloc[0]))

    #urbanized area
    land_use = pd.read_csv(path_data + 'Data/' + country + '/' + city + 
                           '/Land_Cover/grid_ESACCI_LandCover_2015_' + 
                           str.upper(city) + '_' + proj +'.csv')
    d_urbanized_area[city] = np.nansum(land_use.ESACCI190) / 1000000

    #Population
    density = pd.read_csv(path_data + 'Data/' + country + '/' + city +
                          '/Population_Density/grille_GHSL_density_2015_' +
                          str.upper(city) + '.txt', sep = '\s+|,', engine='python')
    density = density.loc[:,density.columns.str.startswith("density")],
    density = np.array(density).squeeze()
    d_population[city] = np.nansum(density)

    #Incomes
    data_gdp = pd.read_csv(path_folder + "API_NY.GDP.PCAP.CD_DS2_en_csv_v2_2163510.csv", header =2) 
    data_gdp["Country Name"][data_gdp["Country Name"] == "Cote d\'Ivoire"] = "Ivory_Coast"
    data_gdp["Country Name"][data_gdp["Country Name"] == "United States"] = "USA"
    data_gdp["Country Name"][data_gdp["Country Name"] == "New Zealand"] = "New_Zealand"
    data_gdp["Country Name"][data_gdp["Country Name"] == "United Kingdom"] = "UK"
    data_gdp["Country Name"][data_gdp["Country Name"] == "South Africa"] = "South_Africa"
    data_gdp["Country Name"][data_gdp["Country Name"] == "Russian Federation"] = "Russia"
    data_gdp["Country Name"][data_gdp["Country Name"] == "Hong Kong SAR, China"] = "Hong_Kong"
    data_gdp["Country Name"][data_gdp["Country Name"] == "Iran, Islamic Rep."] = "Iran"
    data_gdp["Country Name"][data_gdp["Country Name"] == "Czech Republic"] = "Czech_Republic"
    d_income[city] = data_gdp["2018"][data_gdp["Country Name"] == country].squeeze()
    
    #Land prices
    d_land_price[city] = import_agricultural_rent(path_folder, country)

    #Commuting times
    transport_source = list_city.TransportSource[list_city.City == city].iloc[0]
    hour = list_city.RushHour[list_city.City == city].iloc[0]
    driving = pd.read_csv(path_data + 'Data/' +country + '/' + city + 
                      '/Transport/interpDrivingTimes' + transport_source + '_' 
                      + city + '_' + hour + '_' + proj +'.csv')
    
    transit = pd.read_csv(path_data + 'Data/' +country + '/' + city + 
                      '/Transport/interpTransitTimesGoogle_'+ city + '_' + 
                      hour + '_' + proj + '.csv')
    
    speed_driving = (driving.Distance / 1000) / (driving.Duration / 3600)
    speed_transit = (transit.Distance / 1000) / (transit.Duration / 3600)
    max_speed = np.fmax(speed_driving, speed_transit)
    d_commuting_time[city] = np.nansum(max_speed * density / np.nansum(density))
    
    #Modal share/Access to public transport ? Or average speed to go to the city center?
    #Diesel prices
    fuel_price = import_fuel_price(country, 'gasoline', path_folder) #L/100km
    #fuel_consumption = import_fuel_conso(country, path_folder) #dollars/L
    d_commuting_price[city] = fuel_price # * fuel_consumption / 100
    
    #Polycentricity index
    polycentricity = pd.read_excel(path_data + 'Article/Data_Criterias/CBD_Criterias_Table.ods', engine="odf")
    polycentricity = polycentricity.iloc[:, [2, 16]]
    polycentricity.columns = ['city', 'polycentricity_index']
    polycentricity.city[polycentricity.city == "Addis Ababa"] = "Addis_Ababa"
    polycentricity.city[polycentricity.city == "Belo Horizonte"] = "Belo_Horizonte"
    polycentricity.city[polycentricity.city == "Buenos Aires"] = "Buenos_Aires"
    polycentricity.city[polycentricity.city == "Cape Town"] = "Cape_Town"
    polycentricity.city[polycentricity.city == "Chiang Mai"] = "Chiang_Mai"
    polycentricity.city[polycentricity.city == "Cluj-Napoca"] = "Cluj_Napoca"
    polycentricity.city[polycentricity.city == "Frankfurt am Main"] = "Frankfurt_am_Main"
    polycentricity.city[polycentricity.city == "Goiânia"] = "Goiania"
    polycentricity.city[polycentricity.city == "Hong Kong"] = "Hong_Kong"
    polycentricity.city[polycentricity.city == "Los Angeles"] = "Los_Angeles"
    polycentricity.city[polycentricity.city == "Malmö"] = "Malmo"
    polycentricity.city[polycentricity.city == "Mar del Plata"] = "Mar_del_Plata"
    polycentricity.city[polycentricity.city == "Mexico City"] = "Mexico_City"
    polycentricity.city[polycentricity.city == "New York"] = "New_York"
    polycentricity.city[polycentricity.city == "Nizhny Novgorod"] = "Nizhny_Novgorod"
    polycentricity.city[polycentricity.city == "Porto Alegre"] = "Porto_Alegre"
    polycentricity.city[polycentricity.city == "Rio de Janeiro"] = "Rio_de_Janeiro"
    polycentricity.city[polycentricity.city == "Rostov-on-Don"] = "Rostov_on_Don"
    polycentricity.city[polycentricity.city == "San Diego"] = "San_Diego"
    polycentricity.city[polycentricity.city == "San Fransisco"] = "San_Fransisco"
    polycentricity.city[polycentricity.city == "Sao Paulo"] = "Sao_Paulo"
    polycentricity.city[polycentricity.city == "St Petersburg"] = "St_Petersburg"
    polycentricity.city[polycentricity.city == "The Hague"] = "The_Hague"
    polycentricity.city[polycentricity.city == "Ulan Bator"] = "Ulan_Bator"
    polycentricity.city[polycentricity.city == "Washington DC"] = "Washington_DC"
    polycentricity.city[polycentricity.city == "Zürich"] = "Zurich"
    d_polycentricity[city] = polycentricity.polycentricity_index[polycentricity.city == city].squeeze()
    
df = pd.DataFrame()
df["city"] = d_urbanized_area.keys()
df["urbanized_area"] = d_urbanized_area.values()
df["population"] = d_population.values()
df["income"] = d_income.values()
df["land_prices"] = d_land_price.values()
df["commuting_price"] = d_commuting_price.values()
df["commuting_time"] = d_commuting_time.values()
df["polycentricity"] = d_polycentricity.values()

df.to_excel("C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/dataset_regressions_v3.xlsx")
df = pd.read_excel("C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/dataset_regressions_v2.xlsx")

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

y = np.array((df.urbanized_area))
X = pd.DataFrame(df.iloc[:, [2, 3, 4, 5, 6, 7, 8]])
ols = sm.OLS(y, X)
reg1 = ols.fit(cov_type='HC1') #HC3 censé être mieux
reg1.summary()

y = np.array((df.log_urbanized_area))
X = pd.DataFrame(df.iloc[:, [17, 18, 19, 20, 21, 7, 8]])
ols = sm.OLS(y, X)
reg2 = ols.fit(cov_type='HC1') #HC3 censé être mieux
reg2.summary()

print(Stargazer([reg1, reg2]).render_latex())

y = np.array(df.urbanized_area[df.Continent == "Europe"])
X = (pd.DataFrame(df.loc[df.Continent == "Europe",["population", "income", "land_prices", "commuting_time", "polycentricity", "constant"]]))
ols = sm.OLS(y, X)
regEurope = ols.fit(cov_type='HC1') #HC3 censé être mieux
regEurope.summary()

y = np.array(df.urbanized_area[df.Continent == "North_America"])
X = (pd.DataFrame(df.loc[df.Continent == "North_America",["population", "income", "land_prices", "commuting_time", "polycentricity", "constant"]]))
ols = sm.OLS(y, X)
North_America = ols.fit(cov_type='HC1') #HC3 censé être mieux
North_America.summary()

y = np.array(df.urbanized_area[df.Continent == "Oceania"])
X = (pd.DataFrame(df.loc[df.Continent == "Oceania",["population", "income", "land_prices", "commuting_time", "polycentricity", "constant"]]))

ols = sm.OLS(y, X)
Oceania = ols.fit(cov_type='HC1') #HC3 censé être mieux
Oceania.summary()

y = np.array(df.urbanized_area[df.Continent == "South_America"])
X = (pd.DataFrame(df.loc[df.Continent == "South_America",["population", "income", "land_prices", "commuting_time", "polycentricity", "constant"]]))

ols = sm.OLS(y, X)
South_America = ols.fit(cov_type='HC1') #HC3 censé être mieux
South_America.summary()

y = np.array(df.urbanized_area[df.Continent == "Africa"])
X = (pd.DataFrame(df.loc[df.Continent == "Africa",["population", "income", "land_prices", "commuting_time", "polycentricity", "constant"]]))

ols = sm.OLS(y, X)
Africa = ols.fit(cov_type='HC1') #HC3 censé être mieux
Africa.summary()

y = np.array(df.log_urbanized_area[df.Continent == "Asia"])
X = (pd.DataFrame(df.loc[df.Continent == "Asia",["population", "income", "land_prices", "commuting_time", "polycentricity", "constant"]]))

ols = sm.OLS(y, X)
Asia = ols.fit(cov_type='HC1') #HC3 censé être mieux
Asia.summary()

print(Stargazer([North_America, regEurope, Oceania, South_America, Asia, Africa]).render_latex())


df["resid"] = reg1.resid

color_tab = {'North_America':'red', 'Europe':'green', 'Asia':'blue', 'Oceania':'yellow', 'South_America': 'brown', 'Africa': 'orange'}
plt.rcParams.update({'font.size': 50})
plt.figure(figsize=(20, 20))
colors = list(df['Continent'].unique())
for i in range(0 , len(colors)):
    data = df.loc[df['Continent'] == colors[i]]
    plt.scatter(np.exp(data.log_urbanized_area), np.exp(data.log_urbanized_area + data.resid), color=data.Continent.map(color_tab), label=colors[i], s = 250)
plt.rcParams.update({'font.size': 50})
plt.legend() 
plt.xlim(0, 12000)
plt.ylim(0, 12000)
plt.plot(np.arange(0, 12000), np.arange(0, 12000))
plt.xlabel("Data")
plt.ylabel("Predictions")
plt.show()

'''
df = df.iloc[:, [0, 1, 16]]
second_stage_urbaized_area = df.merge(second_stage_reg, on = "city")
y = np.array((np.abs(second_stage_urbaized_area.resid / second_stage_urbaized_area.urbanized_area)))
X = pd.DataFrame(second_stage_urbaized_area.iloc[:, [14,15, 16,20, 21, 22, 23, 27, 28]])
ols = sm.OLS(y, X)
reg2 = ols.fit(cov_type='HC1') #HC3 censé être mieux
reg2.summary()

y = np.array((df.urbanized_area))
X = pd.DataFrame(df.iloc[:, [2,3,4, 6,7,8, 11, 12, 13, 14, 15]])
ols = sm.OLS(y, X)
reg2 = ols.fit(cov_type='HC1') #HC3 censé être mieux
reg2.summary()

y = np.array((df.urbanized_area))
X = pd.DataFrame(df.iloc[:, [2,3,4, 5,7,8, 11, 12, 13, 14, 15]])
ols = sm.OLS(y, X)
reg3 = ols.fit(cov_type='HC1') #HC3 censé être mieux
reg3.summary()

from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns
#Moderate correlations

from stargazer.stargazer import Stargazer
print(Stargazer([reg2]).render_latex())

plt.scatter(df.urbanized_area, df.urbanized_area + df.resid)
plt.xlabel("Data")
plt.ylabel("Predictions")


y = np.array(df.urbanized_area[df.Continent == "Europe"])
X = (pd.DataFrame(df.loc[df.Continent == "Europe",["population", "income", "land_prices", "commuting_time", "polycentricity", "constant"]]))
ols = sm.OLS(y, X)
regEurope = ols.fit(cov_type='HC1') #HC3 censé être mieux
regEurope.summary()

y = np.array(df.urbanized_area[df.Continent == "North_America"])
X = (pd.DataFrame(df.loc[df.Continent == "North_America",["population", "income", "land_prices", "commuting_time", "polycentricity", "constant"]]))
ols = sm.OLS(y, X)
North_America = ols.fit(cov_type='HC1') #HC3 censé être mieux
North_America.summary()

y = np.array(df.urbanized_area[df.Continent == "Oceania"])
X = (pd.DataFrame(df.loc[df.Continent == "Oceania",["population", "income", "land_prices", "commuting_time", "polycentricity", "constant"]]))

ols = sm.OLS(y, X)
Oceania = ols.fit(cov_type='HC1') #HC3 censé être mieux
Oceania.summary()

y = np.array(df.urbanized_area[df.Continent == "South_America"])
X = (pd.DataFrame(df.loc[df.Continent == "South_America",["population", "income", "land_prices", "commuting_time", "polycentricity", "constant"]]))

ols = sm.OLS(y, X)
South_America = ols.fit(cov_type='HC1') #HC3 censé être mieux
South_America.summary()

y = np.array(df.urbanized_area[df.Continent == "Africa"])
X = (pd.DataFrame(df.loc[df.Continent == "Africa",["population", "income", "land_prices", "commuting_time", "polycentricity", "constant"]]))

ols = sm.OLS(y, X)
Africa = ols.fit(cov_type='HC1') #HC3 censé être mieux
Africa.summary()

y = np.array(df.log_urbanized_area[df.Continent == "Asia"])
X = (pd.DataFrame(df.loc[df.Continent == "Asia",["population", "income", "land_prices", "commuting_time", "polycentricity", "constant"]]))

ols = sm.OLS(y, X)
Asia = ols.fit(cov_type='HC1') #HC3 censé être mieux
Asia.summary()

print(Stargazer([North_America, regEurope, Oceania, South_America, Asia, Africa]).render_latex())

#### regression ave les density gradients

results = pd.read_excel("C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/validation_20210504/reg_results.xlsx")
density_gradient = results.loc[:,("city", "reg1_params_X", "reg1_pvalues_X")]
df = df.merge(density_gradient, on = "city")

color_tab = {'North_America':sns.color_palette("Dark2", 6)[0], 'Europe':sns.color_palette("Dark2", 6)[1], 'Asia':sns.color_palette("Dark2", 6)[2], 'Oceania':sns.color_palette("Dark2", 6)[3], 'South_America': sns.color_palette("Dark2", 6)[4], 'Africa': sns.color_palette("Dark2", 6)[5]}
colors = list(df['Continent'].unique())
for i in range(0 , len(colors)):
    data = df.loc[df['Continent'] == colors[i]]
    plt.scatter(data.reg1_params_X, data.income, alpha = 0.5, s = (data.population)/100000, color=data.Continent.map(color_tab), label=colors[i])
lgnd = plt.legend()
for handle in lgnd.legendHandles:
    handle.set_sizes([30.0])
plt.xlabel("Density gradients")
plt.ylabel("Income per capita (USD)")

y = np.array((np.log(df.reg1_params_X)))
y = y[(~np.isnan(np.log(df.reg1_params_X)))]
X = pd.DataFrame(df.iloc[:, 2:9][~np.isnan(np.log(df.reg1_params_X))])
ols = sm.OLS(y, X)
reg1 = ols.fit(cov_type='HC1') #HC3 censé être mieux
reg1.summary()

y = np.array(np.log(df.reg1_params_X))
y = y[(~np.isnan(np.log(df.reg1_params_X))) & (df.reg1_pvalues_X < 0.05)]
X = pd.DataFrame(df.iloc[:, 2:9][(~np.isnan(np.log(df.reg1_params_X))) & (df.reg1_pvalues_X < 0.05)])
ols = sm.OLS(y, X)
reg2 = ols.fit(cov_type='HC1') #HC3 censé être mieux
reg2.summary()



from stargazer.stargazer import Stargazer
print(Stargazer([reg1, reg2]).render_latex())


####

results = pd.read_excel("C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/validation_20210504/reg_results.xlsx")
results.city[(results.reg1_pvalues_X < 0.05) & (results.reg1_params_X < 0)]
results.city[(results.reg2_pvalues_X < 0.05) & (results.reg2_params_X < 0)]
results.city[(results.reg3_pvalues_X < 0.05) & (results.reg3_params_X < 0)]

sum((results.reg1_pvalues_X < 0.05) & (results.reg2_pvalues_X < 0.05) & (results.reg3_pvalues_X < 0.05) & (results.reg1_params_X >0)& (results.reg3_params_X >0)& (results.reg2_params_X <0))
sum((results.reg1_pvalues_X < 0.05) & (results.reg2_pvalues_X < 0.05) & (results.reg3_pvalues_X < 0.05) & (results.reg1_params_X >0)& (results.reg3_params_X >0)& (results.reg2_params_X <0) & (results.reg3_params_X > results.reg1_params_X))

sum((results.reg1_pvalues_X < 0.05) & (results.reg2_pvalues_X < 0.05) & (results.reg3_pvalues_X < 0.05) & (results.reg1_params_X >0)& (results.reg3_params_X >0)& (results.reg2_params_X >0))
sum((results.reg1_pvalues_X < 0.05) & (results.reg2_pvalues_X < 0.05) & (results.reg3_pvalues_X < 0.05) & (results.reg1_params_X >0)& (results.reg3_params_X >0)& (results.reg2_params_X >0) & (results.reg3_params_X > results.reg1_params_X))

second_stage_reg = results.loc[:, ["city", "reg1_r2", "reg1_params_X", "reg1_pvalues_X", "reg2_r2", "reg2_params_X", "reg2_pvalues_X", "reg3_r2","reg3_params_X", "reg3_pvalues_X"]]
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
    
    (country, density, rents_and_size, land_use,
     driving, transit, grille, centre, distance_cbd, conversion_rate) = import_data(list_city, path_data, city)        
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

### R2 reg 1, 2 et 3

y = np.array(second_stage_reg.reg1_r2[second_stage_reg.reg1_r2 > -1])
X = pd.DataFrame(second_stage_reg[second_stage_reg.reg1_r2 > -1].iloc[:, [12, 13, 14, 15, 16, 17, 20 ,21, 22, 23]])
ols = sm.OLS(y, X)
reg1 = ols.fit(cov_type='HC1') #HC3 censé être mieux
reg1.summary()

y = np.array(second_stage_reg.reg2_r2[second_stage_reg.reg2_r2 > -1])
X = pd.DataFrame(second_stage_reg[second_stage_reg.reg2_r2 > -1].iloc[:, [12, 13, 14, 15, 16, 17, 20 ,21, 22, 23]])
ols = sm.OLS(y, X)
reg2 = ols.fit(cov_type='HC1') #HC3 censé être mieux
reg2.summary()

y = np.array(second_stage_reg.reg3_r2[second_stage_reg.reg3_r2 > -1])
X = pd.DataFrame(second_stage_reg[second_stage_reg.reg3_r2 > -1].iloc[:, [12, 13, 14, 15, 16, 17, 20 ,21, 22, 23]])
ols = sm.OLS(y, X)
reg3 = ols.fit(cov_type='HC1') #HC3 censé être mieux
reg3.summary()

### pval < 0 reg 1, 2 et 3

y = np.array(second_stage_reg.reg1_pvalues_X[second_stage_reg.reg1_pvalues_X > -1] < 0.01)
X = pd.DataFrame(second_stage_reg[second_stage_reg.reg1_pvalues_X > -1].iloc[:, [12, 13, 14, 18, 19, 20 ,25, 26]])
ols = sm.OLS(y, X)
reg4 = ols.fit(cov_type='HC1') #HC3 censé être mieux
reg4.summary()

second_stage_reg["dummy_reg2"] = 'non_significant'
second_stage_reg["dummy_reg2"][(second_stage_reg.reg2_pvalues_X < 0.05) & (second_stage_reg.reg2_params_X < 0)] = 'negative'
second_stage_reg["dummy_reg2"][(second_stage_reg.reg2_pvalues_X < 0.05) & (second_stage_reg.reg2_params_X > 0)] = 'apositive'

y = np.array(second_stage_reg.reg2_pvalues_X[second_stage_reg.reg2_pvalues_X > -1] < 0.01)
X = pd.DataFrame(second_stage_reg[second_stage_reg.reg2_pvalues_X > -1].iloc[:, [12, 13, 14, 18, 19, 20 ,23, 24, 25, 26]])
ols = sm.OLS(y, X)
reg5 = ols.fit(cov_type='HC1') #HC3 censé être mieux
reg5.summary()

y = np.array(second_stage_reg["dummy_reg2"])
X = pd.DataFrame(second_stage_reg.iloc[:, [12, 13, 14, 18, 19, 20, 22, 23, 24, 25]])
ols = sm.MNLogit(y, X)
reg5 = ols.fit(cov_type='HC0') #HC3 censé être mieux
reg5.summary() #Avec ou sans Gini ? Plutôt sans.
print(reg5.summary().as_latex())

#Non significant: même variables que pour l'autre régression (inf, sea, population, income)
#Negative: gini + data quality. Sans gini, sea + planification + data quality

#inf housing
#nb of cells ratio


second_stage_reg["dummy_reg3"] = 'non_significant'
second_stage_reg["dummy_reg3"][(second_stage_reg.reg3_pvalues_X < 0.05) & (second_stage_reg.reg3_params_X < 0)] = 'negative'
second_stage_reg["dummy_reg3"][(second_stage_reg.reg3_pvalues_X < 0.05) & (second_stage_reg.reg3_params_X > 0)] = 'apositive'


y = np.array(second_stage_reg.reg3_pvalues_X[(second_stage_reg.dummy_reg3 == "non_significant") |(second_stage_reg.dummy_reg3 == "apositive")] > 0.05)
X = pd.DataFrame(second_stage_reg[(second_stage_reg.dummy_reg3 == "non_significant") |(second_stage_reg.dummy_reg3 == "apositive")].iloc[:, [12, 13, 14, 18, 19, 20, 22, 23, 24, 25]])
ols = sm.OLS(y, X)
reg6 = ols.fit(cov_type='HC1') #HC3 censé être mieux
reg6.summary()

#Résultat très robuste: population et income -, sea et informal housing +

second_stage_reg["log_population"] = np.log(second_stage_reg.population)
second_stage_reg["log_income"] = np.log(second_stage_reg.income)

y = np.array(second_stage_reg["dummy_reg3"])
X = pd.DataFrame(second_stage_reg.iloc[:, [12, 13, 14, 18, 19, 20, 22, 23, 24, 25]])
ols = sm.MNLogit(y, X)
reg6 = ols.fit(cov_type='HC3') #HC3 censé être mieux
reg6.summary()

#Planification ? Rent data quality ?

### param (et param < 0) reg1, 2 et 3

y = np.array(second_stage_reg.reg1_params_X)
X = pd.DataFrame(second_stage_reg.iloc[:, [12, 13, 14, 18, 19, 20 ,25, 26]])
ols = sm.OLS(y, X)
reg7 = ols.fit(cov_type='HC1') #HC3 censé être mieux
reg7.summary()

y = np.array(second_stage_reg.reg2_params_X)
X = pd.DataFrame(second_stage_reg.iloc[:, [12, 13, 14, 18, 19, 20 ,23, 24, 25, 26]])
ols = sm.OLS(y, X)
reg8 = ols.fit(cov_type='HC1') #HC3 censé être mieux
reg8.summary()

y = np.array(second_stage_reg.reg3_params_X)
X = pd.DataFrame(second_stage_reg.iloc[:, [12, 13, 14, 18, 19, 20 ,23, 24, 25, 26]])
ols = sm.OLS(y, X)
reg9 = ols.fit(cov_type='HC1') #HC3 censé être mieux
reg9.summary()



print(Stargazer([reg7, reg8, reg9]).render_latex())


print(Stargazer([reg6]).render_latex())

'''