# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 17:17:53 2022

@author: charl
"""

####### VALIDATION PAPER - URBAN AREA SECTION ####

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

from calibration.calibration import *
from calibration.validation import *
from inputs.data import *
from inputs.land_use import *
from model.model import *
from outputs.outputs import *
from inputs.parameters import *
from inputs.transport import *

###### STEP 1: BUILD DATASET

path_data = "C:/Users/charl/OneDrive/Bureau/City_dataStudy/"
path_folder = "C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Data/"
path_outputs = "C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/"
list_city = list_of_cities_and_databases(path_data,'cityDatabase')#cityDatabase_intermediaire

df = pd.DataFrame(index = list_city.City, columns = ["Urban_area", "Population", "Income", "Agri_rent", "Fuel_cost", "Fuel_conso_cost", "Driving_speed", "Transit_speed", "Max_speed", "Monocentricity", "income_group", "informal_housing", "pop_growth", "Country"])

for city in list(df.index):
    
    country = list_city.Country[list_city.City == city].iloc[0]
    proj = str(int(list_city.GridEPSG[list_city.City == city].iloc[0]))
    df.loc[df.index == city, "Country"] = country
    #urbanized area
    df.loc[df.index == city, "Urban_area"] = np.nansum(pd.read_csv(path_data + 'Data/' + country + '/' + city + 
                           '/Land_Cover/grid_ESACCI_LandCover_2015_' + 
                           str.upper(city) + '_' + proj +'.csv').ESACCI190) / 1000000
 
    #Population  
    density = pd.read_csv(path_data + 'Data/' + country + '/' + city +
                          '/Population_Density/grille_GHSL_density_2015_' +
                          str.upper(city) + '.txt', sep = '\s+|,', engine='python')
    density = density.loc[:,density.columns.str.startswith("density")]
    density = np.array(density).squeeze()
    df.loc[df.index == city, "Population"] = np.nansum(density)
    

    #Incomes
    df.loc[df.index == city, "Income"] = import_gdp_per_capita(path_folder, country, "2018")
    
    #Land prices
    df.loc[df.index == city, "Agri_rent"] = import_agricultural_rent(path_folder, country)

    #Diesel prices
    df.loc[df.index == city, "Fuel_cost"] = import_fuel_price(country, 'gasoline', path_folder) #* import_fuel_conso(country, path_folder) / 100
    df.loc[df.index == city, "Fuel_conso_cost"] = import_fuel_price(country, 'gasoline', path_folder) * import_fuel_conso(country, path_folder) / 100

    
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
    df.loc[df.index == city, "Driving_speed"] = np.nansum(speed_driving * density / np.nansum(density))
    df.loc[df.index == city, "Transit_speed"] = np.nansum(speed_transit * density / np.nansum(density))
    df.loc[df.index == city, "Max_speed"] = np.nansum(max_speed * density / np.nansum(density))

    del transport_source, hour, driving, transit, speed_driving, speed_transit, max_speed, density
    
    #Polycentricity index
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
    df.loc[df.index == city, "Monocentricity"] = polycentricity.loc[polycentricity.city == city, "polycentricity_index"].squeeze()
    
    #income group
    income_group = pd.read_csv(path_folder + 'income_group_wb.csv').loc[:, ["Country", "Income group"]]
    income_group.loc[income_group["Country"] == "Cote d\'Ivoire", "Country"] = "Ivory_Coast"
    income_group.loc[income_group["Country"] == "United States", "Country"] = "USA"
    income_group.loc[income_group["Country"] == "New Zealand", "Country"] = "New_Zealand"
    income_group.loc[income_group["Country"] == "United Kingdom", "Country"] = "UK"
    income_group.loc[income_group["Country"] == "South Africa", "Country"] = "South_Africa"
    income_group.loc[income_group["Country"] == "Russian Federation", "Country"] = "Russia"
    income_group.loc[income_group["Country"] == "Hong Kong SAR, China", "Country"] = "Hong_Kong"
    income_group.loc[income_group["Country"] == "Iran, Islamic Rep.", "Country"] = "Iran"
    income_group.loc[income_group["Country"] == "Czech Republic", "Country"] = "Czech_Republic"
    df.loc[df.index == city, "income_group"] = income_group.loc[income_group.Country == country, "Income group"].squeeze()
    
    #inforaml housing
    informal_housing = import_informal_housing(list_city, path_folder)
    df.loc[df.index == city, "informal_housing"] = informal_housing.loc[informal_housing.City == city, "informal_housing"].squeeze().astype(float)
    
    #pop growth
    df.loc[df.index == city, "pop_growth"] = import_city_scenarios(city, country, path_folder)["2015-2020"]
    if np.isnan(df.loc[df.index == city, "pop_growth"].squeeze()):
        df.loc[df.index == city, "pop_growth"]  = import_country_scenarios(country, path_folder)["2015-2020"]
#df.to_excel("C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/dataset_regressions_v3.xlsx")


###### STEP 2: REGRESSION (log-log, with and withou monocentricity, by continent or GDP)

#df = pd.read_excel("C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/dataset_regressions_v2.xlsx")

df["constant"] = np.ones(192)

city_continent = pd.read_csv(path_folder + "CityDatabases/cityDatabase_221NewCities.csv")
city_continent = city_continent.iloc[:, [0, 2]]
city_continent = city_continent.drop_duplicates(subset = "City")
city_continent = city_continent.sort_values('City')
df = df.merge(city_continent, left_index = True, right_on = "City")

df["log_urbanized_area"] = np.log(df.Urban_area.astype(float))
df["log_population"] = np.log(df.Population.astype(float))
df["log_income"] = np.log(df.Income.astype(float))
df["log_land_prices"] = np.log(df.Agri_rent.astype(float))
df["log_commuting_price"] = np.log(df.Fuel_cost.astype(float))
df["log_commuting_price_conso"] = np.log(df.Fuel_conso_cost.astype(float))
df["log_commuting_time"] = np.log(df.Max_speed.astype(float))
df["log_commuting_time_driving"] = np.log(df.Driving_speed.astype(float))
df["log_commuting_time_transit"] = np.log(df.Transit_speed.astype(float))
df["dummy_transit_speed"] = (df.Driving_speed.astype(float) < df.Transit_speed.astype(float)).astype(float)
df["Monocentricity"] = df.Monocentricity.astype(float)
df["Monocentricity"] = df.Monocentricity.astype(float)
df["Fuel_cost"] = df.Fuel_cost.astype(float)
df["Fuel_conso_cost"] = df.Fuel_conso_cost.astype(float)
df["informal_housing"] = df.informal_housing.astype(float)
df["pop_growth"] = df.pop_growth.astype(float)

fixed_effects = pd.get_dummies(df.Continent)
df["Asia"] = fixed_effects.Asia
df["Africa"] = fixed_effects.Africa
df["North_America"] = fixed_effects.North_America
df["Oceania"] = fixed_effects.Oceania
df["South_America"] = fixed_effects.South_America
df["Europe"]= fixed_effects.Europe

### Main specification

#main specification + monocentricity
y = np.array((df.log_urbanized_area))
X = pd.DataFrame(df.loc[:, ['constant', 'log_population', 'log_income', "log_land_prices", "log_commuting_price", "log_commuting_time", "Monocentricity"]])
ols = sm.OLS(y, X)
spec1 = ols.fit(cov_type='HC1') #HC3 censé être mieux
spec1.summary()

y = np.array((df.log_urbanized_area))
X = pd.DataFrame(df.loc[:, ['constant', 'log_population', 'log_income', "log_land_prices", "log_commuting_price", "log_commuting_time", "Monocentricity", "Africa", "Asia", "North_America", "South_America", "Oceania"]])
ols = sm.OLS(y, X)
spec1 = ols.fit(cov_type='HC1') #HC3 censé être mieux
spec1.summary()

#developed/developin

#def 1
df["developed"] = 0
df["developed"][df.income_group == "High income"] = 1


df.City[(df.developed == 1) & (df.Continent == "Asia")] #Hing Kong and Singapore
df.City[(df.developed == 1) & (df.Continent == "South_America")] #Chile and Uruguay
df.City[(df.developed == 0) & (df.Continent == "Europe")] #Bulgaria, Romania, Russia

#def 1
df["developed"] = 0
df["developed"][df.Continent == "North_America"] = 1
df["developed"][df.Continent == "Europe"] = 1
df["developed"][df.Continent == "Oceania"] = 1

df.loc[:, ["constant", "income_group", "developed"]].groupby(["income_group", "developed"]).sum()

df.City[(df.developed == 0) & (df.income_group == "High income")] #Hong Kong, Singapore, Chile, Uruguay
df.City[(df.developed == 1) & (df.income_group == "Upper middle income")] #Bulgaria, Romania, Russia



y = np.array((df.log_urbanized_area[df.developed == 1]))
X = pd.DataFrame(df.loc[df.developed == 1, ['constant', 'log_population', 'log_income', "log_land_prices", "log_commuting_price", "log_commuting_time", "Monocentricity"]])
ols = sm.OLS(y, X)
spec1 = ols.fit(cov_type='HC1') #HC3 censé être mieux^{***}
spec1.summary()


y = np.array((df.log_urbanized_area[df.developed == 0]))
X = pd.DataFrame(df.loc[df.developed == 0, ['constant', 'log_population', 'log_income', "log_land_prices", "log_commuting_price", "log_commuting_time", "Monocentricity"]])
ols = sm.OLS(y, X)
spec1 = ols.fit(cov_type='HC1') #HC3 censé être mieux
spec1.summary()

y = np.array((df.log_urbanized_area[df.developed == 1]))
X = pd.DataFrame(df.loc[df.developed == 1, ['constant', 'log_population', 'log_income', "log_land_prices", "log_commuting_price", "log_commuting_time", "Monocentricity", "North_America", "Oceania"]])
ols = sm.OLS(y, X)
spec1 = ols.fit(cov_type='HC1') #HC3 censé être mieux
spec1.summary()


y = np.array((df.log_urbanized_area[df.developed == 0]))
X = pd.DataFrame(df.loc[df.developed == 0, ['constant', 'log_population', 'log_income', "log_land_prices", "log_commuting_price", "log_commuting_time", "Monocentricity", "Africa", "Europe", "South_America"]])
ols = sm.OLS(y, X)
spec1 = ols.fit(cov_type='HC1') #HC3 censé être mieux
spec1.summary()

#Chow test developed
y = np.array((df.log_urbanized_area))
X = pd.DataFrame(df.loc[:, ['constant', 'log_population', 'log_income', "log_land_prices", "log_commuting_price", "log_commuting_time", "Monocentricity"]])

y1 = np.array((df.log_urbanized_area[df.developed == 1]))
X1 = pd.DataFrame(df.loc[df.developed == 1, ['constant', 'log_population', 'log_income', "log_land_prices", "log_commuting_price", "log_commuting_time", "Monocentricity"]])

y0 = np.array((df.log_urbanized_area[df.developed == 0]))
X0 = pd.DataFrame(df.loc[df.developed == 0, ['constant', 'log_population', 'log_income', "log_land_prices", "log_commuting_price", "log_commuting_time", "Monocentricity"]])

J = X.shape[1]
k = X1.shape[1]
N1 = X1.shape[0]
N2 = X0.shape[0]

model_dummy = sm.OLS(y,X).fit()
RSSd = model_dummy.ssr
model1 = sm.OLS(y1,X1).fit()
RSS1 = model1.ssr
model0 = sm.OLS(y0,X0).fit()
RSS0 = model0.ssr

chow = ((RSSd-(RSS1+RSS0))/J)/((RSS1+RSS0)/(N1+N2-2*k))
scipy.stats.f.ppf(q=1-0.01, dfn=J, dfd=N1+N2-2*k)

#Chow test continent
y = np.array((df.log_urbanized_area[df.developed == 1]))
X = pd.DataFrame(df.loc[df.developed == 1, ['constant', 'log_population', 'log_income', "log_land_prices", "log_commuting_price", "log_commuting_time", "Monocentricity"]])

y1 = np.array((df.log_urbanized_area[df.Continent == 'North_America']))
X1 = pd.DataFrame(df.loc[df.Continent == 'North_America', ['constant', 'log_population', 'log_income', "log_land_prices", "log_commuting_price", "log_commuting_time", "Monocentricity"]])

y0 = np.array((df.log_urbanized_area[(df.Continent == 'Europe')|(df.Continent == 'Oceania')]))
X0 = pd.DataFrame(df.loc[(df.Continent == 'Europe')|(df.Continent == 'Oceania'), ['constant', 'log_population', 'log_income', "log_land_prices", "log_commuting_price", "log_commuting_time", "Monocentricity"]])

J = X.shape[1]
k = X1.shape[1]
N1 = X1.shape[0]
N2 = X0.shape[0]

model_dummy = sm.OLS(y,X).fit()
RSSd = model_dummy.ssr
model1 = sm.OLS(y1,X1).fit()
RSS1 = model1.ssr
model0 = sm.OLS(y0,X0).fit()
RSS0 = model0.ssr

chow = ((RSSd-(RSS1+RSS0))/J)/((RSS1+RSS0)/(N1+N2-2*k))
scipy.stats.f.ppf(q=1-0.05, dfn=J, dfd=N1+N2-2*k)

#Chow test
y = np.array((df.log_urbanized_area))
X = pd.DataFrame(df.loc[:, ['constant', 'log_population', 'log_income', "log_land_prices", "log_commuting_price", "log_commuting_time", "Monocentricity"]])

yNA = np.array((df.log_urbanized_area[df.Continent == 'North_America']))
XNA = pd.DataFrame(df.loc[df.Continent == 'North_America', ['constant', 'log_population', "log_commuting_time", "Monocentricity"]])

ySA = np.array((df.log_urbanized_area[df.Continent == 'South_America']))
XSA = pd.DataFrame(df.loc[df.Continent == 'South_America', ['constant', 'log_population', 'log_income', "log_land_prices", "log_commuting_price", "log_commuting_time", "Monocentricity"]])

yEU = np.array((df.log_urbanized_area[df.Continent == 'Europe']))
XEU = pd.DataFrame(df.loc[df.Continent == 'Europe', ['constant', 'log_population', 'log_income', "log_land_prices", "log_commuting_price", "log_commuting_time", "Monocentricity"]])

yAS = np.array((df.log_urbanized_area[df.Continent == 'Asia']))
XAS = pd.DataFrame(df.loc[df.Continent == 'Asia', ['constant', 'log_population', 'log_income', "log_land_prices", "log_commuting_price", "log_commuting_time", "Monocentricity"]])

yAF = np.array((df.log_urbanized_area[df.Continent == 'Africa']))
XAF = pd.DataFrame(df.loc[df.Continent == 'Africa', ['constant', 'log_population', 'log_income', "log_land_prices", "log_commuting_price", "log_commuting_time", "Monocentricity"]])

yOC = np.array((df.log_urbanized_area[df.Continent == 'Oceania']))
XOC = pd.DataFrame(df.loc[df.Continent == 'Oceania', ['constant', 'log_population', 'log_income', "log_land_prices", "log_commuting_price", "log_commuting_time", "Monocentricity"]])


J = X.shape[1]
k = XSA.shape[1]
NNA = XNA.shape[0]
NSA = XSA.shape[0]
NEU = XEU.shape[0]
NAS = XAS.shape[0]
NAF = XAF.shape[0]
NOC = XOC.shape[0]

model_dummy = sm.OLS(y,X).fit()
RSSd = model_dummy.ssr
modelNA = sm.OLS(yNA,XNA).fit()
RSSNA = modelNA.ssr
modelSA = sm.OLS(ySA,XSA).fit()
RSSSA = modelSA.ssr
modelEU = sm.OLS(yEU,XEU).fit()
RSSEU = modelEU.ssr
modelAS = sm.OLS(yAS,XAS).fit()
RSSAS = modelAS.ssr
modelAF = sm.OLS(yAF,XAF).fit()
RSSAF = modelAF.ssr
modelOC = sm.OLS(yOC,XOC).fit()
RSSOC = modelOC.ssr


chow = ((RSSd-(RSSNA+RSSSA+RSSEU+RSSAS+RSSAF+RSSOC))/J)/((RSSNA+RSSSA+RSSEU+RSSAS+RSSAF+RSSOC)/(NNA+NSA+NEU+NAS+NAF+NOC-5*k-4))
scipy.stats.f.ppf(q=1-0.05, dfn=J, dfd=NNA+NSA+NEU+NAS+NAF+NOC-5*k-4)



####

((model_dummy.resid)**2).sum()
((model1.resid)**2).sum()
((model0.resid)**2).sum()

numerator = (((model_dummy.resid)**2).sum() - (((model0.resid)**2).sum() + ((model1.resid)**2).sum())) / 7

denominator = (((model0.resid)**2).sum() + ((model1.resid)**2).sum()) / (192 - 2*7)
numerator / denominator

scipy.stats.f.ppf(q=1-0.05, dfn=7, dfd=192-14)
scipy.stats.f.ppf(q=1-0.01, dfn=7, dfd=192-14)

### By Continent - Ne marche pas très bien

y = np.array((df.log_urbanized_area[df.Continent == "Europe"]))
X = pd.DataFrame(df.loc[df.Continent == "Europe", ['constant', 'log_population', 'log_income', "log_land_prices", "log_commuting_price", "log_commuting_time", "Monocentricity"]])
ols = sm.OLS(y, X)
reg1 = ols.fit(cov_type='HC1') #HC3 censé être mieux
reg1.summary()
#Population significant only

y = np.array((df.log_urbanized_area[df.Continent == "North_America"]))
X = pd.DataFrame(df.loc[df.Continent == "North_America", ['constant', 'log_population', "log_commuting_time", "Monocentricity"]])
ols = sm.OLS(y, X)
reg1 = ols.fit(cov_type='HC1') #HC3 censé être mieux
reg1.summary()

y = np.array((df.log_urbanized_area[df.Continent == "Oceania"]))
X = pd.DataFrame(df.loc[df.Continent == "Oceania", ['constant', 'log_population', 'log_income', "log_land_prices", "log_commuting_price", "log_commuting_time", "Monocentricity"]])
ols = sm.OLS(y, X)
reg1 = ols.fit(cov_type='HC1') #HC3 censé être mieux
reg1.summary()

y = np.array((df.log_urbanized_area[df.Continent == "South_America"]))
X = pd.DataFrame(df.loc[df.Continent == "South_America", ['constant', 'log_population', 'log_income', "log_land_prices", "log_commuting_price", "log_commuting_time", "Monocentricity"]])
ols = sm.OLS(y, X)
reg1 = ols.fit(cov_type='HC1') #HC3 censé être mieux
reg1.summary()

y = np.array((df.log_urbanized_area[df.Continent == "Africa"]))
X = pd.DataFrame(df.loc[df.Continent == "Africa", ['constant', 'log_population', 'log_income', "log_land_prices", "log_commuting_price", "log_commuting_time", "Monocentricity"]])
ols = sm.OLS(y, X)
reg1 = ols.fit(cov_type='HC1') #HC3 censé être mieux
reg1.summary()

y = np.array((df.log_urbanized_area[df.Continent == "Asia"]))
X = pd.DataFrame(df.loc[df.Continent == "Asia", ['constant', 'log_population', 'log_income', "log_land_prices", "log_commuting_price", "log_commuting_time", "Monocentricity"]])
ols = sm.OLS(y, X)
reg1 = ols.fit(cov_type='HC1') #HC3 censé être mieux
reg1.summary()

#By GDP

plt.hist(df.Income)

y = np.array((df.log_urbanized_area[df.Income < 20000]))
X = pd.DataFrame(df.loc[df.Income < 20000, ['constant', 'log_population', 'log_income', "log_land_prices", "log_commuting_price", "log_commuting_time", "Monocentricity"]])
ols = sm.OLS(y, X)
reg1 = ols.fit(cov_type='HC1') #HC3 censé être mieux
reg1.summary()

y = np.array((df.log_urbanized_area[df.Income > 20000]))
X = pd.DataFrame(df.loc[df.Income > 20000, ['constant', 'log_population', 'log_income', "log_land_prices", "log_commuting_price", "log_commuting_time", "Monocentricity"]])
ols = sm.OLS(y, X)
reg1 = ols.fit(cov_type='HC1') #HC3 censé être mieux
reg1.summary()

#By GDP: better

y = np.array((df.log_urbanized_area[df.Income < 9002]))
X = pd.DataFrame(df.loc[df.Income < 9002, ['constant', 'log_population', 'log_income', "log_land_prices", "log_commuting_price", "log_commuting_time", "Monocentricity"]])
ols = sm.OLS(y, X)
spec2a = ols.fit(cov_type='HC1') #HC3 censé être mieux
spec2a.summary() #GOOD

y = np.array((df.log_urbanized_area[(df.Income > 9002) &(df.Income <18000)]))
X = pd.DataFrame(df.loc[(df.Income > 9002) &(df.Income <18000), ['constant', 'log_population', 'log_income', "log_land_prices", "log_commuting_price", "log_commuting_time", "Monocentricity"]])
ols = sm.OLS(y, X)
spec2b = ols.fit(cov_type='HC1') #HC3 censé être mieux
spec2b.summary()

y = np.array((df.log_urbanized_area[(df.Income > 18000) &(df.Income <47700)]))
X = pd.DataFrame(df.loc[(df.Income > 18000) &(df.Income <47700), ['constant', 'log_population', 'log_income', "log_land_prices", "log_commuting_price", "log_commuting_time", "Monocentricity"]])
ols = sm.OLS(y, X)
spec2c = ols.fit(cov_type='HC1') #HC3 censé être mieux
spec2c.summary()

y = np.array((df.log_urbanized_area[df.Income > 47700]))
X = pd.DataFrame(df.loc[df.Income > 47700, ['constant', 'log_population', 'log_income', "log_land_prices", "log_commuting_price", "log_commuting_time", "Monocentricity"]])
ols = sm.OLS(y, X)
spec2d = ols.fit(cov_type='HC1') #HC3 censé être mieux
spec2d.summary()

print(Stargazer([spec1, spec2a, spec2b, spec2c, spec2d]))

#By income level

y = np.array((df.log_urbanized_area[df.income_group == "Low income"]))
X = pd.DataFrame(df.loc[df.income_group == "Low income", ['constant', 'log_population', 'log_income', "log_land_prices", "log_commuting_price", "log_commuting_time", "Monocentricity"]])
ols = sm.OLS(y, X)
spec2a = ols.fit(cov_type='HC1') #HC3 censé être mieux
spec2a.summary() #GOOD

y = np.array((df.log_urbanized_area[df.income_group == "Lower middle income"]))
X = pd.DataFrame(df.loc[df.income_group == "Lower middle income", ['constant', 'log_population', 'log_income', "log_land_prices", "log_commuting_price", "log_commuting_time", "Monocentricity"]])
ols = sm.OLS(y, X)
spec2b = ols.fit(cov_type='HC1') #HC3 censé être mieux
spec2b.summary()

y = np.array((df.log_urbanized_area[df.income_group == "Upper middle income"]))
X = pd.DataFrame(df.loc[df.income_group == "Upper middle income", ['constant', 'log_population', 'log_income', "log_land_prices", "log_commuting_price", "log_commuting_time", "Monocentricity"]])
ols = sm.OLS(y, X)
spec2c = ols.fit(cov_type='HC1') #HC3 censé être mieux
spec2c.summary()

y = np.array((df.log_urbanized_area[df.income_group == "High income"]))
X = pd.DataFrame(df.loc[df.income_group == "High income", ['constant', 'log_population', 'log_income', "log_land_prices", "log_commuting_price", "log_commuting_time", "Monocentricity"]])
ols = sm.OLS(y, X)
spec2d = ols.fit(cov_type='HC1') #HC3 censé être mieux
spec2d.summary()

print(Stargazer([spec1, spec2a, spec2b, spec2c, spec2d]))


grouped = df.loc[:, ["constant", "Continent", "income_group"]].groupby(["Continent", "income_group"]).sum()
grouped = df.loc[:, ["constant", "Continent"]].groupby(["Continent"]).sum()
grouped = df.loc[:, ["constant", "income_group", "Continent"]].groupby(["income_group", "Continent"]).sum()

grouped = df.loc[:, ["constant", "Continent", "Country"]].groupby(["Continent", "Country"]).sum()

df.loc[:, ["constant", "Continent", "developed"]].groupby(["Continent", "developed"]).sum()
df.loc[:, ["constant", "income_group", "developed"]].groupby(["income_group", "developed"]).sum()

df.City[(df.developed == 0) & (df.income_group == "High income")]

#By growth rate

plt.hist(df.pop_growth)

y = np.array((df.log_urbanized_area[df.pop_growth < 0.47]))
X = pd.DataFrame(df.loc[df.pop_growth < 0.47, ['constant', 'log_population', 'log_income', "log_land_prices", "log_commuting_price", "log_commuting_time", "Monocentricity"]])
ols = sm.OLS(y, X)
reg1 = ols.fit(cov_type='HC1') #HC3 censé être mieux
reg1.summary()

y = np.array((df.log_urbanized_area[(df.pop_growth > 0.47) & (df.pop_growth < 1.03)]))
X = pd.DataFrame(df.loc[(df.pop_growth > 0.47) & (df.pop_growth < 1.03), ['constant', 'log_population', 'log_income', "log_land_prices", "log_commuting_price", "log_commuting_time", "Monocentricity"]])
ols = sm.OLS(y, X)
reg1 = ols.fit(cov_type='HC1') #HC3 censé être mieux
reg1.summary()

y = np.array((df.log_urbanized_area[(df.pop_growth > 1.03) & (df.pop_growth < 1.50)]))
X = pd.DataFrame(df.loc[(df.pop_growth > 1.03) & (df.pop_growth < 1.50), ['constant', 'log_population', 'log_income', "log_land_prices", "log_commuting_price", "log_commuting_time", "Monocentricity"]])
ols = sm.OLS(y, X)
reg1 = ols.fit(cov_type='HC1') #HC3 censé être mieux
reg1.summary()


y = np.array((df.log_urbanized_area[df.pop_growth > 1.50]))
X = pd.DataFrame(df.loc[df.pop_growth > 1.50, ['constant', 'log_population', 'log_income', "log_land_prices", "log_commuting_price", "log_commuting_time", "Monocentricity"]])
ols = sm.OLS(y, X)
reg1 = ols.fit(cov_type='HC1') #HC3 censé être mieux
reg1.summary()
