# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 15:24:11 2022

@author: charl
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 13:41:44 2022

@author: charl
"""

####### VALIDATION PAPER - RENT GRADIENT SECTION ####

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
from statsmodels.sandbox.regression.gmm import IV2SLS
import seaborn as sns

from calibration.calibration import *
from calibration.validation import *
from inputs.data import *
from inputs.land_use import *
from model.model import *
from outputs.outputs import *
from inputs.parameters import *
from inputs.transport import *

###### STEP 1: RUN THE REGRESSIONS

path_data = "C:/Users/charl/OneDrive/Bureau/City_dataStudy/"
path_folder = "C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Data/"

list_city = list_of_cities_and_databases(path_data,'cityDatabase')#cityDatabase_intermediaire

df = pd.DataFrame(index = list_city.City, columns = ["r_squared_ols", "param_g_ols", "pval_g_ols", "param_h_ols", "pval_h_ols", "r_squared_2SLS", "param_g_2SLS", "pval_g_2SLS", "param_h_2SLS", "pval_h_2SLS", "instr_relevance_coeff", "instr_relevance_pval"])

for city in np.unique(list_city.City):
    
    try:
    
        print("\n*** " + city + " ***\n")
    
        #Import city data
        (country, proj, density, rents_and_size, land_use, land_cover_ESACCI, driving, transit, grille, 
         centre, distance_cbd, conversion_rate) = import_data(list_city, path_data, city, path_folder)        
    
        closed_to_urb = land_cover_ESACCI.ESACCI210 + land_cover_ESACCI.ESACCI220+ land_cover_ESACCI.ESACCI160 + land_cover_ESACCI.ESACCI170 + land_cover_ESACCI.ESACCI180
        opened_to_urb = np.nansum(land_cover_ESACCI.loc[:, ['ESACCI10', 'ESACCI11', 'ESACCI12',
               'ESACCI14', 'ESACCI20', 'ESACCI30', 'ESACCI40', 'ESACCI50', 'ESACCI60',
               'ESACCI61', 'ESACCI62', 'ESACCI70', 'ESACCI71', 'ESACCI72', 'ESACCI80',
               'ESACCI81', 'ESACCI82', 'ESACCI90', 'ESACCI100', 'ESACCI110',
               'ESACCI120', 'ESACCI121', 'ESACCI122', 'ESACCI130', 'ESACCI140',
               'ESACCI150', 'ESACCI151', 'ESACCI152', 'ESACCI153', 'ESACCI190', 'ESACCI200', 'ESACCI201',
               'ESACCI202', 'ESACCI230']], 1)
        
        #Density
        density = density.loc[:,density.columns.str.startswith("density")]
        density = np.array(density).squeeze()
        #coeff_land = (land_use.OpenedToUrb / land_use.TotalArea)
        coeff_land = opened_to_urb / (opened_to_urb + closed_to_urb)
        density = density / coeff_land
        density[coeff_land < 0.1] = np.nan
        #size = rents_and_size.medSize.astype(float)
        
        #Income
        income = import_gdp_per_capita(path_folder, country, "2018", 'PP')
    
        #Import transport data
        fuel_price = import_fuel_price(country, 'gasoline', path_folder) #L/100km
        fuel_consumption = import_fuel_conso(country, path_folder) #dollars/L
        monetary_cost_pt = import_public_transport_cost_data(path_folder, city).squeeze()       
        prix_driving = driving.Duration * income / (3600 * 24) / 365 + driving.Distance * fuel_price * fuel_consumption / 100000
        prix_transit = ((transit.Duration) * income / (3600 * 24) / 365) + monetary_cost_pt
        tous_prix=np.vstack((prix_driving,prix_transit))
        prix_transport=np.amin(tous_prix, axis=0)
        prix_transport[np.isnan(prix_transit)]=prix_driving[np.isnan(prix_transit)]
        prix_transport=prix_transport*2*365
        prix_transport=pd.Series(prix_transport)
        
        #tous_prix=np.vstack((driving.Duration,transit.Duration))
        #prix_transport=np.amin(tous_prix, axis=0)
        #prix_transport[np.isnan(transit.Duration)]=driving.Duration[np.isnan(transit.Duration)]
        #prix_transport=pd.Series(prix_transport)
        
        #prix_transport = distance_cbd
        
        
        data_for_reg = pd.DataFrame({"density":density, "net_income":income-prix_transport, "distance": distance_cbd}) #income-prix_transport
        #data_for_reg = pd.DataFrame({"density":density, "net_income":income-prix_transport, "sizes": size, "distance": distance_cbd}) #income-prix_transport
        #data_for_reg = data_for_reg[~np.isnan(data_for_reg.density) &~np.isnan(data_for_reg.net_income) &~np.isnan(data_for_reg.distance)& (data_for_reg.distance > 0)& (data_for_reg.net_income > 0) & (data_for_reg.density > 0)]
        data_for_reg = data_for_reg[~np.isnan(data_for_reg.density)  &~np.isnan(data_for_reg.distance)& (data_for_reg.distance > 0)&(data_for_reg.density > 0)]
        
        #data_for_reg = data_for_reg.groupby(round(data_for_reg.distance)).mean()
        
        #density / transports
        y= np.array(np.log(data_for_reg.density)).reshape(-1, 1)
        #X = pd.DataFrame({'X': np.log(data_for_reg.net_income),'sizes': np.log(data_for_reg.sizes), 'intercept': np.ones(len(y)).squeeze()})
        X = pd.DataFrame({'X': (data_for_reg.distance),'intercept': np.ones(len(y)).squeeze()})
        ols = sm.OLS(y, X)
        reg_ols= ols.fit()
        reg_ols.summary()
        
        df.loc[df.index == city, "r_squared_ols"] = reg_ols.rsquared
        df.loc[df.index == city, "param_g_ols"] = reg_ols.params['intercept']
        df.loc[df.index == city, "pval_g_ols"] = reg_ols.pvalues['intercept']
        df.loc[df.index == city, "param_h_ols"] = reg_ols.params['X']
        df.loc[df.index == city, "pval_h_ols"] = reg_ols.pvalues['X']
        """"
        #2SLS propre
        y= np.array(np.log(data_for_reg.net_income)).reshape(-1, 1)
        X = pd.DataFrame({'X': np.log(data_for_reg.distance), 'intercept': np.ones(len(y)).squeeze()})
        ols = sm.OLS(y, X)
        instr_relevance = ols.fit()
        instr_relevance.summary()
        
        df.loc[df.index == city, "instr_relevance_coeff"] = instr_relevance.params['X']
        df.loc[df.index == city, "instr_relevance_pval"] = instr_relevance.pvalues['X']
        
        data_for_reg['predicted_net_income'] = instr_relevance.predict()
        y= np.array(np.log(data_for_reg.density)).reshape(-1, 1)
        X = pd.DataFrame({'X': (data_for_reg.predicted_net_income), 'intercept': np.ones(len(y)).squeeze()})
        ols = sm.OLS(y, X)
        sec_stage = ols.fit()
        sec_stage.summary()
        
        #2SLS
        endog = np.array(np.log(data_for_reg.density)).reshape(-1, 1)
        exog = pd.DataFrame({'X': np.log(data_for_reg.net_income), 'intercept': np.ones(len(endog)).squeeze()})
        instr = pd.DataFrame({'X': np.log(data_for_reg.distance), 'intercept': np.ones(len(endog)).squeeze()})
        reg_2SLS = IV2SLS(endog, exog, instrument = instr).fit()
        reg_2SLS.summary()
    
        df.loc[df.index == city, "r_squared_2SLS"] = reg_2SLS.rsquared
        df.loc[df.index == city, "param_g_2SLS"] = reg_2SLS.params['intercept']
        df.loc[df.index == city, "pval_g_2SLS"] = reg_2SLS.pvalues['intercept']
        df.loc[df.index == city, "param_h_2SLS"] = reg_2SLS.params['X']
        df.loc[df.index == city, "pval_h_2SLS"] = reg_2SLS.pvalues['X']
        """
    
    except:
        
        pass
    
#### ANALYSIS

df = df.astype(float)
df = df.loc[(~np.isnan(df.r_squared_ols)) & (~np.isinf(df.r_squared_ols))]

#hist1: R2 (+mean and range)
#plt.hist(df.r_squared_ols)
#plt.hist(df.param_h_2SLS)
sns.kdeplot(data=df, x="r_squared_ols", label = 'OLS')
sns.kdeplot(data=df, x="r_squared_2SLS", label = '2SLS')
plt.legend()
#print(df.r_squared_ols.describe())
#print(df.r_squared_2SLS.describe()) #r2 négatif normal en 2SLS

#hist2: param (+mean and range)
#plt.hist(df.param_h_ols)
#plt.hist(df.param_h_2SLS)
sns.kdeplot(data=df, x="param_h_ols", label = 'OLS')
sns.kdeplot(data=df, x="param_h_2SLS", label = '2SLS')
plt.legend()

bins = np.linspace(-0.32, 0, 40)
#bins = np.linspace(-3.5, -0.5, 40)
plt.rcParams['figure.dpi'] = 360
sns.set(style="whitegrid")
fig, ax = plt.subplots(figsize=(6,4))
plt.hist(df.param_h_ols, bins, label = "OLS", alpha=0.7, color = "#66c2a5")
#plt.hist(df.param_h_2SLS, bins, label = "2SLS", alpha=0.7, color = "#fc8d62")
plt.ylabel("", size=1)
plt.xlabel("", size=1)
#plt.legend(loc='upper right')
ax.tick_params(axis = 'both', labelsize=14, color='#4f4e4e')


print(df.param_h_ols.describe())
print(df.param_h_2SLS.describe())

print(sum((df.instr_relevance_pval < 0.001)&(df.instr_relevance_coeff < 0)))

#Nb of significant/insignificant
print("OLS")
print(np.nansum((df.param_h_ols > 0) & (df.pval_h_ols < 0.1)))
print(np.nansum((df.param_h_ols < 0) & (df.pval_h_ols < 0.1)))
print(np.nansum((df.pval_h_ols > 0.1)))
print("2SLS")
print(np.nansum((df.param_h_2SLS > 0) & (df.pval_h_2SLS < 0.1)))
print(np.nansum((df.param_h_2SLS < 0) & (df.pval_h_2SLS < 0.1)))
print(np.nansum((df.pval_h_2SLS > 0.1)))

df.to_excel("C:/Users/charl/OneDrive/Bureau/Revision RSUE/Revision round 1/results_housing.xlsx")
#### SECOND-STEP REGRESSION

#Build dataset on city characteristics

second_step = pd.DataFrame(index = list_city.City, columns = ["City", "Country", "Population", "Income", "Agri_rent", "Fuel_cost", "Max_speed", "Monocentricity", 'income_group', 'pop_growth']) #, "spatial_data_cover", "market_data_cover"])

for city in list(df.index):
    
    country = list_city.Country[list_city.City == city].iloc[0]
    proj = str(int(list_city.GridEPSG[list_city.City == city].iloc[0]))
    second_step.loc[second_step.index == city, "Country"] = country
    second_step.loc[second_step.index == city, "City"] = city
    
    #Population  
    density = pd.read_csv(path_data + 'Data/' + country + '/' + city +
                          '/Population_Density/grille_GHSL_density_2015_' +
                          str.upper(city) + '.txt', sep = '\s+|,', engine='python')
    density = density.loc[:,density.columns.str.startswith("density")]
    density = np.array(density).squeeze()
    second_step.loc[second_step.index == city, "Population"] = np.nansum(density)
    

    #Incomes
    second_step.loc[second_step.index == city, "Income"] = import_gdp_per_capita(path_folder, country, "2018")
    
    #Land prices
    second_step.loc[second_step.index == city, "Agri_rent"] = import_agricultural_rent(path_folder, country)

    #Diesel prices
    second_step.loc[second_step.index == city, "Fuel_cost"] = import_fuel_price(country, 'gasoline', path_folder) #* import_fuel_conso(country, path_folder) / 100
    
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
    second_step.loc[second_step.index == city, "Max_speed"] = np.nansum(max_speed * density / np.nansum(density))

    #second_step.loc[second_step.index == city, "spatial_data_cover"] = sum(~np.isnan(rents_and_size.medRent)) / sum((pd.to_numeric(density)) > 0) 
    #second_step.loc[second_step.index == city, "market_data_cover"] = np.nansum(density) / np.nansum(rents_and_size.dataCount)
    
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
    second_step.loc[second_step.index == city, "Monocentricity"] = polycentricity.loc[polycentricity.city == city, "polycentricity_index"].squeeze()
    
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
    second_step.loc[second_step.index == city, "income_group"] = income_group.loc[income_group.Country == country, "Income group"].squeeze()
    
    #pop growth
    second_step.loc[second_step.index == city, "pop_growth"] = import_city_scenarios(city, country, path_folder)["2015-2020"]
    if np.isnan(second_step.loc[second_step.index == city, "pop_growth"].squeeze()):
        second_step.loc[second_step.index == city, "pop_growth"]  = import_country_scenarios(country, path_folder)["2015-2020"]


#Gini
gini_index = import_gini_index(path_folder)
second_step = second_step.merge(gini_index, on = "Country")
second_step.gini[second_step.Country == "New_Zealand"] = 36.9
second_step.gini[second_step.Country == "Hong_Kong"] = 53.9
second_step.gini[second_step.Country == "Singapore"] = 45.9

#Informal housing
informal_housing = import_informal_housing(list_city, path_folder)
second_step = second_step.merge(informal_housing.loc[:, ["City", "informal_housing"]], left_on = "City", right_on = "City")

#Regulation and planning and amenities
sea_regulation = pd.read_excel(path_folder + "sea_planification.xlsx", header = 0)
#sea_regulation["planification"] = np.fmax(np.fmax(2 * sea_regulation.strong_regulation, sea_regulation.low_regulation), 0.5*sea_regulation.wiki)
#sea_regulation["planification"][np.isnan(sea_regulation["planification"])] = 0
#sea_regulation["planification"][np.isnan(sea_regulation["planification"])] = sea_regulation.wiki
#sea_regulation["planification"][np.isnan(sea_regulation["planification"])] = sea_regulation.wiki
#sea_regulation.strong_regulation[np.isnan(sea_regulation.strong_regulation)] = 0
sea_regulation["planification"] = sea_regulation.wiki
sea_regulation = sea_regulation.loc[:, ["city", "planification", "sea"]]
#second_step = second_step.drop(columns= ["planification", "sea"])



second_step = second_step.merge(sea_regulation, left_on = "City", right_on = "city")

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
second_step = second_step.merge(informal_jobs, on = "Country", how = "left")
second_step.informal_jobs[np.isnan(second_step.informal_jobs)] = 0

#Regression

second_step = second_step.merge(df, left_on = 'City', right_index = True) 

second_step["constant"] = np.ones(192)

second_step["log_population"] = np.log(second_step.Population.astype(float))
second_step["log_income"] = np.log(second_step.Income.astype(float))
second_step["log_land_prices"] = np.log(second_step.Agri_rent.astype(float))
second_step["log_commuting_price"] = np.log(second_step.Fuel_cost.astype(float))
second_step["log_commuting_time"] = np.log(second_step.Max_speed.astype(float))
second_step["Monocentricity"] = second_step.Monocentricity.astype(float)
second_step["pop_growth"] = second_step.pop_growth.astype(float)


fixed_effects_income = pd.get_dummies(second_step.income_group)
second_step["High income"] = fixed_effects_income["High income"]
second_step["Lower middle income"] = fixed_effects_income["Lower middle income"]

city_continent = pd.read_csv(path_folder + "CityDatabases/cityDatabase_221NewCities.csv")
city_continent = city_continent.iloc[:, [0, 2]]
city_continent = city_continent.drop_duplicates(subset = "City")
city_continent = city_continent.sort_values('City')
second_step = second_step.merge(city_continent, left_on = "City", right_on = "City")
fixed_effects = pd.get_dummies(second_step.Continent)
second_step["Asia"] = fixed_effects.Asia
second_step["Africa"] = fixed_effects.Africa
second_step["North_America"] = fixed_effects.North_America
second_step["Oceania"] = fixed_effects.Oceania
second_step["South_America"] = fixed_effects.South_America
second_step["Europe"]= fixed_effects.Europe

second_step["Q4_income"] = (second_step.Income < 9002).astype(float)
second_step["Q3_income"] = ((second_step.Income < 18000) & (second_step.Income > 9002)).astype(float)
second_step["Q2_income"] = ((second_step.Income < 47700) & (second_step.Income > 18000)).astype(float)
second_step["Q1_income"] = (second_step.Income > 47700).astype(float)

second_step["Q4_pop"] = (second_step.Population < 1260000).astype(float)
second_step["Q3_pop"] = ((second_step.Population < 2400000) & (second_step.Population > 1260000)).astype(float)
second_step["Q2_pop"] = ((second_step.Population < 5450000) & (second_step.Population > 2400000)).astype(float)
second_step["Q1_pop"] = (second_step.Population > 5450000).astype(float)

plt.scatter(second_step.log_population, second_step.param_h_2SLS)
plt.scatter(second_step.log_income, second_step.param_h_2SLS)
plt.scatter(second_step.Monocentricity, second_step.param_h_2SLS)
plt.scatter(second_step.gini, second_step.param_h_2SLS)
plt.scatter(second_step.informal_housing, second_step.param_h_2SLS)
plt.scatter(second_step.informal_jobs, second_step.param_h_2SLS)
plt.scatter(second_step.planification, second_step.param_h_2SLS)
plt.scatter(second_step.Continent, second_step.param_h_2SLS)

### Based on literature

#we except a lower relationship with high population and a higher relationship with monocentricity
y = np.array((second_step.param_h_2SLS))
X = pd.DataFrame(second_step.loc[:, ['log_population', "Monocentricity", 'constant']])
#X = pd.DataFrame(second_step.loc[:, ['log_population', 'log_income', "Monocentricity", 'constant']])
ols = sm.OLS(y, X)
spec1 = ols.fit(cov_type='HC1') #HC3 censé être mieux
spec1.summary()

#+ amenities
y = np.array((second_step.param_h_2SLS))
X = pd.DataFrame(second_step.loc[:, ['constant', 'log_population',"Monocentricity","gini", 'informal_housing', 'sea', "planification"]])
#X = pd.DataFrame(second_step.loc[:, ['constant', 'log_population', 'log_income',"Monocentricity","gini", 'informal_housing', 'sea', "planification"]])
ols = sm.OLS(y, X)
spec1 = ols.fit(cov_type='HC1') #HC3 censé être mieux
spec1.summary()

y = np.array((second_step.param_h_2SLS))
X = pd.DataFrame(second_step.loc[:, ['constant','log_population', "Monocentricity", "log_commuting_price", "log_commuting_time", "gini", 'informal_housing','sea', "planification"]])
ols = sm.OLS(y, X)
spec1 = ols.fit(cov_type='HC1') #HC3 censé être mieux
spec1.summary()
#descriptive
y = np.array((second_step.param_h_ols))
#X = pd.DataFrame(second_step.loc[:, ['constant','log_population', "Monocentricity", "gini", 'informal_housing','sea', "planification", "Asia", "Africa", "Oceania", "North_America", "South_America"]])
X = pd.DataFrame(second_step.loc[:, ['constant','log_population', 'log_income', "Monocentricity", "gini", 'informal_housing','sea', "planification", "Asia", "Africa", "Oceania", "North_America", "South_America"]])
ols = sm.OLS(y, X)
spec1 = ols.fit(cov_type='HC1') #HC3 censé être mieux
spec1.summary()

#Big question: role of income???





y = np.array((second_step.param_h_2SLS)) #2SLS, rsquared
X = pd.DataFrame(second_step.loc[:, ['constant', 'log_population', 'log_income', "log_commuting_price", "log_commuting_time", "Monocentricity", "gini", 'informal_housing', 'planification', 'sea', 'informal_jobs']])
ols = sm.OLS(y, X)
spec1 = ols.fit(cov_type='HC1') #HC3 censé être mieux
spec1.summary()


### Descriptif

y = np.array((second_step.param_h_2SLS)) #2SLS, rsquared
X = pd.DataFrame(second_step.loc[:, ['constant', "Asia", "Africa", "Oceania", "Europe", "South_America"]])
ols = sm.OLS(y, X)
spec1 = ols.fit(cov_type='HC1') #HC3 censé être mieux
spec1.summary()

y = np.array((second_step.param_h_2SLS)) #2SLS, rsquared
X = pd.DataFrame(second_step.loc[:, ['constant', "Q1_income", "Q2_income", "Q4_income"]])
ols = sm.OLS(y, X)
spec1 = ols.fit(cov_type='HC1') #HC3 censé être mieux
spec1.summary()

y = np.array((second_step.param_h_2SLS)) #2SLS, rsquared
X = pd.DataFrame(second_step.loc[:, ['constant', "Q1_pop", "Q2_pop", "Q4_pop"]])
ols = sm.OLS(y, X)
spec1 = ols.fit(cov_type='HC1') #HC3 censé être mieux
spec1.summary()

#Descriptf et explicatif

y = np.array((second_step.param_h_2SLS)) #2SLS, rsquared
X = pd.DataFrame(second_step.loc[:, ['constant', "Asia", "Africa", "Oceania", "North_America", "South_America", 'log_population', 'log_income']])
ols = sm.OLS(y, X)
spec1 = ols.fit(cov_type='HC1') #HC3 censé être mieux
spec1.summary()

y = np.array((second_step.r_squared_ols)) #2SLS, rsquared
X = pd.DataFrame(second_step.loc[:, ['constant', "Monocentricity", "gini", "informal_housing", "planification", "sea", 'log_population', 'log_income']])
ols = sm.OLS(y, X)
spec1 = ols.fit(cov_type='HC1') #HC3 censé être mieux
spec1.summary()

y = np.array((second_step.param_h_2SLS)) #2SLS, rsquared
X = pd.DataFrame(second_step.loc[:, ['constant', "Monocentricity", "gini", "informal_housing", "Q1_pop", "Q2_pop", "Q4_pop", "Q1_income", "Q2_income", "Q4_income"]])
ols = sm.OLS(y, X)
spec1 = ols.fit(cov_type='HC1') #HC3 censé être mieux
spec1.summary()

y = np.array((second_step.param_h_2SLS)) #2SLS, rsquared
X = pd.DataFrame(second_step.loc[:, ['constant', "Asia", "Africa", "Oceania", "North_America", "South_America", "Monocentricity", "gini", "informal_housing", 'log_population', 'log_income']])
ols = sm.OLS(y, X)
spec1 = ols.fit(cov_type='HC1') #HC3 censé être mieux
spec1.summary()


