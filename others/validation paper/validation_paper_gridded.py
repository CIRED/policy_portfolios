# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 15:23:46 2021

@author: charl
"""


### IMPORT PACKAGES

import pandas as pd
import numpy as np
import copy
import pickle
import scipy as sc
from scipy import optimize
import matplotlib.pyplot as plt
import os
import math
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

### IMPORT FUNCTIONS


from calibration.calibration import *
from calibration.validation import *
from inputs.data import *
from inputs.land_use import *
from model.model import *
from outputs.outputs import *
from inputs.parameters import *
from inputs.transport import *

### PATHS

path_data = "C:/Users/charl/OneDrive/Bureau/City_dataStudy/"
path_folder = "C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Data/"

### PARAMETERS

list_city = list_of_cities_and_databases(path_data,'cityDatabase')#cityDatabase_intermediaire

INTEREST_RATE = 0.05
HOUSEHOLD_SIZE = 1
TIME_LAG = 2
DEPRECIATION_TIME = 100
DURATION = 20
CARBON_TAX = 0
COEFF_URB = 0.62
FIXED_COST_CAR = 0
WALKING_SPEED = 5

### OUTCOMES

reg1_r2 = {}
reg1_params_X = {}
reg1_params_intercept = {}
reg1_pvalues_X = {}
reg1_pvalues_intercept = {}
reg1_bse_X= {}
reg1_bse_intercept= {}
reg1_int_X_low= {}
reg1_int_intercept_low= {}
reg1_int_X_high= {}
reg1_int_intercept_high= {}

reg1b_r2 = {}
reg1b_net_income = {}
reg1b_rent = {}
reg1b_pvalues_net_income = {}
reg1b_pvalues_rent = {}
reg1b_bse_X= {}
reg1b_bse_intercept= {}
reg1b_int_X_low= {}
reg1b_int_intercept_low= {}
reg1b_int_X_high= {}
reg1b_int_intercept_high= {}

reg2_r2 = {}
reg2_params_X = {}
reg2_params_b = {}
reg2_params_intercept = {}
reg2_ctrl = {}
reg2_pvalues_X = {}
reg2_pvalues_intercept = {}
reg2_pvalues_ctrl = {}
reg2_bse_X= {}
reg2_bse_intercept= {}
reg2_int_X_low= {}
reg2_int_intercept_low= {}
reg2_int_X_high= {}
reg2_int_intercept_high= {}

reg3_r2 = {}
reg3_params_X = {}
reg3_params_beta = {}
reg3_params_intercept = {}
reg3_pvalues_X = {}
reg3_pvalues_intercept = {}
reg3_bse_X = {}
reg3_bse_intercept= {}
reg3_int_X_low= {}
reg3_int_intercept_low= {}
reg3_int_X_high= {}
reg3_int_intercept_high= {}

### RUN REGRESSIONS

for city in np.unique(list_city.City):
    try:
        print("\n*** " + city + " ***\n")

        ### IMPORT DATA
    
        print("\n** Import data **\n")

        #Import city data
        (country, proj, density, rents_and_size, land_use, land_cover_ESACCI, driving, transit, grille, 
         centre, distance_cbd, conversion_rate) = import_data(list_city, path_data, city, path_folder)        
    
        density = density.loc[:,density.columns.str.startswith("density")],
        density = np.array(density).squeeze()
        rent = (rents_and_size.avgRent / conversion_rate) * 12
        #rent[rents_and_size.dataCount < 4] = np.nan
        size = rents_and_size.medSize
        coeff_land = (land_use.OpenedToUrb / land_use.TotalArea) * COEFF_URB
        population = np.nansum(density)
        region = pd.read_excel(path_folder + "city_country_region.xlsx").region[pd.read_excel(path_folder + "city_country_region.xlsx").city == city].squeeze()
        income = pd.read_excel(path_folder + "income.xlsx").income[pd.read_excel(path_folder + "income.xlsx").city == city].squeeze()
        agricultural_rent = import_agricultural_rent(path_folder, country)
    
        #Import transport data
        fuel_price = import_fuel_price(country, 'gasoline', path_folder) #L/100km
        fuel_consumption = import_fuel_conso(country, path_folder) #dollars/L
        #CO2_emissions_car = import_emissions_per_km(country, path_folder) #gCO2/pkm
        monetary_cost_pt = import_public_transport_cost_data(path_folder, city).squeeze()
        
        #CO2_EMISSIONS_TRANSIT = 15 #gCO2/pkm
            
        #Import scenarios
        imaclim = pd.read_excel(path_folder + "Charlotte_ResultsIncomePriceAutoEmissionsAuto_V1.xlsx", sheet_name = 'Extraction_Baseline')
        population_growth = import_city_scenarios(city, country, path_folder)
        if isinstance(population_growth["2015-2020"], pd.Series) == True:
            population_growth = import_country_scenarios(country, path_folder)
        income_growth = imaclim[imaclim.Region == region][imaclim.Variable == "Index_income"].squeeze()
        emissions_evolution = imaclim[imaclim.Region == region][imaclim.Variable == "Index_emissions_auto"].squeeze()
        transport_cost_evolution = imaclim[imaclim.Region == region][imaclim.Variable == "Index_prix_auto"].squeeze()
    
        print("\n** Transport modelling **\n")
        
        prix_driving = driving.Duration * income / (3600 * 24) / 365 + driving.Distance * fuel_price * fuel_consumption / 100000 + FIXED_COST_CAR
        prix_transit = ((transit.Duration) * income / (3600 * 24) / 365) + monetary_cost_pt
        prix_walking = ((distance_cbd) / WALKING_SPEED) * (income / (24 * 365))
        prix_walking[distance_cbd > 8] = math.inf
        tous_prix=np.vstack((prix_driving,prix_transit, prix_walking))#les prix concaténés
        prix_transport=np.amin(tous_prix, axis=0)
        prix_transport[np.isnan(prix_transit)]=np.amin(np.vstack((prix_driving[np.isnan(prix_transit)], prix_walking[np.isnan(prix_transit)])), axis = 0)
        prix_transport=prix_transport*2*365 # on l'exprime par rapport au revenu
        prix_transport=pd.Series(prix_transport)
        mode_choice=np.argmin(tous_prix, axis=0)
        mode_choice[np.isnan(prix_transit) & (prix_driving < prix_walking)] = 0
        mode_choice[np.isnan(prix_transit) & (prix_driving > prix_walking)] = 2
        
        country = list_city.Country[list_city.City == city].iloc[0]
        proj = str(int(list_city.GridEPSG[list_city.City == city].iloc[0]))

        #urbanized area
        land_use = pd.read_csv(path_data + 'Data/' + country + '/' + city + 
                               '/Land_Cover/grid_ESACCI_LandCover_2015_' + 
                               str.upper(city) + '_' + proj +'.csv')
    
        #array_duration =np.vstack((driving.Duration,transit.Duration))#les prix concaténés
        #array_duration=np.amin(array_duration, axis=0)
        #array_duration[np.isnan(array_duration)]=driving.Duration[np.isnan(array_duration)]
        
        #int_dist = distance_cbd.astype(int)
        
        df = pd.DataFrame({"density":density, "rent":rent, "nat_cons": coeff_land, "sizes": size, "net_income":income-prix_transport}) #income-prix_transport
        #df = pd.DataFrame({"density":density, "rent":rent, "nat_cons": coeff_land, "sizes": size, "net_income":array_duration})
        
        #df["int_dist"] = int_dist
        #df['dummy'] = 1
        
        #density = df.groupby('int_dist')['density'].sum() / df.groupby('int_dist')['dummy'].sum()
        #nat_cons = df.groupby('int_dist')['nat_cons'].sum() / df.groupby('int_dist')['dummy'].sum()
        
        #df['rent_weighted'] = df.rent * df.density
        #rent = df.groupby('int_dist')['rent_weighted'].sum() / df.groupby('int_dist')['density'].sum()
        
        #df['size_weighted'] = df.sizes * df.density
        #sizes = df.groupby('int_dist')['size_weighted'].sum() / df.groupby('int_dist')['density'].sum()
        
        #df['net_income_weighted'] = df.net_income * df.density
        #net_income = df.groupby('int_dist')['net_income_weighted'].sum() / df.groupby('int_dist')['density'].sum()
        
        #df = np.nan
        #df = pd.DataFrame({"density":density, "rent":rent, "nat_cons": coeff_land, "sizes": size, "net_income":income - prix_transport})
        
        #df = pd.DataFrame({"density":density, "rent":rent, "nat_cons": nat_cons, "sizes": sizes, "net_income":net_income})
        
        #distance_cbd = np.arange(max(int_dist) + 1)
        #distance_cbd = distance_cbd[~np.isnan(df.density) &~np.isnan(df.nat_cons)&~np.isnan(df.rent) &~np.isnan(df.sizes) & ~np.isnan(df.net_income) & (df.density != 0) & (df.net_income > 0) & (df.rent != 0)& (df.nat_cons != 0)& (df.sizes != 0)]
        
        
        df = df[~np.isnan(df.density) &~np.isnan(df.nat_cons)&~np.isnan(df.rent) &~np.isnan(df.sizes) & ~np.isnan(df.net_income) & (df.density != 0) & (df.net_income > 0) & (df.rent != 0)& (df.nat_cons != 0)& (df.sizes != 0)]
        
        #if city != "Sfax":
            #    corr_income_housing[city] = scipy.stats.pearsonr(df.net_income, df.housing)[0]
            #    corr_income_density[city] = scipy.stats.pearsonr(df.net_income, df.density)[0]
            #    corr_income_rent[city] = scipy.stats.pearsonr(df.net_income, df.rent)[0]
            #    corr_rent_housing[city] = scipy.stats.pearsonr(df.rent, df.housing)[0]
            #    corr_rent_density[city] = scipy.stats.pearsonr(df.rent, df.density)[0]
            
            #    corr_income_housing_pval[city] = scipy.stats.pearsonr(df.net_income, df.housing)[1]
            #    corr_income_density_pval[city] = scipy.stats.pearsonr(df.net_income, df.density)[1]
            #    corr_income_rent_pval[city] = scipy.stats.pearsonr(df.net_income, df.rent)[1]
            #    corr_rent_housing_pval[city] = scipy.stats.pearsonr(df.rent, df.housing)[1]
            #    corr_rent_density_pval[city] = scipy.stats.pearsonr(df.rent, df.density)[1]


        #Density gradients Berteaud
        y= np.array(np.log(df.density)).reshape(-1, 1)
        X = pd.DataFrame({'X': np.log(df.net_income), 'intercept': np.ones(len(y)).squeeze(), 'ctrl1' : np.log(df.nat_cons), "ctrl2":np.log(df.sizes)})
        ols = sm.OLS(y, X)
        reg1 = ols.fit()
        #plt.scatter(distance_cbd, np.exp(y))
        #plt.scatter(distance_cbd, np.exp(reg1.predict(X)))
        reg1_r2[city] = reg1.rsquared
        reg1_params_X[city] = reg1.params['X']
        reg1_params_intercept[city] = np.exp(reg1.params['intercept'])
        reg1_pvalues_X[city] = reg1.pvalues['X']
        reg1_pvalues_intercept[city] = reg1.pvalues['intercept']
        reg1_bse_X[city] = reg1.bse['X']
        reg1_bse_intercept[city] = reg1.bse['intercept']
        reg1_int_X_low[city] = reg1.conf_int(alpha=0.05, cols=None)[0][0]
        reg1_int_intercept_low[city] = reg1.conf_int(alpha=0.05, cols=None)[0][1]
        reg1_int_X_high[city] = reg1.conf_int(alpha=0.05, cols=None)[1][0]
        reg1_int_intercept_high[city] = reg1.conf_int(alpha=0.05, cols=None)[1][1]
        
        y= np.array(np.log(df.sizes)).reshape(-1, 1)
        X = pd.DataFrame({'X': np.log(df.net_income), 'intercept': np.ones(len(y)).squeeze(), 'ctrl1' : np.log(df.rent)})
        ols = sm.OLS(y, X)
        reg1 = ols.fit()
        #plt.scatter(distance_cbd, np.exp(y))
        #plt.scatter(distance_cbd, np.exp(reg1.predict(X)))
        reg1b_r2[city] = reg1.rsquared
        reg1b_net_income[city] = reg1.params['X']
        reg1b_rent[city] = reg1.params['ctrl1']
        reg1b_pvalues_net_income[city] = reg1.pvalues['X']
        reg1b_pvalues_rent[city] = reg1.pvalues['ctrl1']
        reg1b_bse_X[city] = reg1.bse['X']
        reg1b_bse_intercept[city] = reg1.bse['ctrl1']
        reg1b_int_X_low[city] = reg1.conf_int(alpha=0.05, cols=None)[0][0]
        reg1b_int_intercept_low[city] = reg1.conf_int(alpha=0.05, cols=None)[0][1]
        reg1b_int_X_high[city] = reg1.conf_int(alpha=0.05, cols=None)[1][0]
        reg1b_int_intercept_high[city] = reg1.conf_int(alpha=0.05, cols=None)[1][1]
        
        #housing/rents
        X = pd.DataFrame({'X': np.log(df.rent),'ctrl': np.log(df.nat_cons),'ctrl2': np.log(df.sizes), 'intercept': np.ones(sum(~np.isnan(df.density) & ~np.isnan(df.rent) & (df.rent != 0) & ~np.isnan(df.nat_cons)& (df.nat_cons != 0) & (df.density != 0))).squeeze()})
        y= np.array(np.log(df.density)).reshape(-1, 1)        
        ols = sm.OLS(y, X)
        reg2 = ols.fit()
        reg2_r2[city] = reg2.rsquared
        reg2_params_X[city] = reg2.params['X']
        reg2_params_b[city] = reg2.params['X'] / (1 + reg2.params['X'])
        reg2_params_intercept[city] = reg2.params['intercept']
        reg2_ctrl[city] = reg2.params['ctrl']
        reg2_pvalues_X[city] = reg2.pvalues['X']
        reg2_pvalues_intercept[city] = reg2.pvalues['intercept']
        reg2_pvalues_ctrl[city] = reg2.pvalues['ctrl']
        reg2_bse_X[city] = reg2.bse['X']
        reg2_bse_intercept[city] = reg2.bse['intercept']
        reg2_int_X_low[city] = reg2.conf_int(alpha=0.05, cols=None)[0][0]
        reg2_int_intercept_low[city] = reg2.conf_int(alpha=0.05, cols=None)[0][1]
        reg2_int_X_high[city] = reg2.conf_int(alpha=0.05, cols=None)[1][0]
        reg2_int_intercept_high[city] = reg2.conf_int(alpha=0.05, cols=None)[1][1]
        
        #rents / transports
        y= np.array(np.log(df.rent)).reshape(-1, 1)
        X = pd.DataFrame({'X': np.log(df.net_income), 'intercept': np.ones(len(y)).squeeze()})
        ols = sm.OLS(y, X)
        reg3= ols.fit()
        reg3_r2[city] = reg3.rsquared
        reg3_params_X[city] = reg3.params['X']
        reg3_params_beta[city] = 1 / reg3.params['X']
        reg3_params_intercept[city] = reg3.params['intercept']
        reg3_pvalues_X[city] = reg3.pvalues['X']
        reg3_pvalues_intercept[city] = reg3.pvalues['intercept']
        reg3_bse_X[city] = reg3.bse['X']
        reg3_bse_intercept[city] = reg3.bse['intercept']
        reg3_int_X_low[city] = reg3.conf_int(alpha=0.05, cols=None)[0][0]
        reg3_int_intercept_low[city] = reg3.conf_int(alpha=0.05, cols=None)[0][1]
        reg3_int_X_high[city] = reg3.conf_int(alpha=0.05, cols=None)[1][0]
        reg3_int_intercept_high[city] = reg3.conf_int(alpha=0.05, cols=None)[1][1]
        
    except:
        pass
        
path_outputs = "C:/Users/charl/OneDrive/Bureau/"
        #os.mkdir(path_outputs)
        
df = pd.DataFrame()
df["city"] = reg1_r2.keys()
df["reg1_r2"] = reg1_r2.values()
df["reg1_params_X"] = reg1_params_X.values()
df["reg1_params_intercept"] = reg1_params_intercept.values()
df["reg1_pvalues_X"] = reg1_pvalues_X.values()
df["reg1_pvalues_intercept"] = reg1_pvalues_intercept.values()
df["reg1b_r2"] = reg1b_r2.values()
df["reg1b_net_income"] = reg1b_net_income.values()
df["reg1b_rent"] = reg1b_rent.values()
df["reg1b_pvalues_net_income"] = reg1b_pvalues_net_income.values()
df["reg1b_pvalues_rent"] = reg1b_pvalues_rent.values()
df["reg2_r2"] = reg2_r2.values()
df["reg2_params_X"] = reg2_params_X.values()
df["reg2_params_b"] = reg2_params_b.values()
df["reg2_params_intercept"] = reg2_params_intercept.values()
df["reg2_ctrl"] = reg2_ctrl.values()
df["reg2_pvalues_X"] = reg2_pvalues_X.values()
df["reg2_pvalues_intercept"] = reg2_pvalues_intercept.values()
df["reg2_pvalues_ctrl"] = reg2_pvalues_ctrl.values()
df["reg3_r2"] = reg3_r2.values()
df["reg3_params_X"] = reg3_params_X.values()
df["reg3_params_beta"] = reg3_params_beta.values()
df["reg3_params_intercept"] = reg3_params_intercept.values()
df["reg3_pvalues_X"] = reg3_pvalues_X.values()
df["reg3_pvalues_intercept"] = reg3_pvalues_intercept.values()
df["reg3_bse_X"] = reg3_bse_X.values()
df["reg2_bse_X"] = reg2_bse_X.values()
df["reg1_bse_X"] = reg1_bse_X.values()
df["reg1b_bse_X"] = reg1b_bse_X.values()
df["reg3_bse_intercept"] = reg3_bse_intercept.values()
df["reg2_bse_intercept"] = reg2_bse_intercept.values()
df["reg1_bse_intercept"] = reg1_bse_intercept.values()
df["reg1b_bse_intercept"] = reg1b_bse_intercept.values()
    
df["reg3_int_X_low"] = reg3_int_X_low.values()
df["reg3_int_intercept_low"] = reg3_int_intercept_low.values()
df["reg3_int_X_high"] = reg3_int_X_high.values()
df["reg3_int_intercept_high"] = reg3_int_intercept_high.values()
df["reg2_int_X_low"] = reg2_int_X_low.values()
df["reg2_int_intercept_low"] = reg2_int_intercept_low.values()
df["reg2_int_X_high"] = reg2_int_X_high.values()
df["reg2_int_intercept_high"] = reg2_int_intercept_high.values()
df["reg1_int_X_low"] = reg1_int_X_low.values()
df["reg1_int_intercept_low"] = reg1_int_intercept_low.values()
df["reg1_int_X_high"] = reg1_int_X_high.values()
df["reg1_int_intercept_high"] = reg1_int_intercept_high.values()
df["reg1b_int_X_low"] = reg1b_int_X_low.values()
df["reg1b_int_intercept_low"] = reg1b_int_intercept_low.values()
df["reg1b_int_X_high"] = reg1b_int_X_high.values()
df["reg1b_int_intercept_high"] = reg1b_int_intercept_high.values()
#df["robust"] = robust
df.to_excel(path_outputs + 'euclidian_distance.xlsx')
    

### END OF THE REGRESSIONS

### SUMMARY STATISTICS
path_outputs = 'C:/Users/charl/OneDrive/Bureau/'
df = pd.read_excel(path_outputs + 'euclidian_distance.xlsx')

#Stats des sur les R2
df = df[df.reg1_r2 < 1]
percentiles_R2 = pd.DataFrame(columns = ["reg1", "reg1b", "reg2", "reg3"], index = ["min", "10", "25", "50", "75", "90", "max"])
percentiles_R2["reg1"] = [np.nanmin(df.reg1_r2[df.reg1_r2 > -1]), np.nanpercentile(df.reg1_r2, 10), np.nanpercentile(df.reg1_r2, 25), np.nanpercentile(df.reg1_r2, 50), np.nanpercentile(df.reg1_r2, 75), np.nanpercentile(df.reg1_r2, 90), np.nanmax(df.reg1_r2[df.reg1_r2 > -1])]
percentiles_R2["reg1b"] = [np.nanmin(df.reg1b_r2[df.reg1b_r2 > -1]), np.nanpercentile(df.reg1b_r2, 10), np.nanpercentile(df.reg1b_r2, 25), np.nanpercentile(df.reg1b_r2, 50), np.nanpercentile(df.reg1b_r2, 75), np.nanpercentile(df.reg1b_r2, 90), np.nanmax(df.reg1b_r2[df.reg1b_r2 > -1])]
percentiles_R2["reg2"] = [np.nanmin(df.reg2_r2[df.reg2_r2 > -1]), np.nanpercentile(df.reg2_r2, 10), np.nanpercentile(df.reg2_r2, 25), np.nanpercentile(df.reg2_r2, 50), np.nanpercentile(df.reg2_r2, 75), np.nanpercentile(df.reg2_r2, 90), np.nanmax(df.reg2_r2[df.reg2_r2 > -1])]
percentiles_R2["reg3"] = [np.nanmin(df.reg3_r2[df.reg3_r2 > -1]), np.nanpercentile(df.reg3_r2, 10), np.nanpercentile(df.reg3_r2, 25), np.nanpercentile(df.reg3_r2, 50), np.nanpercentile(df.reg3_r2, 75), np.nanpercentile(df.reg3_r2, 90), np.nanmax(df.reg3_r2[df.reg3_r2 > -1])]
print(percentiles_R2.to_latex())
#0.001081 
#0.023849
#0.075016
#0.222568
#0.405579
#0.581769
#0.901413 &


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

print(sum(df.reg1b_pvalues_net_income > 0.05)) #76
print(sum((df.reg1b_pvalues_net_income < 0.05) & (df.reg1b_net_income < 0)))
print(sum((df.reg1b_pvalues_net_income < 0.05) & (df.reg1b_net_income > 0)))
#Pas significatif dans 76, négatif dans 40 villes, positif dans 75

print(sum(df.reg1b_pvalues_rent > 0.05))
print(sum((df.reg1b_pvalues_rent < 0.05) & (df.reg1b_rent < 0)))
print(sum((df.reg1b_pvalues_rent < 0.05) & (df.reg1b_rent > 0)))
#Pas significatuf dans 36 villes, négatif dans 140 villes, positif dans 15

df["robust_cities"] = ((df.reg3_pvalues_X < 0.05) &(df.reg3_params_X > 0) & (df.reg2_pvalues_X < 0.05) & (df.reg2_params_X > 0) & (df.reg1_pvalues_X < 0.05) & (df.reg1_params_X > 0))
### SECOND STAGE


results = pd.read_excel(path_outputs + 'avg.xlsx')
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
     driving, transit, grille, centre, distance_cbd, conversion_rate) = import_data(list_city, path_data, city, path_folder)        
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

#Planification ? Rent data quality ?

### param (et param < 0) reg1, 2 et 3

y = np.array((second_stage_reg.reg1_params_X)[(~np.isnan(np.log(second_stage_reg.reg1_params_X))) & (second_stage_reg.reg1_params_X > 0) & (second_stage_reg.reg1_pvalues_X < 0.05) ])
X = pd.DataFrame(second_stage_reg.iloc[:, [13, 14, 15, 16, 17, 18, 19]][(~np.isnan(np.log(second_stage_reg.reg1_params_X))) & (second_stage_reg.reg1_params_X > 0) & (second_stage_reg.reg1_pvalues_X < 0.05) ])
#X = np.c_[np.log(X.iloc[:, 0:6]), X.iloc[:, 6]]
ols = sm.OLS(y, X)
reg6 = ols.fit(cov_type='HC1') #HC3 censé être mieux
reg6.summary()

y = np.array(np.log(second_stage_reg.reg1_params_X)[(~np.isnan(np.log(second_stage_reg.reg1_params_X))) & (second_stage_reg.reg1_params_X > 0) & (second_stage_reg.reg1_pvalues_X < 0.05) ])
X = pd.DataFrame(second_stage_reg.iloc[:, [13, 14, 15, 16, 17, 18, 19]][(~np.isnan(np.log(second_stage_reg.reg1_params_X))) & (second_stage_reg.reg1_params_X > 0) & (second_stage_reg.reg1_pvalues_X < 0.05) ])
X = np.c_[np.log(X.iloc[:, 0:5]), X.iloc[:, 5:7]]
X = pd.DataFrame(X, columns = ['ln(population)', 'ln(income)', 'ln(land_prices)', 'ln(commuting_price)', 'ln(commuting_time)', 'polycentricity', 'constant'])
ols = sm.OLS(y, X)
reg7 = ols.fit(cov_type='HC1') #HC3 censé être mieux
reg7.summary()

print(Stargazer([reg6, reg7]).render_latex())














































'''
df = pd.read_excel(path_outputs + 'reg_results.xlsx')
plt.hist(df.reg1_r2)
plt.hist(df.reg3_r2[~np.isinf(df.reg3_r2)])
plt.hist(np.array(list(reg2_r2.values()))[~np.isinf(np.array(list(reg2_r2.values())))])

sum(df["pvalue_density_grad"] > 0.01)
sum(df["reg2_pvalues_X"] > 0.01) #102
sum(df["reg2_pvalues_X"] > 0.05) #102
sum(df["reg3_pvalues_X"] > 0.01) #42

#Housing/rents pas très fiable

plt.hist(df.reg2_params_b, range = (0, 1)) #113 / 192 dans l'intervalle
plt.hist(df.reg3_params_beta, range = (0, 1)) #145 / 192 dans l'intervalle

robust = (df.reg3_params_beta<1) & (df.reg3_params_beta > 0) & (df.reg2_params_b<1) & (df.reg2_params_b > 0) & (df["reg2_pvalues_X"] < 0.05) & (df["reg3_pvalues_X"] < 0.05)
#57 villes
df.city[]
#Robust

results = pd.read_excel("C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/validation_20210504/reg_results.xlsx")
sns.distplot(results.reg1_pvalues_X[(results.reg1_pvalues_X >0) &(results.reg1_pvalues_X <1)]) 
plt.scatter(results.reg1_pvalues_X, results.Continent)
np.nanmean(results.reg1_pvalues_X[(results.reg1_pvalues_X >0) &(results.reg1_pvalues_X <1)])
np.nanmedian(results.reg1_pvalues_X[(results.reg1_pvalues_X >0) &(results.reg1_pvalues_X <1)])
np.nanmin(results.reg1_pvalues_X[(results.reg1_pvalues_X >0) &(results.reg1_pvalues_X <1)])
np.nanmax(results.reg1_pvalues_X[(results.reg1_pvalues_X >0) &(results.reg1_pvalues_X <1)])


#Reg2: rent explains housing
reg2_nonrob = results.city[results.reg2_pvalues_X > 0.1]

city_continent = pd.read_csv(path_folder + "CityDatabases/cityDatabase_221NewCities.csv")
city_continent = city_continent.iloc[:, [0, 2]]
city_continent = city_continent.drop_duplicates(subset = "City")
city_continent = city_continent.sort_values('City')
results = results.merge(city_continent, left_on = "city", right_on = "City")


df = df[((df.reg2_pvalues_X < 0.1) & (df.reg3_pvalues_X < 0.1))]
color_tab = {'North_America':'red', 'Europe':'green', 'Asia':'blue', 'Oceania':'yellow', 'South_America': 'brown', 'Africa': 'orange'}
plt.figure(figsize=(20, 20))
plt.rcParams.update({'font.size': 25})
colors = list(df['Continent'].unique())
for i in range(0 , len(colors)):
    data = df.loc[df['Continent'] == colors[i]]
    plt.scatter(data.reg2_params_X, data.reg3_params_X, color=data.Continent.map(color_tab), label=colors[i], s = 200)
plt.rcParams.update({'font.size': 18})
for i in range(len(df)):
    plt.annotate(df.city.iloc[i], (df.reg2_params_X.iloc[i] + 0.01, df.reg3_params_X.iloc[i] + 0.01))
plt.axvline(0)
plt.axhline(0)
plt.legend() 
plt.xlabel("Rent-housing gradient")
plt.ylabel("Net income - rent gradient")
plt.axvline(0)
plt.axhline(0)
plt.show()

sum(df.reg2_params_X < 0) #41
sum(df.reg3_params_X < 0) #6

plt.rcParams.update({'font.size': 15})
plt.hist(df.reg1_params_X[df.reg1_pvalues_X < 0.1]) #Exception: Brasilia, Cartagena, Singapore
sum(df.reg1_pvalues_X > 0.1) #Exception: Arequipa, Cartagena, Concepcion, Isfahan, Manaus, Mashhad, Medan, Porto_Alegre, Rabat, Recife, Samara, Sousse, Tijuana, Trujillo
df.city[df.reg1_pvalues_X > 0.05]

###### Same set of parameters for all cities?

dg = pd.DataFrame(data = np.repeat(np.zeros(192), 192, axis = 0).reshape(192, 192), index = df.city, columns = df.city)

for i in (df.city):
    for j in (df.city):
        if (i != j):
            print((df["reg3_params_X"][df.city == i].squeeze() - df["reg3_params_X"][df.city == j].squeeze()) / np.sqrt((df["reg3_bse_X"][df.city == i].squeeze() ** 2) + (df["reg3_bse_X"][df.city == j].squeeze() ** 2)))
            dg[i].loc[j] = np.abs((df["reg3_params_X"][df.city == i].squeeze() - df["reg3_params_X"][df.city == j].squeeze()) / np.sqrt((df["reg3_bse_X"][df.city == i].squeeze() ** 2) + (df["reg3_bse_X"][df.city == j].squeeze() ** 2)))
dg.to_excel(path_outputs + 'same_set_parameters.xlsx')

dg = pd.read_excel(path_outputs + 'same_set_parameters.xlsx')

dg_bis = dg
for i in (df.city):
    dg_bis[i][dg_bis[i] < 1.96] = 1
    dg_bis[i][dg_bis[i] > 1.96] = 0
    
dg_bis.index[dg_bis.Paris == 1]
dg_bis.index[dg_bis.Atlanta == 1]
dg_bis.index[dg_bis.Rio_de_Janeiro == 1]

paris = dg_bis["Paris"]
atlanta = dg_bis["Atlanta"]
rio = dg_bis["Rio_de_Janeiro"]

paris.to_excel(path_outputs + 'paris.xlsx')
atlanta.to_excel(path_outputs + 'atlanta.xlsx')
rio.to_excel(path_outputs + 'rio.xlsx')

sum(paris == 1)
sum(atlanta == 1)
sum(rio == 1)

#### GINI

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

list_city = pd.read_csv(path_data + 'CityDatabases/'+'cityDatabase'+'.csv')   
list_city = list_city[['City', 'Country']]  
list_city.columns = ["city", "Country"]
list_city = list_city.drop_duplicates()
results = results.merge(list_city, on = "city",how = "left")
results = results.merge(gini, on = "Country")

df = pd.read_excel("C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/dataset_regressions_v2.xlsx")
results = results.merge(df, on = "city")
plt.scatter(results.reg3_params_X, results.income)

color_tab = {'North_America':'red', 'Europe':'green', 'Asia':'blue', 'Oceania':'yellow', 'South_America': 'brown', 'Africa': 'orange'}
plt.figure(figsize=(20, 20))
plt.rcParams.update({'font.size': 25})
colors = list(results['Continent'].unique())
for i in range(0 , len(colors)):
    data = results.loc[results['Continent'] == colors[i]]
    plt.scatter(data.reg2_params_X, data.income, color=data.Continent.map(color_tab), label=colors[i], s = 200)
plt.rcParams.update({'font.size': 18})
plt.legend() 

plt.show()'''