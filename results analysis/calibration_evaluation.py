# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 09:45:07 2021

@author: charl
"""
import numpy as np
import matplotlib.pyplot as plt

path_outputs = "C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/calibration_20210819b/"

selected_cells = np.load(path_outputs + "d_selected_cells.npy", allow_pickle = True)
selected_cells = np.array(selected_cells, ndmin = 1)[0]
selected_cells = np.array(list(selected_cells.values()))
sum(selected_cells < 100)
selected_cities = (selected_cells > 100)

beta = np.load(path_outputs + "beta.npy", allow_pickle = True)
beta = np.array(beta, ndmin = 1)[0]
plt.figure(figsize = (15, 10))
plt.rcParams.update({'font.size': 60})
plt.hist(beta.values())
plt.xlim(0, 1)
plt.xticks([0.2, 0.4, 0.6, 0.8, 1])
plt.ylabel("Number of cities")

plt.figure(figsize = (15, 10))
plt.rcParams.update({'font.size': 60})
b = np.load(path_outputs + "b.npy", allow_pickle = True)
b = np.array(b, ndmin = 1)[0]
plt.hist(np.array(list(b.values())))
plt.xlim(0, 1)
plt.xticks([0.2, 0.4, 0.6, 0.8, 1])
plt.ylabel("Number of cities")

kappa = np.load(path_outputs + "kappa.npy", allow_pickle = True)
kappa = np.array(kappa, ndmin = 1)[0]
plt.figure(figsize = (15, 10))
plt.rcParams.update({'font.size': 60})
#plt.hist((kappa.values()))
plt.hist(np.array(list(kappa.values())), range = (0, 40000))
plt.ylabel("Number of cities")
plt.hist((kappa.values()), range = (0, 500))
plt.hist((kappa.values()), range = (0, 50))
plt.hist((kappa.values()), range = (0, 5))
plt.hist((kappa.values()), range = (0, 0.5))
plt.hist((kappa.values()), range = (0, 0.05))
sum(np.array(list(kappa.values())) > 40000) #5 outliers excluded

Ro = np.load(path_outputs + "Ro.npy", allow_pickle = True)
Ro = np.array(Ro, ndmin = 1)[0]
plt.figure(figsize = (15, 10))
plt.rcParams.update({'font.size': 60})
plt.hist(Ro.values(), range = (0, 800))
plt.ylabel("Number of cities")
sum(np.array(list(Ro.values())) > 800) #5 outliers excluded
                                                 
r2density_scells2 = np.load(path_outputs + "r2density_scells2.npy", allow_pickle = True)
r2density_scells2 = np.array(r2density_scells2, ndmin = 1)[0]
r2rent_scells2 = np.load(path_outputs + "r2rent_scells2.npy", allow_pickle = True)
r2rent_scells2 = np.array(r2rent_scells2, ndmin = 1)[0]
r2size_scells2 = np.load(path_outputs + "r2size_scells2.npy", allow_pickle = True)
r2size_scells2 = np.array(r2size_scells2, ndmin = 1)[0]

d_corr_density_scells2 = np.load(path_outputs + "d_corr_density_scells2.npy", allow_pickle = True)
d_corr_density_scells2 = np.array(d_corr_density_scells2, ndmin = 1)[0]
d_corr_rent_scells2 = np.load(path_outputs + "d_corr_rent_scells2.npy", allow_pickle = True)
d_corr_rent_scells2 = np.array(d_corr_rent_scells2, ndmin = 1)[0]
d_corr_size_scells2 = np.load(path_outputs + "d_corr_size_scells2.npy", allow_pickle = True)
d_corr_size_scells2 = np.array(d_corr_size_scells2, ndmin = 1)[0]

mae_density_scells2 = np.load(path_outputs + "mae_density_scells2.npy", allow_pickle = True)
mae_density_scells2 = np.array(mae_density_scells2, ndmin = 1)[0]
mae_rent_scells2 = np.load(path_outputs + "mae_rent_scells2.npy", allow_pickle = True)
mae_rent_scells2 = np.array(mae_rent_scells2, ndmin = 1)[0]
mae_size_scells2 = np.load(path_outputs + "mae_size_scells2.npy", allow_pickle = True)
mae_size_scells2 = np.array(mae_size_scells2, ndmin = 1)[0]


rae_density_scells2 = np.load(path_outputs + "rae_density_scells2.npy", allow_pickle = True)
rae_density_scells2 = np.array(rae_density_scells2, ndmin = 1)[0]
rae_rent_scells2 = np.load(path_outputs + "rae_rent_scells2.npy", allow_pickle = True)
rae_rent_scells2 = np.array(rae_rent_scells2, ndmin = 1)[0]
rae_size_scells2 = np.load(path_outputs + "rae_size_scells2.npy", allow_pickle = True)
rae_size_scells2 = np.array(rae_size_scells2, ndmin = 1)[0]

df = pd.DataFrame()
df["City"] = np.array(list(r2density_scells2.keys()))
df["r2density"] = np.array(list(r2density_scells2.values()))
df["r2rent"] = np.array(list(r2rent_scells2.values()))
df["r2size"] = np.array(list(r2size_scells2.values()))
df["corrrdensity"] = np.array(list(d_corr_density_scells2.values()))[:, 0]
df["corrrent"] = np.array(list(d_corr_rent_scells2.values()))[:, 0]
df["corrsize"] = np.array(list(d_corr_size_scells2.values()))[:, 0]
df["mae_density"] = np.array(list(mae_density_scells2.values()))
df["mae_rent"] = np.array(list(mae_rent_scells2.values()))
df["mae_size"] = np.array(list(mae_size_scells2.values()))
df["rae_density"] = np.array(list(rae_density_scells2.values()))
df["rae_rent"] = np.array(list(rae_rent_scells2.values()))
df["rae_size"] = np.array(list(rae_size_scells2.values()))

table_validation = df.describe()





#add continents
city_continent = pd.read_csv(path_folder + "CityDatabases/cityDatabase_221NewCities.csv")
city_continent = city_continent.iloc[:, [0, 2]]
city_continent = city_continent.drop_duplicates(subset = "City")
city_continent = city_continent.sort_values('City')



df = pd.DataFrame()
df["City"] = np.array(list(r2density_scells2.keys()))
df = df.merge(city_continent, on = "City")
df["r2density"] = np.array(list(r2density_scells2.values()))
df["r2rent"] = np.array(list(r2rent_scells2.values()))
df["r2size"] = np.array(list(r2size_scells2.values()))
selected_cities = np.delete(selected_cities, 153)
df = df[selected_cities]

color_tab = {'North_America':'red', 'Europe':'green', 'Asia':'blue', 'Oceania':'yellow', 'South_America': 'brown', 'Africa': 'orange'}
plt.figure(figsize=(20, 20))
plt.rcParams.update({'font.size': 40})
colors = list(df['Continent'].unique())
for i in range(0 , len(colors)):
    data = df.loc[df['Continent'] == colors[i]]
    plt.scatter(data.r2rent, data.r2density, color=data.Continent.map(color_tab), label=colors[i], s = 200)
plt.rcParams.update({'font.size': 25})
plt.legend()
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.axvline(0)
plt.axhline(0)
plt.xlabel("Rents")
plt.ylabel("Density")
plt.show()

sum((np.array(list(r2density.values())) < -0) |  (np.array(list(r2rent.values())) < -0))
sum((df.r2density < -1) |  (df.r2rent < -1))
np.array(list(r2density.keys()))[((np.array(list(r2density.values())) < -0) |  (np.array(list(r2rent.values())) < -0))]
np.nanmean(df.r2rent[df.r2rent > - 100])
np.nanmean(df.r2density[df.r2density > - 100])
np.nanmean(df.r2size[df.r2size > - 100])
np.nanmedian(df.r2rent[df.r2rent > - 100])
np.nanmedian(df.r2density[df.r2density > - 100])
np.nanmedian(df.r2size[df.r2size > - 100])


d_income = np.load(path_outputs + "d_income.npy", allow_pickle = True)
df = pd.read_excel("C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/city_characteristics_20210810.xlsx")
data_income = df.income
plt.scatter(df.income, d_income.values())
plt.xlim(0, 100000)
plt.ylim(0, 100000)

d_share_car = {}
d_share_transit = {}
d_share_walking = {}

for city in np.delete(np.delete(np.delete(np.unique(list_city.City), 168), 153), 78):
    density = np.load(path_no_policy_simulation + city + "_density.npy")
    modal_share = np.load(path_no_policy_simulation + city + "_modal_shares.npy")
    d_share_car[city] = np.nansum(density * (modal_share == 0)) / np.nansum(density)
    d_share_transit[city] = np.nansum(density * (modal_share == 1)) / np.nansum(density)
    d_share_walking[city] = np.nansum(density * (modal_share == 2)) / np.nansum(density)

    
modal_shares = import_modal_shares_wiki(path_folder)
modal_shares = modal_shares.iloc[:, 0:5]

simul = pd.DataFrame()
simul["city"] = np.array(list(d_share_car.keys()))
simul["simul_car"] = np.array(list(d_share_car.values()))
simul["simul_walking"] = np.array(list(d_share_walking.values()))
simul["simul_transit"] = np.array(list(d_share_transit.values()))

simul = simul.merge(modal_shares, on = "city", how = 'left')
epomm = pd.read_excel(path_folder + 'modal_shares_data.ods', engine = "odf")
modal_shares = simul.merge(epomm, on = "city")
modal_shares["private car"].astype(float)[np.isnan(modal_shares["private car"].astype(float))] = modal_shares["car"].astype(float)
modal_shares["public transport"].astype(float)[np.isnan(modal_shares["public transport"].astype(float))] = modal_shares["public_transport"].astype(float)
modal_shares["walking"].astype(float)[np.isnan(modal_shares["walking"].astype(float))] = modal_shares["walk"].astype(float)
modal_shares["cycling"].astype(float)[np.isnan(modal_shares["cycling"].astype(float))] = modal_shares["bike"].astype(float)
modal_shares = modal_shares[(~np.isnan(modal_shares["cycling"].astype(float)))]

plt.figure(figsize = (15, 10))
plt.rcParams.update({'font.size': 20})
plt.scatter(modal_shares["private car"] * 100, modal_shares.simul_car * 100, s = 200)
plt.xlabel("Modal share of private car (EPOMM data - %)", size = 20)
plt.ylabel("Modal share of private car (Simulations - %)", size = 20)
plt.xlim(0, 100)
plt.ylim(0, 100)
for i in np.arange(73):
    plt.annotate(np.array(modal_shares.city)[i], (np.array(modal_shares["private car"])[i] * 100, np.array(modal_shares.simul_car)[i] * 100), size = 20)

from sklearn.linear_model import LinearRegression
x = np.array(modal_shares.simul_car[~np.isnan(modal_shares["private car"].astype(float))]).reshape(-1, 1)
y = np.array(modal_shares["private car"].astype(float)[~np.isnan(modal_shares["private car"].astype(float))]).reshape(-1, 1) 
modeleReg=LinearRegression()
modeleReg.fit(x, y)
modeleReg.score(x, y)


plt.figure(figsize = (15, 10))
plt.rcParams.update({'font.size': 60})
plt.scatter((modal_shares["public transport"])  * 100, modal_shares.simul_transit * 100, s = 200)
plt.xlabel("Data")
plt.ylabel("Simulations")
plt.xlim(0, 100)
plt.ylim(0, 100)

x = np.array(modal_shares.simul_transit[~np.isnan(modal_shares["public transport"].astype(float))]).reshape(-1, 1)
y = np.array(modal_shares["public transport"].astype(float)[~np.isnan(modal_shares["public transport"].astype(float))]).reshape(-1, 1) 
modeleReg=LinearRegression()
modeleReg.fit(x, y)
modeleReg.score(x, y)

plt.figure(figsize = (15, 10))
plt.rcParams.update({'font.size': 60})
plt.scatter((modal_shares["walking"] + modal_shares["cycling"])  * 100, modal_shares.simul_walking * 100, s = 200)
plt.xlabel("Data")
plt.ylabel("Simulations")
plt.xlim(0, 60)
plt.ylim(0, 30)

x = np.array(modal_shares.simul_walking[~np.isnan((modal_shares["walking"] + modal_shares["cycling"]).astype(float))]).reshape(-1, 1)
y = np.array((modal_shares["walking"] + modal_shares["cycling"]).astype(float)[~np.isnan((modal_shares["walking"] + modal_shares["cycling"]).astype(float))]).reshape(-1, 1) 
modeleReg=LinearRegression()
modeleReg.fit(x, y)
modeleReg.score(x, y)

import scipy.stats
scipy.stats.pearsonr(np.array(modal_shares["private car"]), np.array(modal_shares.simul_car))
scipy.stats.pearsonr(np.array(modal_shares["public transport"]), np.array(modal_shares.simul_transit))
scipy.stats.pearsonr(np.array(modal_shares["walking"] + modal_shares["cycling"]), np.array(modal_shares.simul_walking))
#0.22/0.059 for car, 0.28 et 0.017, 0.02 et 0.84 for walking

epomm = pd.read_excel(path_folder + 'modal_shares_data.ods', engine = "odf")
epomm = epomm.merge(simul, on = "city")

plt.figure(figsize=(10, 10))
plt.rcParams.update({'font.size': 25})
plt.scatter(epomm["car"], epomm.simul_car * 100, s = 50)
plt.xlabel("Data")
plt.ylabel("Simulations")
plt.xlim(0, 100)
plt.ylim(0, 100)

x = np.array(epomm.simul_car[~np.isnan(epomm["car"].astype(float))]).reshape(-1, 1)
y = np.array(epomm["car"].astype(float)[~np.isnan(epomm["car"].astype(float))]).reshape(-1, 1) 
modeleReg=LinearRegression()
modeleReg.fit(x, y)
modeleReg.score(x, y)


plt.figure(figsize=(10, 10))
plt.rcParams.update({'font.size': 25})
plt.scatter((epomm["public_transport"]), epomm.simul_transit * 100, s = 50)
plt.xlabel("Data")
plt.ylabel("Simulations")
plt.xlim(0, 100)
plt.ylim(0, 100)

x = np.array(epomm.simul_transit[~np.isnan(epomm["public_transport"].astype(float))]).reshape(-1, 1)
y = np.array(epomm["public_transport"].astype(float)[~np.isnan(epomm["public_transport"].astype(float))]).reshape(-1, 1) 
modeleReg=LinearRegression()
modeleReg.fit(x, y)
modeleReg.score(x, y)

plt.figure(figsize=(10, 10))
plt.rcParams.update({'font.size': 25})
plt.scatter((epomm["walk"] + epomm["bike"]), epomm.simul_walking * 100, s = 50)
plt.xlabel("Data")
plt.ylabel("Simulations")
plt.xlim(0, 70)
plt.ylim(0, 30)

x = np.array(epomm.simul_walking[~np.isnan((epomm["walk"] + epomm["bike"]).astype(float))]).reshape(-1, 1)
y = np.array((epomm["walk"] + epomm["bike"]).astype(float)[~np.isnan((epomm["walk"] + epomm["bike"]).astype(float))]).reshape(-1, 1) 
modeleReg=LinearRegression()
modeleReg.fit(x, y)
modeleReg.score(x, y)

import scipy.stats
scipy.stats.pearsonr(np.array(epomm["car"].astype(float)[~np.isnan(epomm["car"].astype(float))]), np.array(epomm.simul_car)[~np.isnan(epomm["car"].astype(float))])
scipy.stats.pearsonr(np.array(epomm["public_transport"].astype(float)[~np.isnan(epomm["public_transport"].astype(float))]), np.array(epomm.simul_transit.astype(float)[~np.isnan(epomm["public_transport"].astype(float))]))
scipy.stats.pearsonr(np.array(epomm["walk"]).astype(float)[~np.isnan((epomm["walk"] + epomm["bike"]).astype(float))], np.array(epomm.simul_walking).astype(float)[~np.isnan((epomm["walk"] + epomm["bike"]).astype(float))])
#0.22/0.059 for car, 0.28 et 0.017, 0.02 et 0.84 for walking

modal_shares = import_modal_shares_wiki(path_folder)
modal_shares.columns = ["city", "walk_wiki", "bike_wiki", "public_transport_wiki", "car_wiki", "U", "U2", "U3"]
epomm = pd.read_excel(path_folder + 'modal_shares_data.ods', engine = "odf")
epomm.columns = ["city", "country", "walk_epomm", "bike_epomm", "public_transport_epomm", "car_epomm", "U4"]
compare_modal_data = modal_shares.merge(epomm, "inner")
compare_modal_data = compare_modal_data[(~np.isnan(compare_modal_data.car_wiki.astype(float))) & (~np.isnan(compare_modal_data.car_epomm.astype(float)))]

scipy.stats.pearsonr(compare_modal_data.car_wiki.astype(float), compare_modal_data.car_epomm.astype(float))
scipy.stats.pearsonr(compare_modal_data.public_transport_wiki.astype(float), compare_modal_data.public_transport_epomm.astype(float))
scipy.stats.pearsonr(compare_modal_data.walk_wiki.astype(float), compare_modal_data.walk_epomm.astype(float))
scipy.stats.pearsonr(compare_modal_data.bike_wiki.astype(float), compare_modal_data.bike_epomm.astype(float))


from sklearn.linear_model import LinearRegression
plt.figure(figsize=(8, 8))
plt.rcParams.update({'font.size': 12})
x = np.array(compare_modal_data.bike_epomm.astype(float)).reshape(-1, 1)
y = np.array(compare_modal_data.bike_wiki.astype(float)).reshape(-1, 1)
plt.plot(x, y, 'o', color = 'black')
m, b = np.polyfit(np.transpose(x).squeeze(), y, 1)
plt.plot(x, m*x + b, color = 'red')
plt.xlabel("EPOMM")
plt.ylabel("Various sources")

modeleReg=LinearRegression()
modeleReg.fit(x, y)
print(modeleReg.intercept_)
print(modeleReg.coef_)
modeleReg.score(x, y) #13.8


emission_2015 = {}
emission_2035 = {}
utility_2015 = {}
utility_2035 = {}

path_no_policy_simulation = "C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/BAU_20210819/"

for city in np.delete(np.delete(np.delete(np.unique(list_city.City), 168), 153), 78):
    emissions_per_capita = np.load(path_no_policy_simulation + city + "_emissions_per_capita.npy")
    utility = np.load(path_no_policy_simulation + city + "_utility.npy")
    emission_2015[city] = emissions_per_capita[0]
    emission_2035[city] = emissions_per_capita[20]
    utility_2015[city] = utility[0]
    utility_2035[city] = utility[20]
    
no_policy = pd.DataFrame()
no_policy["city"] = emission_2015.keys()
no_policy["emissions_2015"] = emission_2015.values()
no_policy["emissions_2035"] = emission_2035.values()
no_policy["utility_2015"] = utility_2015.values()
no_policy["utility_2035"] = utility_2035.values()

city_continent = pd.read_csv(path_data + "CityDatabases/cityDatabase_221NewCities.csv")
city_continent = city_continent.iloc[:, [0, 2]]
city_continent = city_continent.drop_duplicates(subset = "City")
city_continent = city_continent.sort_values('City')
no_policy = no_policy.merge(city_continent, left_on = "city", right_on = "City", how = 'left')



color_tab = {'North_America':'red', 'Europe':'green', 'Asia':'blue', 'Oceania':'yellow', 'South_America': 'brown', 'Africa': 'orange'}
plt.figure(figsize=(20, 20))
plt.rcParams.update({'font.size': 40})
colors = list(no_policy['Continent'].unique())
for i in range(0 , len(colors)):
    data = no_policy.loc[no_policy['Continent'] == colors[i]]
    plt.scatter(data.utility_2015, data.emissions_2015, color=data.Continent.map(color_tab), label=colors[i], s = 200)
plt.rcParams.update({'font.size': 15})
plt.rcParams.update({'font.size': 25})
plt.legend()
plt.xlabel("Utility")
plt.ylabel("Transport emissions per capita per year (gCO2)")
plt.show()



##### COMPARISON EMISSIONS DATA

### DELIX DATABASE

data_city_emissions = no_policy.iloc[:, 0:4]
felix_data = pd.read_excel(path_folder + "emissions_databases/datapaper felix/DATA/D_FINAL.xlsx", header = 0)
columns_name = felix_data.columns
felix_data_for_comparison = felix_data.loc[:, ["City name", "Scope-1 GHG emissions [tCO2 or tCO2-eq]", "Scope-2 (CDP) [tCO2-eq]", "Total emissions (CDP) [tCO2-eq]","CO2 emissions per capita (PKU) [tCO2/capita]", "Population (CDP)"]]
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
comparison = data_city_emissions.merge(felix_data_for_comparison, left_on = "city", right_on = "City name")
#76

comparison["emissions_felix1"] = comparison["Scope-1 GHG emissions [tCO2 or tCO2-eq]"] / comparison["Population (CDP)"]
comparison["emissions_felix2"] = comparison["Scope-2 (CDP) [tCO2-eq]"] / comparison["Population (CDP)"]


from sklearn.linear_model import LinearRegression
plt.figure(figsize = (13, 8))
plt.rcParams.update({'font.size': 40})
x = np.array(comparison.emissions_2015[~np.isnan(comparison["emissions_felix2"])]).reshape(-1, 1)
y = np.array(comparison["emissions_felix2"][~np.isnan(comparison["emissions_felix2"])]).reshape(-1, 1) * 1000000
plt.plot(x, y, 'o', color = 'black')
m, b = np.polyfit(np.transpose(x).squeeze(), y, 1)
plt.plot(x, m*x + b, color = 'red')
plt.xlabel("Simulations")
plt.ylabel("Data")
plt.ylim(0, 10000000)
for i in range(len(x)):
    plt.annotate(comparison.City[i], (x[i] + 30000, y[i] + 300000), size = 15)

modeleReg=LinearRegression()
modeleReg.fit(x, y)
print(modeleReg.intercept_)
print(modeleReg.coef_)
modeleReg.score(x, y)

scipy.stats.pearsonr(comparison.emissions_2015[~np.isnan(comparison["emissions_felix2"])], comparison["emissions_felix2"][~np.isnan(comparison["emissions_felix2"])])
    
### ERL DATABASE


data_city_emissions = no_policy.iloc[:, 0:4]
erl_data = pd.read_excel(path_folder + "emissions_databases/ERL/ERL_13_064041_SD_Moran Spatial Footprints Data Appendix.xlsx", header = 0, sheet_name = 'S.2.3a - Top 500 Cities')
erl_data = erl_data.iloc[:, [0, 2]]
erl_data.columns = ['city', 'emissions']

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
erl_data.emissions[erl_data.city == 'Valencia'] = 5.7
erl_data.columns = ['city', 'emissions']
data_city_emissions = data_city_emissions.merge(erl_data, how = 'left',on = 'city')


erl_data2 = pd.read_excel("C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Data/emissions_databases/ERL/ERL_13_064041_SD_Moran Spatial Footprints Data Appendix.xlsx", header = 0, sheet_name = 'S.2.3b - Top 500 Per Capita')
erl_data2 = erl_data2.iloc[:, [0, 5]]
erl_data2.columns = ['city', 'emissions2']


data_city_emissions = data_city_emissions.merge(erl_data2, how = 'left', on = 'city')
data_city_emissions.emissions[np.isnan(data_city_emissions.emissions)] = data_city_emissions.emissions2
#data_city_emissions = data_city_emissions.iloc[:, 0:3]
erl = data_city_emissions

from sklearn.linear_model import LinearRegression
plt.figure(figsize = (13, 8))
plt.rcParams.update({'font.size': 40})
x = np.array(data_city_emissions.emissions_2015[~np.isnan(data_city_emissions.emissions)]).reshape(-1, 1)
y = np.array(data_city_emissions.emissions[~np.isnan(data_city_emissions.emissions)]).reshape(-1, 1) * 1000000
plt.plot(x, y, 'o', color = 'black')
m, b = np.polyfit(np.transpose(x).squeeze(), y, 1)
plt.plot(x, m*x + b, color = 'red')
plt.xlabel("Simulations")
plt.ylabel("Data")
for i in range(len(x)):
    plt.annotate(data_city_emissions.city[i], (x[i] + 30000, y[i] + 300000), size = 15)

modeleReg=LinearRegression()
modeleReg.fit(x, y)
print(modeleReg.intercept_)
print(modeleReg.coef_)
modeleReg.score(x, y) #13.8

scipy.stats.pearsonr(data_city_emissions.emissions_2015[~np.isnan(data_city_emissions.emissions)],data_city_emissions.emissions[~np.isnan(data_city_emissions.emissions)])
 

### COM DATABASE

data_city_emissions = no_policy.iloc[:, 0:4]
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
com_data["emissions_per_capita"] = com_data.emissions / com_data.population_in_the_inventory_year
com_data = com_data.merge(com_label, on = 'GCoM_ID', how = 'left')
com_data = com_data[com_data.emission_inventory_sector == 'Transportation']
com_data = com_data[com_data.type_of_emission_inventory == 'baseline_emission_inventory']
direct_emissions_data = com_data[com_data.type_of_emissions == 'direct_emissions']
indirect_emissions_data = com_data[com_data.type_of_emissions == 'indirect_emissions']
direct_emissions_data = direct_emissions_data.iloc[:, 12:14]
indirect_emissions_data = indirect_emissions_data.iloc[:, 12:14]
direct_emissions_data.columns = ['direct_emissions', 'city']
indirect_emissions_data.columns = ['indirect_emissions', 'city']
direct_emissions_data = direct_emissions_data.merge(indirect_emissions_data, on = 'city')
#direct_emissions_data["emissions_per_capita"] = direct_emissions_data.direct_emissions + direct_emissions_data.indirect_emissions
direct_emissions_data["emissions_per_capita"] = direct_emissions_data.direct_emissions 
data_city_emissions = data_city_emissions.merge(direct_emissions_data, on = 'city')

from sklearn.linear_model import LinearRegression
plt.figure(figsize = (13, 8))
plt.rcParams.update({'font.size': 40})
x = np.array(data_city_emissions.emissions_2015[~np.isnan(data_city_emissions.emissions_per_capita)]).reshape(-1, 1)
y = np.array(data_city_emissions.emissions_per_capita[~np.isnan(data_city_emissions.emissions_per_capita)]).reshape(-1, 1) * 1000000
plt.plot(x, y, 'o', color = 'black')
m, b = np.polyfit(np.transpose(x).squeeze(), y, 1)
plt.plot(x, m*x + b, color = 'red')
plt.xlabel("Simulations")
plt.ylabel("Data")
for i in range(len(x)):
    plt.annotate(data_city_emissions.city[i], (x[i] + 30000, y[i] + 300000), size = 15)

modeleReg=LinearRegression()
modeleReg.fit(x, y)
print(modeleReg.intercept_)
print(modeleReg.coef_)
modeleReg.score(x, y) #2.9

scipy.stats.pearsonr(data_city_emissions.emissions_2015[~np.isnan(data_city_emissions.emissions_per_capita)], data_city_emissions.emissions_per_capita[~np.isnan(data_city_emissions.emissions_per_capita)])
    
#### COMPARISON THREE DATABASES

felix_data_for_comparison["emissions_felix1"] = felix_data_for_comparison["Scope-1 GHG emissions [tCO2 or tCO2-eq]"] / felix_data_for_comparison["Population (CDP)"]
felix_data_for_comparison["emissions_felix2"] = felix_data_for_comparison["Scope-2 (CDP) [tCO2-eq]"] / felix_data_for_comparison["Population (CDP)"]

erl_data2.columns = ["City", "ERL"]
erl_data.columns = ["City", "ERL"]
erl = erl_data.append(erl_data2)
erl = erl.drop_duplicates()
erl.City[erl.City == "Cologne"] = "Koeln"
erl.City[erl.City == "Hong Kong"] = "Hong_Kong"
erl.City[erl.City == "Jixi, Heilongjiang"] = "Jixi"
erl.City[erl.City == "Belo Horizonte"] = "Belo_Horizonte"
erl.City[erl.City == "Brussels"] = "Bruxelles-Capitale"
erl.City[erl.City == "Frankfurt_am_Main"] = "Frankfurt am Main"
erl.City[erl.City == "Lyon"] = "Grand Lyon"
erl.City[erl.City == "St. Louis"] = "St Louis"
erl.City[erl.City == "New York"] = "New_York"
felix_data_for_comparison = felix_data_for_comparison.iloc[:, [0, 6]]
felix_data_for_comparison.columns = ["City", "Felix"]
felix_data_for_comparison.City[felix_data_for_comparison.City == "Barreiro"] = "Barreiro"
felix_data_for_comparison.City[felix_data_for_comparison.City == "Nashville and Davidson"] = "Nashville"
felix_data_for_comparison.City[felix_data_for_comparison.City == "Portland, OR"] = "Portland"
felix_data_for_comparison.City[felix_data_for_comparison.City == "Richmond, VA"] = "Richmond"
felix_data_for_comparison.City[felix_data_for_comparison.City == "District of Columbia"] = "Columbia"
felix_data_for_comparison.City[felix_data_for_comparison.City == "Milano"] = "Milan"
direct_emissions_data = direct_emissions_data.iloc[:, [1, 3]]
direct_emissions_data.columns = ["City", "COM"]
direct_emissions_data.City[direct_emissions_data.City == "Barreiro"] = "Barreiro"
direct_emissions_data.City[direct_emissions_data.City == "Dublin City Council"] = "Dublin"
direct_emissions_data.City[direct_emissions_data.City == "Århus"] = "Aarhus"
direct_emissions_data.City[direct_emissions_data.City == "Lille Métropole"] = "Lille"
direct_emissions_data.City[direct_emissions_data.City == "Bucharest District 1"] = "Bucharest"


comparison_data_emissions = felix_data_for_comparison.merge(erl, on = "City", how = 'outer')
comparison_data_emissions = comparison_data_emissions.merge(direct_emissions_data, on = "City", how = "outer")
comparison_data_emissions = comparison_data_emissions.loc[~comparison_data_emissions.City.str.startswith("Unknown"), :]


felix_erl = comparison_data_emissions[(~np.isnan(comparison_data_emissions.ERL)) & (~np.isnan(comparison_data_emissions.Felix))]

from sklearn.linear_model import LinearRegression
plt.figure(figsize=(8, 8))
plt.rcParams.update({'font.size': 12})
x = np.array(felix_erl.Felix).reshape(-1, 1)
y = np.array(felix_erl.ERL).reshape(-1, 1)
plt.plot(x, y, 'o', color = 'black')
m, b = np.polyfit(np.transpose(x).squeeze(), y, 1)
plt.plot(x, m*x + b, color = 'red')
plt.xlabel("Nangini et al.")
plt.ylabel("Moran et al.")

modeleReg=LinearRegression()
modeleReg.fit(x, y)
print(modeleReg.intercept_)
print(modeleReg.coef_)
modeleReg.score(x, y) #13.8
scipy.stats.pearsonr(x.squeeze(), y.squeeze())

com_erl = comparison_data_emissions[(~np.isnan(comparison_data_emissions.ERL)) & (~np.isnan(comparison_data_emissions.COM))]

from sklearn.linear_model import LinearRegression
plt.figure(figsize=(8, 8))
plt.rcParams.update({'font.size': 12})
x = np.array(com_erl.COM).reshape(-1, 1)
y = np.array(com_erl.ERL).reshape(-1, 1)
plt.plot(x, y, 'o', color = 'black')
m, b = np.polyfit(np.transpose(x).squeeze(), y, 1)
plt.plot(x, m*x + b, color = 'red')
plt.xlabel("Covenant of Mayors")
plt.ylabel("Moran et al.")

modeleReg=LinearRegression()
modeleReg.fit(x, y)
print(modeleReg.intercept_)
print(modeleReg.coef_)
modeleReg.score(x, y) #13.8
scipy.stats.pearsonr(x.squeeze(), y.squeeze())

com_felix = comparison_data_emissions[(~np.isnan(comparison_data_emissions.Felix)) & (~np.isnan(comparison_data_emissions.COM))]

from sklearn.linear_model import LinearRegression
plt.figure(figsize=(8, 8))
plt.rcParams.update({'font.size': 12})
x = np.array(com_felix.COM).reshape(-1, 1)
y = np.array(com_felix.Felix).reshape(-1, 1)
plt.plot(x, y, 'o', color = 'black')
m, b = np.polyfit(np.transpose(x).squeeze(), y, 1)
plt.plot(x, m*x + b, color = 'red')
plt.xlabel("Covenant of Mayors")
plt.ylabel("Nangini et al.")

modeleReg=LinearRegression()
modeleReg.fit(x, y)
print(modeleReg.intercept_)
print(modeleReg.coef_)
modeleReg.score(x, y) #13.8

scipy.stats.pearsonr(x.squeeze(), y.squeeze())









