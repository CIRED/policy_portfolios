# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 11:55:39 2021

@author: charl
"""

compute_density = pd.read_excel("C:/Users/charl/OneDrive/Bureau/scenarios_densities_20220309.xlsx")
#sample_of_cities = pd.read_excel("C:/Users/charl/OneDrive/Bureau/sample_of_cities.xlsx").loc[:, ['City', 'final_sample']]
compute_density = compute_density.merge(sample_of_cities, on = "City")
compute_density = compute_density.loc[compute_density.final_sample == 1, :]

#### COMPUTE DENSITY - METHODS

compute_density["data_density_2015"] = 0.01 * compute_density.data_pop_2015 / compute_density.data_land_cover_2015
compute_density["data_density_2015_v2"] = 0.01 * compute_density.ESA_population_2015 / compute_density.data_land_cover_2015
compute_density["predicted_density_2015"] = 0.01 * compute_density.data_pop_2015 / compute_density.predicted_land_cover_2015
compute_density["predicted_density_2015_corrected"] = 0.01 * compute_density.data_pop_2015 / compute_density.predicted_land_cover_2015_corrected
compute_density["predicted_density_2015_v2"] = 0.01 * compute_density.predicted_population_2015 / compute_density.predicted_land_cover_2015
compute_density["predicted_density_2015_corrected_v2"] = 0.01 * compute_density.predicted_population_2015_corrected / compute_density.predicted_land_cover_2015_corrected

compute_density["predicted_density_2035"] = 0.01 * compute_density.data_pop_2035 / compute_density.predicted_land_cover_2035
compute_density["predicted_density_2035_corrected"] = 0.01 * compute_density.data_pop_2035 / compute_density.predicted_land_cover_2035_corrected
compute_density["predicted_density_2035_v2"] = 0.01 * compute_density.predicted_population_2035 / compute_density.predicted_land_cover_2035
compute_density["predicted_density_2035_corrected_v2"] = 0.01 * compute_density.predicted_population_2035_corrected / compute_density.predicted_land_cover_2035_corrected

np.nanmean(np.abs((compute_density["predicted_density_2035"] - compute_density["predicted_density_2015"]) / compute_density["predicted_density_2015"]))

plt.hist((compute_density["predicted_density_2035"] - compute_density["predicted_density_2015"]) / compute_density["predicted_density_2015"])
plt.hist((compute_density["predicted_density_2035_corrected"] - compute_density["predicted_density_2015_corrected"]) / compute_density["predicted_density_2015_corrected"])
plt.hist((compute_density["predicted_density_2035_v2"] - compute_density["predicted_density_2015_v2"]) / compute_density["predicted_density_2015_v2"])
plt.hist((compute_density["predicted_density_2035_corrected_v2"] - compute_density["predicted_density_2015_corrected_v2"]) / compute_density["predicted_density_2015_corrected_v2"])


np.abs(100 * (compute_density["predicted_density_2015"] - compute_density["data_density_2015"]) / compute_density["data_density_2015"]).astype(float).describe()
np.abs(100 * (compute_density["predicted_density_2015_corrected"] - compute_density["data_density_2015"]) / compute_density["data_density_2015"]).astype(float).describe()
np.abs(100 * (compute_density["predicted_density_2015_v2"] - compute_density["data_density_2015"]) / compute_density["data_density_2015"]).astype(float).describe()
np.abs(100 * (compute_density["predicted_density_2015_corrected_v2"] - compute_density["data_density_2015"]) / compute_density["data_density_2015"]).astype(float).describe()

np.abs(100 * (compute_density["predicted_density_2015"] - compute_density["data_density_2015_v2"]) / compute_density["data_density_2015_v2"]).astype(float).describe()
np.abs(100 * (compute_density["predicted_density_2015_corrected"] - compute_density["data_density_2015_v2"]) / compute_density["data_density_2015_v2"]).astype(float).describe()
np.abs(100 * (compute_density["predicted_density_2015_v2"] - compute_density["data_density_2015_v2"]) / compute_density["data_density_2015_v2"]).astype(float).describe()
np.abs(100 * (compute_density["predicted_density_2015_corrected_v2"] - compute_density["data_density_2015_v2"]) / compute_density["data_density_2015_v2"]).astype(float).describe()

compute_density["var_density"] = 100 * (compute_density.predicted_density_2035 - compute_density.predicted_density_2015) / compute_density.predicted_density_2015
compute_density["var_density"].astype(float).describe()
plt.hist(compute_density["var_density"])


### VALIDATION - 2015

#atlas of urban expansion

data_density = pd.read_excel("C:/Users/charl/OneDrive/Bureau/Urban sprawl and density/atlas_of_urban_expansion.xlsx")
comparison_aue = compute_density.merge(data_density, on = "City")

plt.scatter(comparison_aue.urban_extent_density_t3, comparison_aue.predicted_density_2015)
plt.xlim(0, 320)
plt.ylim(0, 320)
plt.xlabel("Atlas of urban expansion")
plt.ylabel("Our data")
xpoints = ypoints = plt.xlim()
plt.plot(xpoints, ypoints, linestyle='--', color='k', lw=3, scalex=False, scaley=False)
print(sc.stats.pearsonr(comparison_aue.urban_extent_density_t3, comparison_aue.predicted_density_2015_corrected_v2))

modeleReg=LinearRegression()
modeleReg.fit(np.array(comparison_aue.urban_extent_density_t3).reshape(-1, 1), comparison_aue.predicted_density_2015_corrected_v2)   
modeleReg.score(np.array(comparison_aue.urban_extent_density_t3).reshape(-1, 1), comparison_aue.predicted_density_2015_corrected_v2)   

#güneralp 2015

guneralp = pd.DataFrame(index = ['CPA', 'EEU', 'FSU', 'LAC', 'MNA', 'NAM', 'PAS', 'POECD', 'SAS', 'SSA', 'WEU', 'GLOBAL'], columns = ['2015_S50', '2035_S25', '2035_S50', '2035_S75'])
guneralp["2035_S25"] = [19.92, 26.13, 39.56, 64.88, 64.93, 13.05, 48.31, 44.22, 75.47, 55.49, 21.80, 49.69]
guneralp["2035_S50"] = [65.17, 32.29, 53.28, 87.97, 101.80, 16.72, 92.65, 71.93, 119.87, 102.84, 35.79, 86.42]
guneralp["2035_S75"] = [187.17, 43.03, 79.21, 120.88, 158.55, 21.64, 171.75, 108.40, 188.75, 157.33, 50.02, 149.50]
guneralp["2015_S50"] = [80.05, 44.66, 61.78, 98.71, 120.66, 20.11, 118.55, 96.53, 172.77, 90.79, 48.46, 104.93]

density_scenarios = compute_density.merge(list_city.loc[:, ['City', 'Country']], on = "City")
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

def my_agg_2015(x):
    names = {'weighted_density_2015': (x['predicted_density_2015'] * x['data_pop_2015']).sum()/x['data_pop_2015'].sum()}
    return pd.Series(names)

def my_agg_pop_2015(x):
    names = {'sum_pop_2015': (x['data_pop_2015']).sum()}
    return pd.Series(names)

aggregated_by_region_2015 = density_scenarios.loc[:, ['region_guneralp','predicted_density_2015', 'data_pop_2015']].groupby('region_guneralp').apply(my_agg_2015)
aggregated_pop_2015 = density_scenarios.loc[:, ['region_guneralp', 'data_pop_2015']].groupby('region_guneralp').apply(my_agg_pop_2015)
aggregated_by_region_2015 = aggregated_by_region_2015.merge(guneralp, left_index = True, right_index = True)
aggregated_by_region_2015 = aggregated_by_region_2015.merge(aggregated_pop_2015, left_index = True, right_index = True)

plt.figure(figsize = (15, 10))
plt.rcParams.update({'font.size': 25})
plt.scatter(aggregated_by_region_2015["2015_S50"], aggregated_by_region_2015.weighted_density_2015, s = aggregated_by_region_2015.sum_pop_2015 / 100000)
plt.xlabel("Güneralp - 2015 S50")
plt.ylabel("Our data")
plt.xlim(0, 350)
plt.ylim(0, 350)
xpoints = ypoints = plt.xlim()
plt.plot(xpoints, ypoints, linestyle='--', color='k', lw=3, scalex=False, scaley=False)
for i in range(len(aggregated_by_region_2015)):
    plt.annotate(aggregated_by_region_2015.index[i], (aggregated_by_region_2015["2015_S50"][i], aggregated_by_region_2015.weighted_density_2015[i]))
    
modeleReg=LinearRegression()
modeleReg.fit(np.array(aggregated_by_region_2015["2015_S50"]).reshape(-1, 1), aggregated_by_region_2015.weighted_density_2015)
modeleReg.score(np.array(aggregated_by_region_2015["2015_S50"]).reshape(-1, 1), aggregated_by_region_2015.weighted_density_2015)
    

#güneralp 2020

guneralp_2020_city = pd.read_excel("C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Data/guneralp_2020/Input_ranked-by-LocationName_WUP300K.xlsx").loc[:, ['UrbanAgg', 'Country', 'Region', 'PD2010']]
guneralp_2020_city.UrbanAgg[guneralp_2020_city.UrbanAgg == "Addis Ababa"] = "Addis_Ababa"
guneralp_2020_city.UrbanAgg[guneralp_2020_city.UrbanAgg == "Belo Horizonte"] = "Belo_Horizonte"
guneralp_2020_city.UrbanAgg[guneralp_2020_city.UrbanAgg == "Bruxelles-Brussel"] = "Brussels"
guneralp_2020_city.UrbanAgg[guneralp_2020_city.UrbanAgg == "Buenos Aires"] = "Buenos_Aires"
guneralp_2020_city.UrbanAgg[guneralp_2020_city.UrbanAgg == "Cluj-Napoca"] = "Cluj_Napoca"
guneralp_2020_city.UrbanAgg[guneralp_2020_city.UrbanAgg == "Fes"] = "Fez"
guneralp_2020_city.UrbanAgg[guneralp_2020_city.UrbanAgg == "Hong Kong"] = "Hong_Kong"
guneralp_2020_city.UrbanAgg[guneralp_2020_city.UrbanAgg == "Ji\'nan in Shandong"] = "Jinan"
guneralp_2020_city.UrbanAgg[guneralp_2020_city.UrbanAgg == "Krung Thep (Bangkok)"] = "Bangkok"
guneralp_2020_city.UrbanAgg[guneralp_2020_city.UrbanAgg == "Lisboa (Lisbon)"] = "Lisbon"
guneralp_2020_city.UrbanAgg[guneralp_2020_city.UrbanAgg == "Milano (Milan)"] = "Milan"
guneralp_2020_city.UrbanAgg[guneralp_2020_city.UrbanAgg == "Mumbai (Bombay)"] = "Mumbai"
guneralp_2020_city.UrbanAgg[guneralp_2020_city.UrbanAgg == "Munchen (Munich)"] = "Munich"
guneralp_2020_city.UrbanAgg[guneralp_2020_city.UrbanAgg == "Napoli(Naples)"] = "Naples"
guneralp_2020_city.UrbanAgg[guneralp_2020_city.UrbanAgg == "New York-Newark"] = "New_York"
guneralp_2020_city.UrbanAgg[guneralp_2020_city.UrbanAgg == "Roma (Rome)"] = "Rome"
guneralp_2020_city.UrbanAgg[guneralp_2020_city.UrbanAgg == "Sao Paulo"] = "Sao_Paulo"
guneralp_2020_city.UrbanAgg[guneralp_2020_city.UrbanAgg == "Warszawa (Warsaw)"] = "Warsaw"
guneralp_2020_city.UrbanAgg[guneralp_2020_city.UrbanAgg == "Washington in D.C."] = "Washington_DC"
guneralp_2020_city.UrbanAgg[guneralp_2020_city.UrbanAgg == "Bucuresti (Bucharest)"] = "Bucharest"
guneralp_2020_city.UrbanAgg[guneralp_2020_city.UrbanAgg == "Praha (Prague)"] = "Prague"
guneralp_2020_city = guneralp_2020_city.merge(compute_density.loc[:, ['City', 'predicted_density_2015', 'data_pop_2015']], left_on = 'UrbanAgg', right_on = 'City')
guneralp_2020_city["predicted_density_2015"] = guneralp_2020_city["predicted_density_2015"].astype(float)
guneralp_2020_city["data_pop_2015"] = guneralp_2020_city["data_pop_2015"].astype(float)
guneralp_2020_city = guneralp_2020_city.loc[guneralp_2020_city.PD2010 != ' ', :]

guneralp_2020_city.PD2010 = guneralp_2020_city.PD2010.astype(float) / 100

guneralp_2020_city = guneralp_2020_city.groupby("City").mean()
city_continent = pd.read_csv(path_data + "CityDatabases/cityDatabase_221NewCities.csv")
city_continent = city_continent.iloc[:, [0, 2]]
city_continent = city_continent.drop_duplicates(subset = "City")
city_continent = city_continent.sort_values('City')
guneralp_2020_city = guneralp_2020_city.merge(city_continent, on = "City", how = 'left')

plt.figure(figsize = (15, 10))
plt.rcParams.update({'font.size': 25})
color_tab = {'North_America':'red', 'Europe':'green', 'Asia':'blue', 'Oceania':'yellow', 'South_America': 'brown', 'Africa': 'orange'}
colors = list(guneralp_2020_city['Continent'].unique())
for i in range(0 , len(colors)):
    data = guneralp_2020_city.loc[guneralp_2020_city['Continent'] == colors[i]]
    plt.scatter(data.PD2010, data.predicted_density_2015, color=data.Continent.map(color_tab), label=colors[i], s = data.data_pop_2015 / 100000)
plt.xlim(0, 320)
plt.ylim(0, 320)
plt.xlabel("Guneralp 2020")
plt.ylabel("Our data")
xpoints = ypoints = plt.xlim()
plt.plot(xpoints, ypoints, linestyle='--', color='k', lw=3, scalex=False, scaley=False)
plt.legend()

print(sc.stats.pearsonr(guneralp_2020_city.PD2010, guneralp_2020_city.predicted_density_2015))


modeleReg=LinearRegression()
modeleReg.fit(np.array(guneralp_2020_city.PD2010).reshape(-1, 1), guneralp_2020_city.predicted_density_2015)   
modeleReg.score(np.array(guneralp_2020_city.PD2010).reshape(-1, 1), guneralp_2020_city.predicted_density_2015)   


### COMPARE WITH GUNERALP - 2035

density_scenarios["predicted_density_2035_corrected_v2"] = 0.01 * compute_density.data_pop_2035 / compute_density.predicted_land_cover_2035_corrected
density_scenarios["var_density"] = 100 * (density_scenarios.predicted_density_2035_corrected_v2 - density_scenarios.predicted_density_2015_corrected_v2) / density_scenarios.predicted_density_2015_corrected_v2

def my_agg_2035(x):
    names = {'weighted_density_2035': (x['predicted_density_2035'] * x['data_pop_2035']).sum()/x['data_pop_2035'].sum()}
    return pd.Series(names)

aggregated_by_region_2035 = density_scenarios.loc[:, ['region_guneralp','predicted_density_2035', 'data_pop_2035']].groupby('region_guneralp').apply(my_agg_2035)

def my_agg_var(x):
    names = {'weighted_var': (x['var_density'] * x['data_pop_2035']).sum()/x['data_pop_2035'].sum()}
    return pd.Series(names)

aggregated_by_region_var = density_scenarios.loc[:, ['region_guneralp','var_density', 'data_pop_2035']].groupby('region_guneralp').apply(my_agg_var)
#aggregated_by_region_var = density_scenarios.loc[:, ['region_guneralp','var_density']].groupby('region_guneralp').mean()
#aggregated_by_region_var.columns = ['weighted_var']

def my_agg_pop_2035(x):
    names = {'sum_pop_2035': (x['data_pop_2035']).sum()}
    return pd.Series(names)

aggregated_pop_2035 = density_scenarios.loc[:, ['region_guneralp', 'data_pop_2035']].groupby('region_guneralp').apply(my_agg_pop_2035)


df = aggregated_by_region_2035.merge(aggregated_by_region_var, left_index = True, right_index = True)
df = df.merge(guneralp, left_index = True, right_index = True)
df = df.merge(aggregated_pop_2035, left_index = True, right_index = True)

plt.figure(figsize = (15, 10))
plt.rcParams.update({'font.size': 25})
plt.scatter(df["2035_S50"], df.weighted_density_2035, s = df.sum_pop_2035 / 100000)
plt.xlabel("Güneralp - 2035 S50")
plt.ylabel("Our model")
plt.xlim(0, 350)
plt.ylim(0, 350)
xpoints = ypoints = plt.xlim()
plt.plot(xpoints, ypoints, linestyle='--', color='k', lw=3, scalex=False, scaley=False)
for i in range(len(df)):
    plt.annotate(df.index[i], (df["2035_S50"][i], df.weighted_density_2035[i]))
    
modeleReg=LinearRegression()
modeleReg.fit(np.array(df["2035_S50"]).reshape(-1, 1), df.weighted_density_2035)
modeleReg.score(np.array(df["2035_S50"]).reshape(-1, 1), df.weighted_density_2035)   
    
plt.figure(figsize = (15, 10))
plt.rcParams.update({'font.size': 25})
plt.scatter(100 * (df["2035_S50"] - df["2015_S50"]) / df["2015_S50"], df.weighted_var, s = df.sum_pop_2035 / 100000)
plt.xlabel("Güneralp - S50")
plt.ylabel("Our model")
plt.title("Density variations between 2015 and 2035 (%)")
plt.xlim(-60, 80)
plt.ylim(-60, 80)
xpoints = ypoints = plt.xlim()
plt.plot(xpoints, ypoints, linestyle='--', color='k', lw=3, scalex=False, scaley=False)
for i in range(len(df)):
    plt.annotate(df.index[i], ((100 * (df["2035_S50"] - df["2015_S50"]) / df["2015_S50"])[i], df.weighted_var[i]))
    
modeleReg=LinearRegression()
modeleReg.fit(np.array(100 * (df["2035_S50"] - df["2015_S50"]) / df["2015_S50"]).reshape(-1, 1), df.weighted_var)   
modeleReg.score(np.array(100 * (df["2035_S50"] - df["2015_S50"]) / df["2015_S50"]).reshape(-1, 1), df.weighted_var)   

scenarios_guneralp = density_scenarios.merge(guneralp, left_on = "region_guneralp", right_index = True)
plt.figure(figsize = (15, 10))
plt.rcParams.update({'font.size': 40})
plt.xlim(-100, 300)
plt.scatter(scenarios_guneralp["var_density"], scenarios_guneralp["region_guneralp"], c='lightgrey', s = scenarios_guneralp["predicted_population_2035_corrected"] / 10000)
plt.scatter((100 * (scenarios_guneralp["2035_S50"] - scenarios_guneralp["2015_S50"]) / scenarios_guneralp["2015_S50"]), scenarios_guneralp["region_guneralp"], c = 'red', s = 200)
plt.title("Density variations between 2015 and 2035 (%)")

### IMPACT D'UNE TAXE SUR LE PRIX DE L'ESSENCE DE 10%

compute_density = pd.read_excel("C:/Users/charl/OneDrive/Bureau/scenarios_densities_20211115.xlsx")
compute_density_tax = pd.read_excel("C:/Users/charl/OneDrive/Bureau/scenarios_densities_carbon_tax_20211116.xlsx")

sample_of_cities = pd.read_excel("C:/Users/charl/OneDrive/Bureau/sample_of_cities.xlsx").loc[:, ['City', 'final_sample']]
compute_density = compute_density.merge(sample_of_cities, on = "City")
compute_density = compute_density.loc[compute_density.final_sample == 1, :]
compute_density_tax = compute_density_tax.merge(sample_of_cities, on = "City")
compute_density_tax = compute_density_tax.loc[compute_density_tax.final_sample == 1, :]


compute_density["predicted_density_2035_corrected_v2"] = 0.01 * compute_density.data_pop_2035 / compute_density.predicted_land_cover_2035_corrected
compute_density["predicted_density_2015_corrected_v2"] = 0.01 * compute_density.data_pop_2015 / compute_density.predicted_land_cover_2015_corrected
compute_density["var_density"] = 100 * (compute_density.predicted_density_2035_corrected_v2 - compute_density.predicted_density_2015_corrected_v2) / compute_density.predicted_density_2015_corrected_v2

compute_density_tax["predicted_density_2035_corrected_v2"] = 0.01 * compute_density_tax.data_pop_2035 / compute_density_tax.predicted_land_cover_2035_corrected
compute_density_tax["predicted_density_2015_corrected_v2"] = 0.01 * compute_density_tax.data_pop_2015 / compute_density_tax.predicted_land_cover_2015_corrected
compute_density_tax["var_density"] = 100 * (compute_density_tax.predicted_density_2035_corrected_v2 - compute_density_tax.predicted_density_2015_corrected_v2) / compute_density_tax.predicted_density_2015_corrected_v2

compute_density = compute_density.loc[:, ['City', 'predicted_density_2035_corrected_v2', 'var_density', 'data_pop_2035']]
compute_density.columns = ['City', 'density_2035_BAU', 'var_density_BAU', 'data_pop_2035']

compute_density_tax = compute_density_tax.loc[:, ['City', 'predicted_density_2035_corrected_v2', 'var_density']]
compute_density_tax.columns = ['City', 'density_2035_tax', 'var_density_tax']

df = compute_density.merge(compute_density_tax, on = 'City')

city_continent = pd.read_csv(path_data + "CityDatabases/cityDatabase_221NewCities.csv")
city_continent = city_continent.iloc[:, [0, 2]]
city_continent = city_continent.drop_duplicates(subset = "City")
city_continent = city_continent.sort_values('City')
df = df.merge(city_continent, on = "City", how = 'left')
color_tab = {'North_America':'red', 'Europe':'green', 'Asia':'blue', 'Oceania':'yellow', 'South_America': 'brown', 'Africa': 'orange'}

plt.figure(figsize = (15, 10))
plt.rcParams.update({'font.size': 20})
colors = list(df['Continent'].unique())
for i in range(0 , len(colors)):
    data = df.loc[df['Continent'] == colors[i]]
    plt.scatter(data.density_2035_BAU, data.density_2035_tax, color=data.Continent.map(color_tab), label=colors[i], s = data.data_pop_2035 / 100000)
plt.xlim(0, 250)
plt.ylim(0, 250)
xpoints = ypoints = plt.xlim()
plt.plot(xpoints, ypoints, linestyle='--', color='k', lw=3, scalex=False, scaley=False)
plt.legend()

plt.figure(figsize = (15, 10))
plt.rcParams.update({'font.size': 20})
colors = list(df['Continent'].unique())
for i in range(0 , len(colors)):
    data = df.loc[df['Continent'] == colors[i]]
    plt.scatter(data.var_density_BAU, data.var_density_tax, color=data.Continent.map(color_tab), label=colors[i], s = data.data_pop_2035 / 100000)
plt.xlim(-100, 25)
plt.ylim(-100, 25)
xpoints = ypoints = plt.xlim()
plt.plot(xpoints, ypoints, linestyle='--', color='k', lw=3, scalex=False, scaley=False)
plt.legend()

plt.figure(figsize = (15, 10))
plt.rcParams.update({'font.size': 20})
plt.hist(100 * (df.density_2035_tax - df.density_2035_BAU) / df.density_2035_BAU)
plt.ylabel("Density change in 2035 due to the tax")
plt.xticks([0, 50, 100, 150, 200, 250], labels = ["+0%", "+50%", "+100%", "+150%", "+200%", '+250%'])

colors = list(df['Continent'].unique())
for i in range(0 , len(colors)):
    data = df.loc[df['Continent'] == colors[i]]
    plt.figure(figsize = (15, 10))
    plt.rcParams.update({'font.size': 20})
    plt.hist(100 * (data.density_2035_tax - data.density_2035_BAU) / data.density_2035_BAU)
    plt.ylabel("Density change in 2035 due to the tax")
    plt.xticks([0, 50, 100, 150, 200, 250], labels = ["+0%", "+50%", "+100%", "+150%", "+200%", '+250%'])
    plt.title(colors[i])
   
colors = ['blue', 'red', '#D94D1A']
colors1 = dict(color=colors[0])
colors2 = dict(color=colors[1])

fig, ax = plt.subplots(figsize = (15, 10))
bp1 = ax.boxplot([df.density_2035_BAU[df.Continent == 'Africa'], df.density_2035_BAU[df.Continent == 'Asia'], df.density_2035_BAU[df.Continent == 'Europe'], df.density_2035_BAU[df.Continent == 'North_America'], df.density_2035_BAU[df.Continent == 'Oceania'], df.density_2035_BAU[df.Continent == 'South_America']], positions = np.arange(1, 7), widths = 0.2, boxprops = colors1, medianprops=colors1, whiskerprops=colors1, capprops=colors1, flierprops = colors1)
bp2 = ax.boxplot([df.density_2035_tax[df.Continent == 'Africa'], df.density_2035_tax[df.Continent == 'Asia'], df.density_2035_tax[df.Continent == 'Europe'], df.density_2035_tax[df.Continent == 'North_America'], df.density_2035_tax[df.Continent == 'Oceania'], df.density_2035_tax[df.Continent == 'South_America']], positions = np.arange(1.3, 7.3), widths = 0.2, boxprops = colors2, medianprops=colors2, whiskerprops=colors2, capprops=colors2, flierprops = colors2)
plt.xticks(ticks = np.arange(1.15, 7.15), labels = ['Africa', 'Asia', 'Europe', 'North_America', 'Oceania', 'South_America'])
ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ['BAU', 'TAX'], loc='upper right')
