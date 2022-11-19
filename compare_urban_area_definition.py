# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 17:12:55 2022

@author: charl
"""

comparison = pd.DataFrame(index = np.delete(np.unique(list_city.City), 153), columns = ["data", "150t", "1000t", "150c", "1000c"])




for city in np.delete(np.unique(list_city.City), 153):
#for city in ["Turku"]:
    
    (country, proj, density, rents_and_size, land_use, land_cover_ESACCI, driving, transit, grille, 
     centre, distance_cbd, conversion_rate) = import_data(list_city, path_data, city, path_folder)        
    
    #Density, rents and dwelling sizes
    density = density.loc[:,density.columns.str.startswith("density")]
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
    #informal_housing_city = informal_housing.informal_housing[informal_housing.City == city]
    #rent = (rentemp_capita_ppp.city == city].squeeze() == "WB":
    #    income = income * 1.33
    agricultural_rent = import_agricultural_rent(path_folder, country)
    population = np.nansum(density)
    region = pd.read_excel(path_folder + "city_country_region.xlsx").region[pd.read_excel(path_folder + "city_country_region.xlsx").city == city].squeeze()
    income = pd.read_excel(path_folder + "income.xlsx").income[pd.read_excel(path_folder + "income.xlsx").city == city].squeeze()
    coeff_land = (land_use.OpenedToUrb / land_use.TotalArea) * COEFF_URB
    density_to_cover = convert_density_to_urban_footprint(city, path_data, list_city)
    agricultural_rent_2015 = copy.deepcopy(agricultural_rent)
    
    urban_area_data=sum(land_cover_ESACCI.ESACCI190) / 1000000
    density_1000 = sum(density > 1000)
    ua_reg = sum(predict_urbanized_area(density_to_cover, density))
    
    comparison.loc[comparison.index == city, "data"] = urban_area_data
    comparison.loc[comparison.index == city, "1000t"] = density_1000
    comparison.loc[comparison.index == city, "reg"] = ua_reg
    
    plt.scatter(grille.XCOORD, grille.YCOORD, c = land_cover_ESACCI.ESACCI190 / 1000000)
    plt.colorbar()
    plt.show()
    plt.title("Data")
    plt.clf()

    plt.scatter(grille.XCOORD, grille.YCOORD, c = density > 1000)
    plt.colorbar()
    plt.show()
    plt.title("Threshold")
    plt.clf()

    plt.scatter(grille.XCOORD, grille.YCOORD, c = predict_urbanized_area(density_to_cover, density))
    plt.colorbar()
    plt.show()
    plt.title("Reg")
    plt.clf()
    
plt.scatter(comparison.data, comparison["1000t"], s = 8, c = "red", edgecolors = "black", linewidth=1)
plt.xlim(0, 7000)
plt.ylim(0, 7000)
plt.plot([0, 7000], [0, 7000], linewidth = 0.5, c = "black")

plt.scatter(comparison.data, comparison["1000t"], s = 8, c = "red", edgecolors = "black", linewidth=1)
plt.xlim(0, 1000)
plt.ylim(0, 1000)
plt.plot([0, 1000], [0, 1000], linewidth = 0.5, c = "black")

plt.scatter(comparison.data, comparison["reg"], s = 8, c = "red", edgecolors = "black", linewidth=1)
plt.xlim(0, 7000)
plt.ylim(0, 7000)
plt.plot([0, 7000], [0, 7000], linewidth = 0.5, c = "black")

plt.scatter(comparison.data, comparison["reg"], s = 8, c = "red", edgecolors = "black", linewidth=1)
plt.xlim(0, 1000)
plt.ylim(0, 1000)
plt.plot([0, 1000], [0, 1000], linewidth = 0.5, c = "black")

plt.scatter(grille.XCOORD, grille.YCOORD, c = land_cover_ESACCI.ESACCI190 / 1000000)
plt.colorbar()

plt.scatter(grille.XCOORD, grille.YCOORD, c = density > 1000)
plt.colorbar()

plt.scatter(grille.XCOORD, grille.YCOORD, c = predict_urbanized_area(density_to_cover, density))
plt.colorbar()
    

from scipy.stats import pearsonr
corr, _ = pearsonr(comparison.data, comparison["1000t"])
print('Pearsons correlation: %.3f' % corr)

corr, _ = pearsonr(comparison.data, comparison["reg"])
print('Pearsons correlation: %.3f' % corr)