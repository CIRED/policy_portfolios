# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 15:48:14 2022

@author: charl
"""

plt.plot(save_urbanized_area / save_urbanized_area[0], label = 'UA')
plt.plot(save_population / save_population[0], label = 'pop')
plt.plot(save_income / save_income[0], label = 'inc')
plt.legend()

for i in np.arange(1, 20):
    print(((save_urbanized_area[i]/save_urbanized_area[i-1]) - 1) * 100)
    
path_ssp1 = "C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/ssp1_constpop/"
path_ssp2 = "C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/ssp2_constpop/"
path_ssp3 = "C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/ssp3_constpop/"
path_ssp4 = "C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/ssp4_constpop/"
path_ssp5 = "C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/ssp5_constpop/"

path_ssp1 = "C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/ssp1_20220627/"
path_ssp2 = "C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/ssp2_20220627/"
path_ssp3 = "C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/ssp3_20220627/"
path_ssp4 = "C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/ssp4_20220627/"
path_ssp5 = "C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/ssp5_20220627/"

#for city in ["Paris", "Atlanta", "Bangalore", "Bangkok", "Beijing", "Belo_Horizonte"]:
#for city in np.delete(np.unique(list_city.City), 153):
for city in list(sample_of_cities.City[sample_of_cities.final_sample == 1]):
    
    
    
    save_density_ssp1 = np.load(path_ssp1 + city + "_density.npy")
    save_density_ssp2 = np.load(path_ssp2 + city + "_density.npy")
    save_density_ssp3 = np.load(path_ssp3 + city + "_density.npy")
    save_density_ssp4 = np.load(path_ssp4 + city + "_density.npy")
    save_density_ssp5 = np.load(path_ssp5 + city + "_density.npy")
       
    urba_ssp1 = np.nansum(predict_urbanized_area2(density_to_cover, save_density_ssp1), 1)
    urba_ssp2 = np.nansum(predict_urbanized_area2(density_to_cover, save_density_ssp2), 1)
    urba_ssp3 = np.nansum(predict_urbanized_area2(density_to_cover, save_density_ssp3), 1)
    urba_ssp4 = np.nansum(predict_urbanized_area2(density_to_cover, save_density_ssp4), 1)
    urba_ssp5 = np.nansum(predict_urbanized_area2(density_to_cover, save_density_ssp5), 1)

    plt.figure()
    plt.plot(urba_ssp1, label = "SSP1")
    plt.plot(urba_ssp2, label = "SSP2")
    plt.plot(urba_ssp3, label = "SSP3")
    plt.plot(urba_ssp4, label = "SSP4")
    plt.plot(urba_ssp5, label = "SSP5")
    plt.legend()
    plt.title(city + "- urban area")
    plt.show()

    plt.clf()
    
    save_pop_ssp1 = np.load(path_ssp1 + city + "_pop.npy")
    save_pop_ssp2 = np.load(path_ssp2 + city + "_pop.npy")
    save_pop_ssp3 = np.load(path_ssp3 + city + "_pop.npy")
    save_pop_ssp4 = np.load(path_ssp4 + city + "_pop.npy")
    save_pop_ssp5 = np.load(path_ssp5 + city + "_pop.npy")
       
    plt.figure()
    plt.plot(save_pop_ssp1, label = "SSP1")
    plt.plot(save_pop_ssp2, label = "SSP2")
    plt.plot(save_pop_ssp3, label = "SSP3")
    plt.plot(save_pop_ssp4, label = "SSP4")
    plt.plot(save_pop_ssp5, label = "SSP5")
    plt.legend()
    plt.title(city+ "- population")
    plt.show()

    plt.clf()
    
    save_inc_ssp1 = np.load(path_ssp1 + city + "_income.npy")
    save_inc_ssp2 = np.load(path_ssp2 + city + "_income.npy")
    save_inc_ssp3 = np.load(path_ssp3 + city + "_income.npy")
    save_inc_ssp4 = np.load(path_ssp4 + city + "_income.npy")
    save_inc_ssp5 = np.load(path_ssp5 + city + "_income.npy")
       
    plt.figure()
    plt.plot(save_inc_ssp1, label = "SSP1")
    plt.plot(save_inc_ssp2, label = "SSP2")
    plt.plot(save_inc_ssp3, label = "SSP3")
    plt.plot(save_inc_ssp4, label = "SSP4")
    plt.plot(save_inc_ssp5, label = "SSP5")
    plt.legend()
    plt.title(city+ "- income")
    plt.show()

    plt.clf()
    
    
plt.plot(inc_ssp3)
plt.plot(inc_ssp5)

plt.scatter(grille.XCOORD, grille.YCOORD, c = 100*(density_ssp3[15] - density_ssp1[15]) / density_ssp1[15])
plt.colorbar()

plt.scatter(grille.XCOORD, grille.YCOORD, c = predict_urbanized_area2(density_to_cover, density_ssp3[15]) * np.ones(len(density_ssp3[15])) - np.ones(len(density_ssp3[15])) * predict_urbanized_area2(density_to_cover, density_ssp1[15]))
plt.colorbar()

plt.scatter(grille.XCOORD, grille.YCOORD, c = predict_urbanized_area2(density_to_cover, density_ssp3[15]))
plt.colorbar()

plt.scatter(grille.XCOORD, grille.YCOORD, c = predict_urbanized_area2(density_to_cover, density_ssp1[15]))
plt.colorbar()



plt.scatter(distance_cbd, density_ssp3[85])
plt.scatter(distance_cbd, density_ssp5[85])

plt.plot(urba_ssp3, label = "SSP3")
plt.plot(urba_ssp5, label = "SSP5")

plt.plot(pop_ssp3, label = "SSP3")
plt.plot(pop_ssp5, label = "SSP5")

plt.plot(inc_ssp3, label = "SSP3")
plt.plot(inc_ssp5, label = "SSP5")

path_ssp2_croissant = "C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/ssp2_tcost_croissant/"
path_ssp2_constant = "C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/ssp2_tcost_ctt/"

for city in list(sample_of_cities.City[sample_of_cities.final_sample == 1]):
    
    
    
    save_density_ssp2_croissant = np.load(path_ssp2_croissant + city + "_density.npy")
    save_density_ssp2_constant = np.load(path_ssp2_constant + city + "_density.npy")
       
    urba_ssp2_croissant = np.nansum(predict_urbanized_area2(density_to_cover, save_density_ssp2_croissant), 1)
    urba_ssp2_constant = np.nansum(predict_urbanized_area2(density_to_cover, save_density_ssp2_constant), 1)
    
    urba_ssp2_croissant =  np.load(path_ssp2_croissant + city + "_ua.npy")
    urba_ssp2_constant = np.load(path_ssp2_constant + city + "_ua.npy")
  
    plt.figure()
    plt.plot(urba_ssp2_croissant[0:20], label = "Croissant")
    plt.plot(urba_ssp2_constant[0:20], label = "Constant")
    plt.title(city + "- urban area")
    plt.legend()
    plt.show()

    plt.clf()
