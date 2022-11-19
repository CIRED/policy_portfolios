# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 15:02:05 2022

@author: charl
"""

from osgeo import gdal, ogr

#Chen
scenario = ['SSP1', 'SSP2', 'SSP3', 'SSP4', 'SSP5']
year = ['2020', '2050', '2100']
chen = pd.DataFrame(columns = ['chen_2020', 'chen_2050', 'chen_2100'], index = scenario) 

for i in year:
    for scen in scenario:
        dataset = gdal.Open(r'C:/Users/charl/OneDrive/Bureau/Urban sprawl/chen_2020/' + scen + '/global_' + scen + '_' + i + '.tif')
        band = dataset.GetRasterBand(1)
        arr = band.ReadAsArray()
        chen["chen_" + i].loc[scen] = np.nansum(arr == 2) #133145613 #133289615 #133458857

chen.to_excel('C:/Users/charl/OneDrive/Bureau/Urban sprawl/chen_2020/global_sprawl.xlsx')

chen = pd.read_excel('C:/Users/charl/OneDrive/Bureau/Urban sprawl/chen_2020/global_sprawl.xlsx')

#huang
scenario = ['ssp1', 'ssp2', 'ssp3', 'ssp4', 'ssp5']
huang = pd.DataFrame(columns = ['huang_2050'], index = scenario) 

for i in year:
    for scen in scenario:
        dataset = gdal.Open(r'C:/Users/charl/OneDrive/Bureau/Urban sprawl/huang_2019/urban-' + scen + '.tif')
        band = dataset.GetRasterBand(1)
        arr = band.ReadAsArray()
        #chen["chen_" + i].loc[scen] = np.nansum(arr == 2) #133145613 #133289615 #133458857

#chen.to_excel('C:/Users/charl/OneDrive/Bureau/Urban sprawl/chen_2020/global_sprawl.xlsx')