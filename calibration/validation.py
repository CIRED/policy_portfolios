# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 14:26:33 2021

@author: charl
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

def initialize_dict():
    a1 = {}
    a = {}
    a2 = {}
    a3 = {}
    a4 = {}
    a5 = {}
    a6 = {}
    a7 = {}
    a8 = {}
    a9 = {}
    a10 = {}
    a11 = {}
    a12 = {}
    a13 = {}
    a14 = {}
    a15 = {}
    a16 = {}
    a17 = {}
    a18 = {}
    a19 = {}
    a20 = {}
    a21 = {}
    a22 = {}
    a23 = {}
    a24 = {}
    a25 = {}
    a26 = {}
    a27 = {}
    a28 = {}
    a29 = {}
    a30 = {}
    a31 = {}
    a32 = {}
    a33 = {}
    a34 = {}
    a35 = {}
    a36 = {}
    a37 = {}
    a38 = {}
    a39 = {}
    a40 = {}
    a41 = {}
    a42 = {}
    a43 = {}
    a44 = {}
    a45 = {}
    a46 = {}
    a47 = {}
    return a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a30, a31, a32, a33, a34, a35, a36, a37, a38, a39, a40, a41, a42, a43, a44, a45, a46, a47

def compute_rae(data, simul, selected_cells):
    num = np.nansum(np.abs(simul[selected_cells] - data[selected_cells]))
    denom = np.nansum(np.abs(data[selected_cells] - np.nanmean(data[selected_cells])))
    return num/denom

def compute_corr(density, rent, size, simul_density, simul_rent, simul_size, selected_cells, selected_cells2):
    corr_density_scells2 = pearsonr(density[selected_cells2], simul_density[selected_cells2])
    corr_rent_scells2 = pearsonr(rent[selected_cells2], simul_rent[selected_cells2])
    corr_size_scells2 = pearsonr(size[selected_cells2], simul_size[selected_cells2])
    corr_density_scells1 = pearsonr(density[selected_cells], simul_density[selected_cells])
    corr_rent_scells1 = pearsonr(rent[selected_cells], simul_rent[selected_cells])
    corr_size_scells1 = pearsonr(size[selected_cells], simul_size[selected_cells])
    corr_density_scells0 = pearsonr(density[~np.isnan(density)], simul_density[~np.isnan(density)])
    corr_rent_scells0 = pearsonr(rent[rent.notnull()], simul_rent[rent.notnull()])
    corr_size_scells0 = pearsonr(size[size.notnull()], simul_size[size.notnull()])
    return corr_density_scells2, corr_rent_scells2, corr_size_scells2, corr_density_scells1, corr_rent_scells1, corr_size_scells1, corr_density_scells0, corr_rent_scells0, corr_size_scells0

def compute_r2(density, rent, size, simul_density, simul_rent, simul_size, selected_cells, selected_cells2):
    r2density2 = r2_score(density[selected_cells2], simul_density[selected_cells2])
    r2rent2 = r2_score(rent[selected_cells2], simul_rent[selected_cells2])
    r2size2 = r2_score(size[selected_cells2], simul_size[selected_cells2])
    r2density1 = r2_score(density[selected_cells], simul_density[selected_cells])
    r2rent1 = r2_score(rent[selected_cells], simul_rent[selected_cells])
    r2size1 = r2_score(size[selected_cells], simul_size[selected_cells])
    r2density0 = r2_score(density[~np.isnan(density)], simul_density[~np.isnan(density)])
    r2rent0 = r2_score(rent[rent.notnull()], simul_rent[rent.notnull()])
    r2size0 = r2_score(size[size.notnull()], simul_size[size.notnull()])
    return r2density2, r2rent2, r2size2, r2density1, r2rent1, r2size1, r2density0, r2rent0, r2size0

def compute_maes(density, rent, size, simul_density, simul_rent, simul_size, selected_cells, selected_cells2):
    mae_density2 = mean_absolute_error(density[selected_cells2], simul_density[selected_cells2])
    mae_rent2 = mean_absolute_error(rent[selected_cells2], simul_rent[selected_cells2])
    mae_size2 = mean_absolute_error(size[selected_cells2], simul_size[selected_cells2])
    mae_density1 = mean_absolute_error(density[selected_cells], simul_density[selected_cells])
    mae_rent1 = mean_absolute_error(rent[selected_cells], simul_rent[selected_cells])
    mae_size1 = mean_absolute_error(size[selected_cells], simul_size[selected_cells])
    mae_density0 = mean_absolute_error(density[~np.isnan(density)], simul_density[~np.isnan(density)])
    mae_rent0 = mean_absolute_error(rent[rent.notnull()], simul_rent[rent.notnull()])
    mae_size0 = mean_absolute_error(size[size.notnull()], simul_size[size.notnull()])
    return mae_density2, mae_rent2, mae_size2, mae_density1, mae_rent1, mae_size1, mae_density0, mae_rent0, mae_size0

def compute_raes(density, rent, size, simul_density, simul_rent, simul_size, selected_cells, selected_cells2):
    rae_density2 = compute_rae(density, simul_density, selected_cells2)
    rae_rent2 = compute_rae(rent, simul_rent, selected_cells2)
    rae_size2 = compute_rae(size, simul_size, selected_cells2)
    rae_density1 = compute_rae(density, simul_density, selected_cells)
    rae_rent1 = compute_rae(rent, simul_rent, selected_cells)
    rae_size1 = compute_rae(size, simul_size, selected_cells)
    rae_density0 = compute_rae(density, simul_density, (~np.isnan(density)))
    rae_rent0 = compute_rae(rent, simul_rent, (rent.notnull()))
    rae_size0 = compute_rae(size, simul_size, (size.notnull()))
    return rae_density2, rae_rent2, rae_size2, rae_density1, rae_rent1, rae_size1, rae_density0, rae_rent0, rae_size0

def compute_modal_shares(density, mode_choice, population):
    share_transit = np.sum(density[mode_choice == 1]) / population
    share_car = np.sum(density[mode_choice == 0]) / population
    share_walk = np.sum(density[mode_choice == 2]) / population
    return share_transit, share_car, share_walk

def export_charts(distance_cbd, simul_rent, rent, simul_density, density, simul_size, size, path_calibration, city):
    plt.scatter(distance_cbd, simul_rent, label = "Simul", s=2, c="darkred")
    plt.scatter(distance_cbd, rent, label = "Data", s=2, c="grey")
    plt.legend(markerscale=6,
               scatterpoints=1, fontsize=10)
    plt.title("Rent calibration")
    plt.savefig(path_calibration + city + "_rent.png")
    plt.close()

    plt.scatter(distance_cbd, simul_density, label = "Simul", s=2, c="darkred")
    plt.scatter(distance_cbd, density, label = "Data", s=2, c="grey")
    plt.legend(markerscale=6,
               scatterpoints=1, fontsize=10)
    plt.title("Density calibration")
    plt.savefig(path_calibration + city + "_density.png")
    plt.close()

    plt.scatter(distance_cbd, simul_size, label = "Simul", s=2, c="darkred")
    plt.scatter(distance_cbd, size, label = "Data", s=2, c="grey")
    plt.legend(markerscale=6,
               scatterpoints=1, fontsize=10)
    plt.title("Dwelling size calibration")
    plt.savefig(path_calibration + city + "_size.png")
    plt.close()
    
def save_outputs_validation(path_calibration, d_beta, d_b, d_kappa, d_Ro, d_share_transit, d_share_car, d_share_walking, d_emissions_per_capita, d_utility, d_income, d_selected_cells, d_corr_density0, d_corr_rent0, d_corr_size0, d_corr_density1, d_corr_rent1, d_corr_size1, d_corr_density2, d_corr_rent2, d_corr_size2, r2_density0, r2_rent0, r2_size0, r2_density1, r2_rent1, r2_size1, r2_density2, r2_rent2, r2_size2, mae_density0, mae_rent0, mae_size0, mae_density1, mae_rent1, mae_size1, mae_density2, mae_rent2, mae_size2, rae_density0, rae_rent0, rae_size0, rae_density1, rae_rent1, rae_size1, rae_density2, rae_rent2, rae_size2):
    np.save(path_calibration + 'beta.npy', d_beta)
    np.save(path_calibration + 'b.npy', d_b)
    np.save(path_calibration + 'kappa.npy', d_kappa)
    np.save(path_calibration + 'Ro.npy', d_Ro)

    np.save(path_calibration + 'd_share_car.npy', d_share_car)
    np.save(path_calibration + 'd_share_walking.npy', d_share_walking)
    np.save(path_calibration + 'd_share_transit.npy', d_share_transit)
    np.save(path_calibration + 'd_emissions_per_capita.npy', d_emissions_per_capita)
    np.save(path_calibration + 'd_utility.npy', d_utility)
    np.save(path_calibration + 'd_selected_cells.npy', d_selected_cells)
    np.save(path_calibration + 'd_income.npy', d_income)

    np.save(path_calibration + 'd_corr_density_scells2', d_corr_density2)
    np.save(path_calibration + 'd_corr_rent_scells2', d_corr_rent2)
    np.save(path_calibration + 'd_corr_size_scells2', d_corr_size2)
    np.save(path_calibration + 'd_corr_density_scells1', d_corr_density1)
    np.save(path_calibration + 'd_corr_rent_scells1', d_corr_rent1)
    np.save(path_calibration + 'd_corr_size_scells1', d_corr_size1)
    np.save(path_calibration + 'd_corr_density_scells0', d_corr_density0)
    np.save(path_calibration + 'd_corr_rent_scells0', d_corr_rent0)
    np.save(path_calibration + 'd_corr_size_scells0', d_corr_size0)

    np.save(path_calibration + 'r2density_scells2', r2_density2)
    np.save(path_calibration + 'r2density_scells1', r2_density1)
    np.save(path_calibration + 'r2density_scells0', r2_density0)
    np.save(path_calibration + 'r2rent_scells2', r2_rent2)
    np.save(path_calibration + 'r2rent_scells1', r2_rent1)
    np.save(path_calibration + 'r2rent_scells0', r2_rent0)
    np.save(path_calibration + 'r2size_scells2', r2_size2)
    np.save(path_calibration + 'r2size_scells1', r2_size1)
    np.save(path_calibration + 'r2size_scells0', r2_size0)

    np.save(path_calibration + 'mae_density_scells2', mae_density2)
    np.save(path_calibration + 'mae_rent_scells2', mae_rent2)
    np.save(path_calibration + 'mae_size_scells2', mae_size2)
    np.save(path_calibration + 'mae_density_scells1', mae_density1)
    np.save(path_calibration + 'mae_rent_scells1', mae_rent1)
    np.save(path_calibration + 'mae_size_scells1', mae_size1)
    np.save(path_calibration + 'mae_density_scells0', mae_density0)
    np.save(path_calibration + 'mae_rent_scells0', mae_rent0)
    np.save(path_calibration + 'mae_size_scells0', mae_size0)

    np.save(path_calibration + 'rae_density_scells2', rae_density2)
    np.save(path_calibration + 'rae_rent_scells2', rae_rent2)
    np.save(path_calibration + 'rae_size_scells2', rae_size2)
    np.save(path_calibration + 'rae_density_scells1', rae_density1)
    np.save(path_calibration + 'rae_rent_scells1', rae_rent1)
    np.save(path_calibration + 'rae_size_scells1', rae_size1)
    np.save(path_calibration + 'rae_density_scells0', rae_density0)
    np.save(path_calibration + 'rae_rent_scells0', rae_rent0)
    np.save(path_calibration + 'rae_size_scells0', rae_size0)
    
def import_modal_shares_wiki(path_data):
    modal_shares = pd.read_excel(path_data + "modal_shares.xlsx", sheet_name = "Feuille3", header = 0) 
    for i in range(len(modal_shares)):
        modal_shares.iloc[i, 0] = re.sub(r'.*?\xa0', '', modal_shares.iloc[i, 0])
    modal_shares.cycling[modal_shares.city == "Brussels"] = 0.025
    modal_shares.cycling[modal_shares.city == "Hong Kong"] = 0.005
    modal_shares.cycling[modal_shares.city == "Madrid"] = 0.005
    modal_shares.cycling[modal_shares.city == "Prague"] = 0.004
    modal_shares.cycling[modal_shares.city == "Jakarta"] = 0.002
    modal_shares["private car"][modal_shares.city == "Jakarta"] = 0.78
    modal_shares.cycling[modal_shares.city == "Calgary"] = 0.015
    modal_shares.walking[modal_shares.city == "Calgary"] = 0.047
    modal_shares["private car"][modal_shares.city == "Calgary"] = 0.794
    modal_shares["public transport"][modal_shares.city == "Calgary"] = 0.144
    modal_shares.cycling[modal_shares.city == "Edmonton"] = 0.01
    modal_shares.walking[modal_shares.city == "Edmonton"] = 0.037
    modal_shares["private car"][modal_shares.city == "Edmonton"] = 0.84
    modal_shares["public transport"][modal_shares.city == "Edmonton"] = 0.113
    #modal_shares["transit_share"] = (modal_shares["walking"] + modal_shares["public transport"]) / (modal_shares["walking"] + modal_shares["public transport"] + modal_shares["private car"])
    
    modal_shares["city"][modal_shares.city == "Hong Kong"] = 'Hong_Kong'
    modal_shares["city"][modal_shares.city == "Los Angeles"] = 'Los_Angeles'
    modal_shares["city"][modal_shares.city == "New York City"] = 'New_York'
    modal_shares["city"][modal_shares.city == "Rio de Janeiro"] = 'Rio_de_Janeiro'
    modal_shares["city"][modal_shares.city == "San Diego"] = 'San_Diego'
    modal_shares["city"][modal_shares.city == "San Francisco"] = 'San_Fransisco'
    modal_shares["city"][modal_shares.city == "Washington, D.C."] = 'Washington_DC'
    modal_shares["city"][modal_shares.city == "Frankfurt"] = 'Frankfurt_am_Main'
    modal_shares["city"][modal_shares.city == "The Hague"] = 'The_Hague'
    modal_shares["city"][modal_shares.city == "Zürich"] = 'Zurich'
    modal_shares["city"][modal_shares.city == "Córdoba"] = 'Cordoba'
    modal_shares["city"][modal_shares.city == "São Paulo"] = 'Sao_Paulo'
    modal_shares["city"][modal_shares.city == "Málaga"] = 'Malaga'
    modal_shares["city"][modal_shares.city == "Gent"] = 'Ghent'
    modal_shares["city"][modal_shares.city == "Malmö"] = 'Malmo'
    
    
    return modal_shares