import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def calibration(channels_pique):
    energies = np.array([122, 511, 662])
    a, b = np.polyfit(channels_pique, energies, 1)
    return a, b

def mean_time(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    line_with_durations = lines[lines.index("$MEAS_TIM:\n") + 1]
    duration1, duration2 = map(int, line_with_durations.split())
    return duration1

def gaussian(x, a, mu, sigma):
    return a * np.exp(-(x - mu)**2 / (2 * sigma**2))


def fit_gaussien(data_values, nombre_de_cannaux):
    # Estimer les paramètres initiaux pour l'ajustement
    a_init = np.max(data_values)
    mu_init = nombre_de_cannaux[np.argmax(data_values)]
    sigma_init = np.std(data_values)

    # Ajustement de la courbe
    popt, pcov = curve_fit(gaussian, nombre_de_cannaux, data_values, p0=[a_init, mu_init, sigma_init], maxfev=5000)
    return popt

def spectre(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Supprimer les 12 premières lignes et les 16 dernières lignes
    data_lines = lines[12:-18]

    # Convertir les chaînes de caractères en entiers et stocker les valeurs dans une liste
    data_values = [int(line.strip()) for line in data_lines]

    # génère un tableau d'entiers séquentiels allant de 0 jusqu'à la longueur de data_values moins un
    channels = np.arange(len(data_values))

    return data_values, channels

def Roy_finder_Co57(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    line_15_from_end = lines[-15].strip()
    lower_index, upper_index = map(int, line_15_from_end.split())
    return lower_index, upper_index

def Roy_finder_Na22(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    line_14_from_end = lines[-14].strip()
    lower_index, upper_index = map(int, line_14_from_end.split())
    return lower_index, upper_index

def Roy_finder_Cs137(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    line_13_from_end = lines[-13].strip()
    lower_index, upper_index = map(int, line_13_from_end.split())
    return lower_index, upper_index

file_path = r'C:\Users\طه\Desktop\Rapport final a remettre\Taha et Joan reprise electron\Etalonnage-final.Spe'
Nb_de_coup, cannaux = spectre(file_path)
#Pour le Co57, on a:
Nb_de_coup_Co57  = Nb_de_coup[Roy_finder_Co57(file_path)[0]:Roy_finder_Co57(file_path)[1]]
cannaux_Co57 = cannaux[Roy_finder_Co57(file_path)[0]:Roy_finder_Co57(file_path)[1]]
canal_max_Co57 = fit_gaussien(Nb_de_coup_Co57, cannaux_Co57)[1]
Incertitude_Co57 = fit_gaussien(Nb_de_coup_Co57, cannaux_Co57)[2]/2.335
#Pour le Na22, on a:
Nb_de_coup_Na22  = Nb_de_coup[Roy_finder_Na22(file_path)[0]:Roy_finder_Na22(file_path)[1]]
cannaux_Na22 = cannaux[Roy_finder_Na22(file_path)[0]:Roy_finder_Na22(file_path)[1]]
canal_max_Na22 = fit_gaussien(Nb_de_coup_Na22, cannaux_Na22)[1]
Incertitude_Na22 = fit_gaussien(Nb_de_coup_Na22, cannaux_Na22)[2]/2.335
#Pour le Cs137, on a:
Nb_de_coup_Cs137 = Nb_de_coup[Roy_finder_Cs137(file_path)[0]:Roy_finder_Cs137(file_path)[1]]
cannaux_Cs137 = cannaux[Roy_finder_Cs137(file_path)[0]:Roy_finder_Cs137(file_path)[1]]
canal_max_Cs137 = fit_gaussien(Nb_de_coup_Cs137, cannaux_Cs137)[1]
Incertitude_Cs137 = fit_gaussien(Nb_de_coup_Cs137, cannaux_Cs137)[2]/2.335

#channels_etallonage = np.array([636, 2463, 3136])
#Etallonage pour photons
#channels_etallonage_debut = np.array([616, 2263, 2868])/incertitude = [38.73, 46.78, 52.17]
#channels_etallonage_fin = np.array([616, 2265, 2871])/incertitude = [40.84, 49.16, 47.30]
#Etallonage pour electrons
#channels_etallonage_debut = np.array([636, 2463, 3136])/incertitude = [39.20, 55.92, 52.52]
#channels_etallonage_fin = np.array([688, 2664, 3384])/incertitude = [38.97, 63.33, 62.06]

print('Cannal du pic Co57:', canal_max_Co57)
print('Cannal du pic Na22:', canal_max_Na22)
print('Cannal du pic Cs1377:', canal_max_Cs137)
print('Incertitude Cannal du pic Co57:', Incertitude_Co57)
print('Incertitude Cannal du pic Na22:', Incertitude_Na22)
print('Incertitude Cannal du pic Cs1377:', Incertitude_Cs137)