import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def Efficacité_détecteur(E):
    a, b = np.polyfit(np.log(np.array([200, 400, 600])), np.log(np.array([0.5, 0.28, 0.17])), 1)

    return np.exp(np.ln(E) * a + b)


def calibration(channels_pique):
    energies = np.array([122, 511, 662])
    a, b = np.polyfit(channels_pique, energies, 1)
    return a, b


def channel_to_energy(channels_pique, channel):
    calibration_param = calibration(channels_pique)
    return calibration_param[0]*channel + calibration_param[1]


def gaussian(x, a, mu, sigma):
    return a * np.exp(-(x - mu)**2 / (2 * sigma**2))


def spectre(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Supprimer les 12 premières lignes et les 16 dernières lignes
    data_lines = lines[12:-16]

    # Convertir les chaînes de caractères en entiers et stocker les valeurs dans une liste
    data_values = [int(line.strip()) for line in data_lines]

    # génère un tableau d'entiers séquentiels allant de 0 jusqu'à la longueur de data_values moins un
    channels = np.arange(len(data_values))

    return data_values, channels


def fit_gaussien(data_values, nombre_de_cannaux):
    # Estimer les paramètres initiaux pour l'ajustement
    a_init = np.max(data_values)
    mu_init = nombre_de_cannaux[np.argmax(data_values)]
    sigma_init = np.std(data_values)

    # Ajustement de la courbe
    popt, pcov = curve_fit(gaussian, nombre_de_cannaux, data_values, p0=[a_init, mu_init, sigma_init], maxfev=5000)
    return popt


def take_off_noises(coef, file_path, data_values, channels, popt):
    # Création d'un tableau avec les valeurs ajustées pour chaque canal
    fitted_counts = gaussian(channels, *popt)

    lower_channel, upper_channel = Roy_finder(file_path)

    # Déterminer le seuil
    threshold = popt[0] / coef

    # Tolérance pour comparer les valeurs
    tolerance = 15

    # Trouver les canaux dont le nombre de coups ajusté est exactement égal au seuil
    selected_channels = np.where(np.isclose(fitted_counts, threshold, atol=tolerance))

    # Convertir le tableau NumPy en liste d'indices
    matching_channels = selected_channels[0].tolist()

    # Utiliser les indices pour accéder aux éléments de data_value
    Values_for_linear_fit = [data_values[index] for index in matching_channels]

    # Effectuer un ajustement linéaire
    slope, intercept = np.polyfit(matching_channels, Values_for_linear_fit, 1)

    # Générer l'ensemble des canaux
    all_channels = np.arange(4095)

    # Calculer les valeurs estimées des coups pour chaque canal en utilisant le modèle linéaire ajusté
    background_noise = slope * all_channels + intercept

    # S'assurer que les valeurs estimées des coups sont positives (ou au moins égales à zéro)
    background_noise = np.maximum(0, background_noise)

    # Rendre tous les elements du bruit des entier:
    background_noise = [int(i) for i in background_noise]

    background_noise = background_noise[lower_channel:upper_channel]
    
    # Enlever le bruit de fond, et prendre juste la les valeurs d'energies qui delimitent les canaux qui sont sous le pic
    new_data = np.array(data_values) - background_noise
    new_data_zeroed = np.array([new_data[i] if matching_channels[0] <= i <= matching_channels[-1] else 0 for i in range(len(new_data))])
    
    # S'assurer que tous les valeurs sont positifs
    new_data_zeroed1 = np.clip(new_data_zeroed, 0, None)
    return new_data_zeroed1


def Roy_finder(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    line_13_from_end = lines[-13].strip()
    lower_index, upper_index = map(int, line_13_from_end.split())
    return lower_index, upper_index


def energie_pic(file_path, channels_pique, coef=8):
    data_values, channels = spectre(file_path)

    lower_channel, upper_channel = Roy_finder(file_path)

    data_values_2 = data_values[lower_channel:upper_channel]

    channels_2 = channels[lower_channel:upper_channel]

    popt = fit_gaussien(data_values_2, channels_2)

    data_pic_without_noise = take_off_noises(coef, file_path, data_values_2, channels_2, popt)

    popt1 = fit_gaussien(data_pic_without_noise, channels_2)

    calibration_params = calibration(channels_pique)

    energy_peak_fit1 = channel_to_energy(channels_pique, popt[1])
    energy_uncertainty_fit1 = popt1[2] *  calibration_params[0]

    # Arrondir les résultats avec 2 chiffres significatifs
    energy_peak = round(energy_peak_fit1, 2)
    energy_uncertainty = round(energy_uncertainty_fit1, 2)

    return energy_peak, energy_uncertainty


def nombre_de_coup_sous_le_pic(file_path, coef2=6):
    data_values, channels = spectre(file_path)

    lower_channel, upper_channel = Roy_finder(file_path)

    data_values_2 = data_values[lower_channel:upper_channel]

    channels_2 = channels[lower_channel:upper_channel]

    popt = fit_gaussien(data_values_2, channels_2)

    # Création d'un tableau avec les valeurs ajustées pour chaque canal
    fitted_counts = gaussian(channels_2, *popt)

    # Déterminer le seuil
    threshold = popt[0] / coef2

    # Tolérance pour comparer les valeurs
    tolerance = 15

    # Trouver les canaux dont le nombre de coups ajusté est exactement égal au seuil
    selected_channels = np.where(np.isclose(fitted_counts, threshold, atol=tolerance))

    # Convertir le tableau NumPy en liste d'indices
    matching_channels = selected_channels[0].tolist()

    # le nombre total de coups sur l'ensemble des canaux sous le pic
    total_counts = np.sum(data_values[matching_channels[0]:matching_channels[-1]])
    incertitude = np.sqrt(total_counts)

    mean_duration = mean_time(file_path)
    air_net_per_sec = total_counts/mean_duration

    # Arrondir les résultats avec 2 chiffres significatifs
    air_net_per_sec_rounded = round(air_net_per_sec, 2)
    incertitude_rounded = round(incertitude / mean_duration, 2)

    return air_net_per_sec_rounded, incertitude_rounded


def mean_time(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    line_with_durations = lines[lines.index("$MEAS_TIM:\n") + 1]
    duration1, duration2 = map(int, line_with_durations.split())
    return duration1


def plot(file_path, channels_pique, coef=10):
    data_values, channels = spectre(file_path)

    lower_channel, upper_channel = Roy_finder(file_path)

    data_values_2 = data_values[lower_channel:upper_channel]

    channels_2 = channels[lower_channel:upper_channel]

    energies = [channel_to_energy(channels_pique, channel) for channel in channels_2]

    popt = fit_gaussien(data_values_2, channels_2)

    data_pic_without_noise = take_off_noises(coef, file_path, data_values_2, channels_2, popt)

    popt1 = fit_gaussien(data_pic_without_noise, channels_2)

    a_fit, mu_fit, sigma_fit = popt

    a_fit1, mu_fit1, sigma_fit1 = popt1
    
    plt.plot(energies, data_values_2, label="Nb de coups avec bruits")
    plt.plot(energies, data_pic_without_noise, label="Nb de coups sans bruits")
    plt.plot(energies, gaussian(channels_2, a_fit, mu_fit, sigma_fit), label="Gaussian fit")
    plt.xlabel("Énergie (keV)")
    plt.ylabel("Nombre de coups détectés")
    plt.title("Spectre d'énergie des rayons gamma diffusés (diffusion avec Diffuseur_Al_gros_110_degres)")
    plt.legend()
    plt.show()
    
#print(plot(r'C:\Users\طه\Desktop\Rapport final a remettre\data pour la diffusion\45_degres_avec_un_diffuseur_de_fer.Spe', np.array([607, 2349, 2951])))
print(energie_pic(r'C:\Users\طه\Desktop\Rapport final a remettre\Taha et Joan reprise electron\45-Coincidence.Spe',np.array([688, 2664, 3384]), 10))
#print(nombre_de_coup_sous_le_pic(r'C:\Users\طه\Desktop\Rapport final a remettre\data pour la diffusion\45_degres_avec_un_diffuseur_de_fer.Spe'))