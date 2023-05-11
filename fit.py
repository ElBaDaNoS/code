import numpy as np
import matplotlib.pyplot as plt

# Génération de données exemple
channels = np.array([616, 2265, 2871])
Erreur_channels = np.array([40.84, 49.16, 47.30])
energies = np.array([122, 511, 662])

# Calcul de la moyenne de x et y
x_mean = np.mean(channels)
y_mean = np.mean(energies)

# Calcul des écarts pour x et y
x_deviation = channels - x_mean
y_deviation = energies - y_mean

# Calcul de la pente (a) et de l'ordonnée à l'origine (b)
a = np.sum(x_deviation * y_deviation) / np.sum(x_deviation**2)
b = y_mean - a * x_mean

# Calcul des valeurs prédites de y
y_pred = a * channels + b

# Calcul de R²
ss_res = np.sum((energies - y_pred)**2)  # Correction ici
ss_tot = np.sum((energies - y_mean)**2)
r_squared = 1 - (ss_res / ss_tot)

print("Coefficient de détermination R²:", r_squared)

# nuage de point
plt.scatter(channels, energies, color='r', marker='p', s=70)

# légende
plt.xlabel('Cannal (C)')
plt.ylabel('E[keV]')

# axes
axes = plt.gca()
axes.yaxis.set_tick_params(direction='in')
axes.xaxis.set_tick_params(direction='in')

# titre et affichage 
plt.errorbar(channels, energies, xerr=Erreur_channels, fmt = 'none', capsize = 4, ecolor = 'black', zorder = 10)


x_line = np.linspace(np.min(channels), np.max(channels), 100)
y_line = a * x_line + b
plt.plot(x_line, y_line, color='black', label=f"Fit linéaire: y = {a:.2f}x + {b:.2f}, R² = {r_squared:.3f}")
plt.legend()

plt.show()