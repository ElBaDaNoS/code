import numpy as np
import matplotlib.pyplot as plt

def Efficacité_détecteur(E):
    a, b = np.polyfit(np.log(np.array([200, 400, 600])), np.log(np.array([0.5, 0.28, 0.17])), 1)

    return np.exp(np.log(E) * a + b)

E = np.array([122, 511, 662])
Section_efficace = np.array([0.3347, 0.22, 0.1589, 0.1304, 0.1176])
Erreur_E = np.array([38.97, 63.33, 62.06])
theta = np.array([688, 2664, 3384])

# nuage de point
plt.scatter(theta, E, color='r', marker='p', s=70, label='Section efficace mesurée')
#plt.scatter(theta, Section_efficace, color='b', marker='o', s=70, label='Section efficace théorique')

# légende
plt.xlabel('Cannal (C)')
plt.ylabel('E [KeV]')

# axes
axes = plt.gca()
axes.yaxis.set_tick_params(direction='in')
axes.xaxis.set_tick_params(direction='in')

# titre et affichage 
plt.errorbar(theta, E, xerr=Erreur_E, fmt = 'none', capsize = 4, ecolor = 'black', zorder = 10)
plt.legend()
plt.show()