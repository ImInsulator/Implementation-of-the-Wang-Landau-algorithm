#!/usr/bin/env python
"""
  Wang-Landau algorithm for 2D Ising model
  Jueyuan Xiao, Aug.6, 2021, using python 3.6
"""

import matplotlib.pyplot as plt
import numpy as np
import math
import time

MCsweeps = 1000000   # Total MC sweeps
L = 256              # 2D Ising model lattice = L x L
flatness = 0.8      # “flat histogram”: histogram H(E) for all possible E is not less than 80% of the average histogram
N = L * L           # 2D Ising model lattice = L x L

def calEnergy(lattice):
    # Energy of a 2D Ising lattice
    E_N = 0
    for i in range(L):
        for j in range(L):
            S = lattice[i, j]
            WF = lattice[(i+1) % L, j] + lattice[i, (j+1) % L] + lattice[(i-1) % L, j] + lattice[i, (j-1) % L]
            E_N += -WF * S  # Each neighbor gives environment energy
    return int(E_N / 2.)  # Counted twice

def thermod(T, lngE, Energies, E0):
    # Thermodynamics using DOS, T: Temperature
    Z = 0   # Z = sum of g(E) * exp(-beta*E)
    E_T = 0 # = U(T) = (sum of E * g(E) * exp(-beta*E)) / (sum of g(E) * exp(-beta*E))
    E2_T = 0
    kB = 1
    for i, E in enumerate(Energies):
        try:
            w = float(math.exp(lngE[i] - lngE[0] - (E + E0) / T))
        except OverflowError:
            w = float('inf')
        Z += w
        E_T += w * E
        E2_T += w * E ** 2
    E_T *= 1. / Z
    C_T = (E2_T / Z - E_T ** 2) / T ** 2    # the DoS is calculation of the specific heat
    F_T = -kB * T * math.log(Z)
    #P_ET = math.pow(w, 1 / (kB * T))
    #print(P_ET, ", ", E_T/N)
    S_T = (E_T - F_T) / T
    return (F_T / N, C_T / N, E_T / N, S_T)

def WangLandau(MCsweeps, L, N, indE, E0, flatness):
    # Ising lattice at infinite temperature, Generates a random 2D Ising lattice
    latt = np.random.random_integers(-1, high=0, size=(L, L))
    latt[latt == 0] = 1
    # Corresponding energy
    Ene = calEnergy(latt)
    lngE = np.zeros(len(Energies), dtype=np.float)  # Logarithm of the density of states log(g(E))
    Hist = np.zeros(len(Energies), dtype=np.float)  # Histogram
    lnf = 1.0   # g(E) -> g(E)*f, or equivalently, lngE[i] -> lngE[i] + lnf. f0=e

    for itt in range(MCsweeps):
        n = int(np.random.rand() * N)           # The site to flip
        (i, j) = (int(n % L), int(n / L))       # The coordinates of the site
        S = latt[i, j]                          # its spin
        WF = latt[(i + 1) % L, j] + latt[i, (j + 1) % L] + latt[(i - 1) % L, j] + latt[i, (j - 1) % L]
        Enew = Ene + 2 * S * WF                 # The energy of the tried step

        #P = exp(lngE[indE[Ene + E0]] - lngE[indE[Enew + E0]])  # Probability to accept according to Wang-Landau
        lnP = lngE[indE[Ene + E0]] - lngE[indE[Enew + E0]]
        if lnP > math.log((np.random.rand())):  # Metropolis condition, use log for easy cal
            latt[i, j] = -S  # step is accepted, update lattice
            Ene = Enew  # accept the new energy

        Hist[indE[Ene + E0]] += 1.  # Histogram is update at each Monte Carlo step!
        lngE[indE[Ene + E0]] += lnf  # Density of states is also modified at each step!

        if itt % 100 == 0:
            aH = sum(Hist) / (N + 0.0)  # mean Histogram
            mH = min(Hist)  # minimum of the Histogram
            if mH > aH * flatness:  # min(Histogram) >= average(Histogram)*flatness as "flat"
                # Normalize the histogram
                # Hist *= len(Hist)/float(sum(Hist))
                lgC = lngE - lngE[0]
                plt.plot(Energies, lgC, '-o', label='log(g(E))')
                plt.plot(Energies, Hist, '-s', label='Histogram')
                plt.xlabel('Energy')
                plt.legend(loc='best')
                print(itt, 'steps , min H =', mH, ',  average H =', aH, ',  f =', math.exp(lnf))
                #plt.show()
                Hist = np.zeros(len(Hist))  # Resetting histogram
                lnf /= 2.  # and reducing the modification factor
    return (lngE, Hist)

time_start = time.time()
# Possible energies of the Ising model
Energies = (4 * np.arange(N + 1) - 2 * N).tolist()
Energies.pop(1)  # Note that energies Emin+4 and Emax-4
Energies.pop(-2)  # Remove impossible value

# Maximum energy
E0 = Energies[-1]
# Index array which will give us position in the Histogram array from knowing the Energy
indE = -np.ones(E0 * 2 + 1, dtype=int)
for i, E in enumerate(Energies): indE[E + E0] = i

(lngE, Hist) = WangLandau(MCsweeps, L, N, indE, E0, flatness)

# Normalize the density of states, knowing that the lowest energy state is double degenerate
# lgC = log( (exp(lngE[0])+exp(lngE[-1]))/4. )
if lngE[-1] < lngE[0]:
    lgC = lngE[0] + math.log(1 + math.exp(lngE[-1] - lngE[0])) - math.log(4.)
else:
    lgC = lngE[-1] + math.log(1 + math.exp(lngE[0] - lngE[-1])) - math.log(4.)
lngE -= lgC
for i in range(len(lngE)):
    if lngE[i] < 0: lngE[i] = 0
# Normalize the histogram
Hist *= len(Hist) / float(sum(Hist))

Te = np.linspace(0.4, 8., 600)  # T varies from 0.4 to 8
Thm = []
for T in Te:
    Thm.append(thermod(T, lngE, Energies, E0))
Thm = np.array(Thm)
time_end = time.time()
print('Time cost', time_end - time_start, 's')

plt.plot(Energies, lngE, '-o', label='log(g(E))')
plt.plot(Energies, Hist, '-s', label='Histogram')
plt.xlabel('Energy')
plt.legend(loc='best')
plt.show()
fig = plt.figure(1)
ax1 = plt.subplot(2, 1, 1)
plt.plot(Te, Thm[:, 0], label='F(T)/N')
plt.xlabel('T')
plt.legend(loc='best')
ax2 = plt.subplot(2, 1, 2)
plt.plot(Te, Thm[:, 1], label='C(T)/N')
plt.xlabel('T')
plt.legend(loc='best')
plt.show()
plt.plot(Te, Thm[:, 3], '-s', label='S(T)')
plt.xlabel('T')
plt.legend(loc='best')
plt.show()