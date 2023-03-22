import numpy as np


def calculate_epsR_epsL(MM, ME, excited_energy, nxct, W, alpha):
    elementary_charge = 1.6e-19
    vol = 42000  # in AU^3
    pref = 16.0 * np.pi * elementary_charge ** 2  # / ((vol)*((5.2179E-11)**3))

    Y = np.zeros_like(W)
    Y1 = np.zeros_like(W)
    Y2 = np.zeros_like(W)
    for s in range(nxct):
        energyDif = excited_energy[s]
        E = (ME[s, 0] + 1j * ME[s, 1]) / 2.17 / 2 ** 0.5
        M = (-1j * MM[s, 0] + MM[s, 1]) / 2 ** 0.5
        M /= -2
        Y1 += (abs(M + E)) ** 2 * np.exp(-alpha * (W - energyDif) ** 2)

        E = (ME[s, 0] - 1j * ME[s, 1]) / 2.17 / 2 ** 0.5
        M = (1j * MM[s, 0] + MM[s, 1]) / 2 ** 0.5
        M /= -2
        Y2 += (abs(M + E)) ** 2 * np.exp(-alpha * (W - energyDif) ** 2)

    Y = Y1 - Y2

    Y *= pref
    Y1 *= pref
    Y2 *= pref

    Y[1:] /= (W[1:] ** 2 * (1.6E-19) ** 2)
    Y1[1:] /= (W[1:] ** 2 * (1.6E-19) ** 2)
    Y2[1:] /= (W[1:] ** 2 * (1.6E-19) ** 2)

    return Y, Y1, Y2
