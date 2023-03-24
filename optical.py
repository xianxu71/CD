import numpy as np
from math_function import delta_lorentzian
import matplotlib.pyplot as plt


def calculate_epsR_epsL(nk,MM, ME, excited_energy, nxct, W, eta, volume):
    pref = 16.0 * np.pi**2/volume/nk
    RYD = 13.6057039763
    Y = np.zeros_like(W)
    Y1 = np.zeros_like(W)
    Y2 = np.zeros_like(W)
    for s in range(nxct):
        energyDif = excited_energy[s]
        #E = (ME[s, 0] + 1j * ME[s, 1]) / 2.17 / 2 ** 0.5
        E = (ME[s, 0] + 1j * ME[s, 1]) / 2 ** 0.5
        M = (-1j * MM[s, 0] + MM[s, 1]) / 2 ** 0.5/(-20)
        #Y1 += (abs(M + E)) ** 2 * np.exp(-alpha * (W - energyDif) ** 2)
        Y1 += (abs(M + E)) ** 2 * delta_lorentzian(W/RYD, energyDif/RYD, eta/RYD)/(energyDif/RYD)**2
        #Y1 += np.imag(E)* delta_lorentzian(W / RYD, energyDif / RYD, eta / RYD)

        #E = (ME[s, 0] - 1j * ME[s, 1]) / 2.17 / 2 ** 0.5
        E = (ME[s, 0] - 1j * ME[s, 1]) / 2 ** 0.5
        M = (1j * MM[s, 0] + MM[s, 1]) / 2 ** 0.5/(-20)
        #Y2 += (abs(M + E)) ** 2 * np.exp(-alpha * (W - energyDif) ** 2)
        Y2 += (abs(M + E)) ** 2 * delta_lorentzian(W/RYD, energyDif/RYD, eta/RYD)/(energyDif/RYD)**2
        #Y2 += np.imag(E) ** 2 * delta_lorentzian(W / RYD, energyDif / RYD, eta / RYD)
    Y = Y1 - Y2

    Y *= pref
    Y1 *= pref
    Y2 *= pref

    # Y[1:] /= (W[1:] ** 2 * (1.6E-19) ** 2)
    # Y1[1:] /= (W[1:] ** 2 * (1.6E-19) ** 2)
    # Y2[1:] /= (W[1:] ** 2 * (1.6E-19) ** 2)
    # Y[1:] /= ((W[1:]/RYD) ** 2)
    # Y1[1:] /= ((W[1:]/RYD) ** 2)
    # Y2[1:] /= ((W[1:]/RYD) ** 2)

    plt.figure()
    plt.plot(W, Y, 'r')
    plt.plot(W, Y1, 'b')
    plt.plot(W, Y2, 'g')
    plt.show()

    data = np.array([W, Y1, Y2, Y])
    np.savetxt('CD.dat', data.T)

    return Y, Y1, Y2

def calculate_absorption_eh(nk,MM, ME, excited_energy, nxct, W, eta, volume):
    pref = 16.0 * np.pi**2/volume/nk
    RYD = 13.6057039763
    Y = np.zeros_like(W)
    Y1 = np.zeros_like(W)
    Y2 = np.zeros_like(W)
    for s in range(nxct):
        energyDif = excited_energy[s]
        #E = (ME[s, 0] + 1j * ME[s, 1]) / 2.17 / 2 ** 0.5
        E = (ME[s, 0])
        Y1 += (abs(E)) ** 2 * delta_lorentzian(W/RYD, energyDif/RYD, eta/RYD)
        #Y1 += np.imag(E)* delta_lorentzian(W / RYD, energyDif / RYD, eta / RYD)

        #E = (ME[s, 0] - 1j * ME[s, 1]) / 2.17 / 2 ** 0.5
        E = (ME[s, 1])
        #Y2 += (abs(M + E)) ** 2 * np.exp(-alpha * (W - energyDif) ** 2)
        Y2 += (abs(E)) ** 2 * delta_lorentzian(W/RYD, energyDif/RYD, eta/RYD)
        #Y2 += np.imag(E) ** 2 * delta_lorentzian(W / RYD, energyDif / RYD, eta / RYD)
    Y = Y1 - Y2

    Y *= pref
    Y1 *= pref
    Y2 *= pref

    # Y[1:] /= (W[1:] ** 2 * (1.6E-19) ** 2)
    # Y1[1:] /= (W[1:] ** 2 * (1.6E-19) ** 2)
    # Y2[1:] /= (W[1:] ** 2 * (1.6E-19) ** 2)
    Y[1:] /= ((W[1:]/RYD) ** 2)
    Y1[1:] /= ((W[1:]/RYD) ** 2)
    Y2[1:] /= ((W[1:]/RYD) ** 2)

    plt.figure()
    plt.plot(W, Y, 'r')
    plt.plot(W, Y1, 'b')
    plt.plot(W, Y2, 'g')
    plt.show()

    data = np.array([W, Y1, Y2, Y])
    np.savetxt('absp.dat', data.T)

    return Y, Y1, Y2

def calculate_absorption_noeh(noeh_dipole, nk, nv, nc, energy_dft, W, eta, volume):
    pref = 16.0 * np.pi**2/volume/nk
    RYD = 13.6057039763
    Y = np.zeros_like(W)
    Y1 = np.zeros_like(W)
    Y2 = np.zeros_like(W)

    for ik in range(nk):
        for iv in range(nv):
            for ic in range(nc):
                energyDif = energy_dft[ik,ic+nv]-energy_dft[ik,iv]
                Y1 += np.abs(noeh_dipole[ik,iv,ic+nv,0])**2 * delta_lorentzian(W/RYD, energyDif, eta/RYD)/(energyDif**2)
                Y2 += np.abs(noeh_dipole[ik, iv, ic + nv, 1]) ** 2 * delta_lorentzian(W / RYD, energyDif,
                                                                                      eta / RYD)/(energyDif**2)


    Y = Y1 - Y2

    Y *= pref
    Y1 *= pref
    Y2 *= pref

    # Y[1:] /= (W[1:] ** 2 * (1.6E-19) ** 2)
    # Y1[1:] /= (W[1:] ** 2 * (1.6E-19) ** 2)
    # Y2[1:] /= (W[1:] ** 2 * (1.6E-19) ** 2)
    # Y[1:] /= ((W[1:]/RYD) ** 2)
    # Y1[1:] /= ((W[1:]/RYD) ** 2)
    # Y2[1:] /= ((W[1:]/RYD) ** 2)

    plt.figure()
    plt.plot(W, Y, 'r')
    plt.plot(W, Y1, 'b')
    plt.plot(W, Y2, 'g')
    plt.show()

    data = np.array([W, Y1, Y2, Y])
    np.savetxt('absp.dat', data.T)

    return Y, Y1, Y2