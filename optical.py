import numpy as np
from math_function import delta_lorentzian, delta_guass
import matplotlib.pyplot as plt

def calculate_epsR_epsL_noeh(E_kvc, L_kvc,nk, nv, nc, energy_dft, W, eta, volume):
    pref = 16.0 * np.pi**2/volume/nk
    RYD = 13.6057039763 #W=W/RYD
    Y = np.zeros_like(W)
    Y1 = np.zeros_like(W)
    Y2 = np.zeros_like(W)

    E1 = (E_kvc[: , :, :, 0] + 1j * E_kvc[:, :, :, 1])
    # M = (-1j * MM[s, 0] + MM[s, 1]) / 2 ** 0.5/(-2)/5
    M1 = (-1j * L_kvc[:,:,:,0] + L_kvc[:,:,:, 1]) / 2/15

    E2 = (E_kvc[: , :, :, 0] - 1j * E_kvc[:, :, :, 1])
    # M = (-1j * MM[s, 0] + MM[s, 1]) / 2 ** 0.5/(-2)/5
    M2 = (+1j * L_kvc[:,:,:,0] + L_kvc[:,:,:, 1]) / 2/15
    for ik in range(nk):
        for iv in range(nv):
            for ic in range(nc):
                energyDif = energy_dft[ik,ic+nv]-energy_dft[ik,iv]
                Y1 += np.abs(E1[ik,iv,ic]+M1[ik,iv,ic])**2 * delta_lorentzian(W/RYD, energyDif, eta/RYD)/(energyDif**2)
                Y2 += np.abs(E2[ik, iv, ic] + M2[ik, iv, ic]) ** 2 * delta_lorentzian(W / RYD, energyDif, eta / RYD) / (
                            energyDif ** 2)



    Y = Y1 - Y2

    Y *= pref*0.1
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
    #plt.plot(W, Y1, 'b')
    #plt.plot(W, Y2, 'g')
    plt.ylim([-0.04,0.1])
    plt.show()

    data = np.array([W, Y1, Y2, Y])
    np.savetxt('absp.dat', data.T)

    return Y, Y1, Y2
def calculate_epsR_epsL_eh(nk,MM, ME, excited_energy, nxct, W, eta, volume):
    pref = 16.0 * np.pi**2/volume/nk
    RYD = 13.6057039763
    light_speed = 137
    epsilon_r = 1
    Y = np.zeros_like(W)
    Y1 = np.zeros_like(W)
    Y2 = np.zeros_like(W)
    for s in range(nxct):
        energyDif = excited_energy[s]
        #E = (ME[s, 0] + 1j * ME[s, 1]) / 2.17 / 2 ** 0.5
        E_L = (ME[s, 0] + 1j * ME[s, 1])
        #M = (-1j * MM[s, 0] + MM[s, 1]) / 2 ** 0.5/(-2)/5
        M_L = (-1j*MM[s, 0] + MM[s, 1])/2/nk * W/RYD/light_speed/epsilon_r #*volume*nk/(10**15)
        #Y1 += (abs(M + E)) ** 2 * np.exp(-alpha * (W - energyDif) ** 2)
        Y1 += (abs(M_L + E_L)) ** 2 * delta_lorentzian(W/RYD, energyDif/RYD, eta/RYD)/(energyDif/RYD)**2/2
        #Y1 += np.imag(E)* delta_lorentzian(W / RYD, energyDif / RYD, eta / RYD)

        #E = (ME[s, 0] - 1j * ME[s, 1]) / 2.17 / 2 ** 0.5
        E_R = (ME[s, 0] - 1j * ME[s, 1])
        #M = (1j * MM[s, 0] + MM[s, 1]) / 2 ** 0.5/(-2)/5
        M_R = (1j*MM[s, 0] + MM[s, 1])/2/nk * W/RYD/light_speed/epsilon_r #*volume*nk/(10**15)
        #Y2 += (abs(M + E)) ** 2 * np.exp(-alpha * (W - energyDif) ** 2)
        Y2 += (abs(M_R + E_R)) ** 2 * delta_lorentzian(W/RYD, energyDif/RYD, eta/RYD)/(energyDif/RYD)**2/2

        Y += np.abs(M_L*np.conj(E_L)+E_L*np.conj(M_L)-M_R*np.conj(E_R)-E_R*np.conj(M_R))* delta_lorentzian(W/RYD, energyDif/RYD, eta/RYD)/(energyDif/RYD)**2/2
        #Y2 += np.imag(E) ** 2 * delta_lorentzian(W / RYD, energyDif / RYD, eta / RYD)


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
    plt.plot(W, Y, 'r', label = 'L-R')
    #plt.plot(W, Y1, 'b', label = 'L')
    #plt.plot(W, Y2, 'g', label = 'R')
    plt.legend()
    plt.show()

    data = np.array([W, Y1, Y2, Y])
    np.savetxt('CD.dat', data.T)

    return Y, Y1, Y2

def calculate_absorption_eh(nk,MM, ME, excited_energy, nxct, W, eta, volume):
    pref = 16.0 * np.pi**2/volume/nk
    RYD = 13.6057039763
    # Y = np.zeros_like(W)
    eps_2 = np.zeros_like(W)
    eps_1 = np.zeros_like(W)
    # Y2 = np.zeros_like(W)
    for s in range(nxct):
        energyDif = excited_energy[s]
        #E = (ME[s, 0] + 1j * ME[s, 1]) / 2.17 / 2 ** 0.5
        E = (ME[s, 0])
        eps_2 += (abs(E)) ** 2 * delta_guass(W/RYD, energyDif/RYD, eta/RYD)/np.sqrt(2)
        eps_1 += (abs(E)) ** 2 * delta_lorentzian(W/RYD, energyDif/RYD, eta/RYD)*(energyDif-W)/eta /(energyDif/RYD)**2/np.sqrt(2)
        #eps_1 += -(abs(E)) ** 2 * delta_lorentzian(-W / RYD, energyDif / RYD, eta / RYD) * (energyDif + W) / eta / RYD
        #Y1 += np.imag(E)* delta_lorentzian(W / RYD, energyDif / RYD, eta / RYD)

        #E = (ME[s, 0] - 1j * ME[s, 1]) / 2.17 / 2 ** 0.5
        # E = (ME[s, 1])
        #Y2 += (abs(M + E)) ** 2 * np.exp(-alpha * (W - energyDif) ** 2)
        # Y2 += (abs(E)) ** 2 * delta_lorentzian(W/RYD, energyDif/RYD, eta/RYD)
        #Y2 += np.imag(E) ** 2 * delta_lorentzian(W / RYD, energyDif / RYD, eta / RYD)
    # Y = Y1 - Y2

    eps_2[1:] /= ((W[1:] / RYD) ** 2)
    #eps_1[1:] /= ((W[1:] / RYD) ** 2)
    # Y *= pref
    eps_2 *= pref
    eps_1 = 1 + pref*eps_1
    # Y2 *= pref

    # Y[1:] /= (W[1:] ** 2 * (1.6E-19) ** 2)
    # Y1[1:] /= (W[1:] ** 2 * (1.6E-19) ** 2)
    # Y2[1:] /= (W[1:] ** 2 * (1.6E-19) ** 2)
    # Y[1:] /= ((W[1:]/RYD) ** 2)

    # Y2[1:] /= ((W[1:]/RYD) ** 2)

    plt.figure()
    #plt.plot(W, Y, 'r')
    plt.plot(W, eps_2, 'b',label='eps2')
    plt.plot(W,eps_1,'r',label='eps1')
    #plt.plot(W, Y2, 'g')
    plt.legend()
    plt.show()

    data = np.array([W, eps_2, eps_1])
    np.savetxt('absp.dat', data.T)

    return

def calculate_absorption_noeh(noeh_dipole, nk, nv, nc, energy_dft, W, eta, volume, use_eqp=False, eqp_corr = None):
    pref = 16.0 * np.pi**2/volume/nk
    RYD = 13.6057039763
    #Y = np.zeros_like(W)
    eps_2 = np.zeros_like(W)
    eps_1 = np.zeros_like(W)
    #Y2 = np.zeros_like(W)

    for ik in range(nk):
        for iv in range(nv):
            for ic in range(nc):
                energyDif = energy_dft[ik,ic+nv]-energy_dft[ik,iv]
                if use_eqp:
                    energyDif2 =energyDif + eqp_corr[ik,nv+ic]/RYD-eqp_corr[ik,iv]/RYD
                else:
                    energyDif2 =energyDif

                eps_2 += np.abs(noeh_dipole[ik,iv,ic+nv,0])**2 * (delta_guass(W/RYD, energyDif2, eta/RYD))/(energyDif**2)/2
                eps_1 += np.abs(noeh_dipole[ik, iv, ic + nv, 0]) ** 2 * (
                    delta_lorentzian(W / RYD, energyDif2, eta / RYD)) / (energyDif ** 2)*(energyDif2-W/RYD)/eta * RYD/2
                #Y2 += np.abs(noeh_dipole[ik, iv, ic + nv, 1]) ** 2 * (delta_lorentzian(W / RYD, energyDif,
                #                                                                      eta / RYD)-delta_lorentzian(-W/RYD, energyDif, eta/RYD))/(energyDif**2)


    #Y = Y1 - Y2

    #Y *= pref
    eps_2 *= pref
    eps_1 = pref*eps_1 + 1
    #Y2 *= pref

    # Y[1:] /= (W[1:] ** 2 * (1.6E-19) ** 2)
    # Y1[1:] /= (W[1:] ** 2 * (1.6E-19) ** 2)
    # Y2[1:] /= (W[1:] ** 2 * (1.6E-19) ** 2)
    # Y[1:] /= ((W[1:]/RYD) ** 2)
    # Y1[1:] /= ((W[1:]/RYD) ** 2)
    # Y2[1:] /= ((W[1:]/RYD) ** 2)

    plt.figure()
    #plt.plot(W, Y, 'r')
    plt.plot(W, eps_2, 'b')
    plt.plot(W, eps_1, 'r')
    #plt.plot(W, Y2, 'g')
    plt.show()

    data = np.array([W, eps_2, eps_1])
    np.savetxt('absp.dat', data.T)

    return

def calculate_m(nk,MM, ME, excited_energy, nxct, W, eta, volume):
    pref = 16.0 * np.pi**2/volume/nk
    RYD = 13.6057039763
    Y = np.zeros_like(W)
    Y1 = np.zeros_like(W)
    Y2 = np.zeros_like(W)
    for s in range(nxct):
        energyDif = excited_energy[s]
        #E = (ME[s, 0] + 1j * ME[s, 1]) / 2.17 / 2 ** 0.5
        M = (-1j * MM[s, 0] + MM[s, 1]) / 2
        #M = (MM[s, 0] + 1j*MM[s, 1]) / 2 ** 0.5 / (-20)
        #Y1 += (abs(M + E)) ** 2 * np.exp(-alpha * (W - energyDif) ** 2)
        Y1 += (abs(M)) ** 2 * delta_lorentzian(W/RYD, energyDif/RYD, eta/RYD)/(energyDif/RYD)**2
        #Y1 += np.imag(E)* delta_lorentzian(W / RYD, energyDif / RYD, eta / RYD)

        #E = (ME[s, 0] - 1j * ME[s, 1]) / 2.17 / 2 ** 0.5
        M = (1j * MM[s, 0] + MM[s, 1]) / 2
        #M = (MM[s, 0] - 1j*MM[s, 1]) / 2 ** 0.5 / (-20)
        #Y2 += (abs(M + E)) ** 2 * np.exp(-alpha * (W - energyDif) ** 2)
        Y2 += (abs(M)) ** 2 * delta_lorentzian(W/RYD, energyDif/RYD, eta/RYD)/(energyDif/RYD)**2
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
