import numpy as np
from math_function import delta_lorentzian, delta_gauss
import matplotlib.pyplot as plt

def calculate_epsR_epsL_noeh(E_kvc, L_kvc,nk, nv, nc, energy_dft, W, eta, volume, use_eqp=False, eqp_corr = None):
    pref = 16.0 * np.pi**2/volume/nk
    RYD = 13.6057039763 #W=W/RYD
    light_speed = 137
    epsilon_r = 1
    Y = np.zeros_like(W)
    # Y1 = np.zeros_like(W)
    # Y2 = np.zeros_like(W)
    Y1_eps2 = np.zeros_like(W)
    Y2_eps2 = np.zeros_like(W)
    Y1_eps1 = np.zeros_like(W)
    Y2_eps1 = np.zeros_like(W)
    Y1_eps2_0 = np.zeros_like(W)
    Y2_eps2_0 = np.zeros_like(W)
    Y1_eps1_0 = np.zeros_like(W)
    Y2_eps1_0 = np.zeros_like(W)
    alpha1 = np.zeros_like(W)
    alpha2 = np.zeros_like(W)
    alpha1_0 = np.zeros_like(W)
    alpha2_0 = np.zeros_like(W)

    E1 = (E_kvc[: , :, :, 0] + 1j * E_kvc[:, :, :, 1])
    # M = (-1j * MM[s, 0] + MM[s, 1]) / 2 ** 0.5/(-2)/5
    M1 = (-1j * L_kvc[:,:,:,0] + L_kvc[:,:,:, 1]) / 2/nk

    E2 = (E_kvc[: , :, :, 0] - 1j * E_kvc[:, :, :, 1])
    # M = (-1j * MM[s, 0] + MM[s, 1]) / 2 ** 0.5/(-2)/5
    M2 = (+1j * L_kvc[:,:,:,0] + L_kvc[:,:,:, 1]) / 2/nk
    for ik in range(nk):
        for iv in range(nv):
            for ic in range(nc):
                energyDif = energy_dft[ik,ic+nv]-energy_dft[ik,iv]
                if use_eqp:
                    energyDif2 =energyDif + eqp_corr[ik,nv+ic]/RYD-eqp_corr[ik,iv]/RYD
                else:
                    energyDif2 =energyDif
                Y1_eps2 += np.abs(E1[ik,iv,ic]+M1[ik,iv,ic]* W/RYD/light_speed/epsilon_r)**2\
                      *delta_gauss(W/RYD, energyDif2, eta/RYD)/(energyDif**2)/2/2
                Y2_eps2 += np.abs(E2[ik, iv, ic] + M2[ik, iv, ic]* W/RYD/light_speed/epsilon_r) ** 2\
                      * delta_gauss(W / RYD, energyDif2, eta / RYD) / (energyDif ** 2)/2/2
                Y1_eps1 += np.abs(E1[ik, iv, ic] + M1[ik, iv, ic] * W / RYD / light_speed / epsilon_r) ** 2 \
                           * delta_gauss(W / RYD, energyDif2, eta / RYD) / (energyDif ** 2) / 2 / 2 *(energyDif-W/RYD)/eta
                Y2_eps1 += np.abs(E2[ik, iv, ic] + M2[ik, iv, ic] * W / RYD / light_speed / epsilon_r) ** 2 \
                           * delta_gauss(W / RYD, energyDif2, eta / RYD) / (energyDif ** 2) / 2 / 2 *(energyDif-W/RYD)/eta

                Y1_eps2_0 += np.abs(E1[ik, iv, ic]) ** 2 \
                           * delta_gauss(W / RYD, energyDif2, eta / RYD) / (energyDif ** 2) / 2 / 2
                Y2_eps2_0 += np.abs(E2[ik, iv, ic]) ** 2 \
                           * delta_gauss(W / RYD, energyDif2, eta / RYD) / (energyDif ** 2) / 2 / 2
                Y1_eps1_0 += np.abs(E1[ik, iv, ic]) ** 2 \
                           * delta_gauss(W / RYD, energyDif2, eta / RYD) / (energyDif ** 2) / 2 / 2 * (
                                       energyDif - W/RYD) / eta
                Y2_eps1_0 += np.abs(E2[ik, iv, ic] ) ** 2 \
                           * delta_gauss(W / RYD, energyDif2, eta / RYD) / (energyDif ** 2) / 2 / 2 * (
                                       energyDif - W/RYD) / eta

                # Y += (np.abs(E1[ik,iv,ic]+M1[ik,iv,ic]* W/RYD/light_speed/epsilon_r)**2-
                #       np.abs(E2[ik, iv, ic] + M2[ik, iv, ic]* W/RYD/light_speed/epsilon_r) ** 2-
                #       np.abs(E1[ik,iv,ic])**2+
                #       np.abs(E2[ik,iv,ic])**2)\
                #       *delta_gauss(W/RYD, energyDif2, eta/RYD)/(energyDif**2)/2/2

    Y = Y1_eps2 - Y2_eps2 - (Y1_eps2_0 - Y2_eps2_0)
    Y *= pref
    Y1_eps2 *= pref
    Y2_eps2 *= pref
    Y1_eps1 = pref * Y1_eps1 + 1
    Y2_eps1 = pref * Y2_eps1 + 1

    Y1_eps2_0 *= pref
    Y2_eps2_0 *= pref
    Y1_eps1_0 = pref * Y1_eps1_0 + 1
    Y2_eps1_0 = pref * Y2_eps1_0 + 1

    alpha1 = W * np.sqrt(np.sqrt(Y1_eps1 ** 2 + Y1_eps2 ** 2) - Y1_eps1)
    alpha2 = W * np.sqrt(np.sqrt(Y2_eps1 ** 2 + Y2_eps2 ** 2) - Y2_eps1)

    alpha1_0 = W * np.sqrt(np.sqrt(Y1_eps1_0 ** 2 + Y1_eps2_0 ** 2) - Y1_eps1_0)
    alpha2_0 = W * np.sqrt(np.sqrt(Y2_eps1_0 ** 2 + Y2_eps2_0 ** 2) - Y2_eps1_0)

    CD = alpha1 - alpha2 - (alpha1_0 - alpha2_0)

    # Y[1:] /= (W[1:] ** 2 * (1.6E-19) ** 2)
    # Y1[1:] /= (W[1:] ** 2 * (1.6E-19) ** 2)
    # Y2[1:] /= (W[1:] ** 2 * (1.6E-19) ** 2)
    # Y[1:] /= ((W[1:]/RYD) ** 2)
    # Y1[1:] /= ((W[1:]/RYD) ** 2)
    # Y2[1:] /= ((W[1:]/RYD) ** 2)

    plt.figure()
    plt.plot(W, Y, 'r', label = 'L-R')
    #plt.plot(W, CD, 'r', label='CD')
    # plt.plot(W, alpha1_0, 'b', label='alpha1_0')
    # plt.plot(W, alpha2_0, 'g', label='alpha2_0')
    #plt.plot(W, alpha1, 'b', label='alpha1')
    #plt.plot(W, alpha2, 'g', label='alpha2')

    #plt.plot(W, Y1_eps2, 'b', label = 'L_eps2')
    #plt.plot(W, Y2_eps2, 'g', label = 'R_eps2')
    # plt.plot(W, Y1_eps1, 'k', label = 'L_eps1')
    # plt.plot(W, Y2_eps1, 'y', label = 'R_eps1')
    plt.legend()
    plt.show()

    data = np.array([W, CD])
    np.savetxt('CD.dat', data.T)

    return
def calculate_epsR_epsL_eh(nk,MM, ME, excited_energy, nxct, W, eta, volume):
    pref = 16.0 * np.pi**2/volume/nk
    RYD = 13.6057039763
    light_speed = 137
    epsilon_r = 1
    Y = np.zeros_like(W)
    YY1 = np.zeros_like(W)
    YY2 = np.zeros_like(W)
    #Y1 = np.zeros_like(W)
    #Y2 = np.zeros_like(W)
    Y1_eps2 = np.zeros_like(W)
    Y2_eps2 = np.zeros_like(W)
    Y1_eps1 = np.zeros_like(W)
    Y2_eps1 = np.zeros_like(W)
    Y1_eps2_0 = np.zeros_like(W)
    Y2_eps2_0 = np.zeros_like(W)
    Y1_eps1_0 = np.zeros_like(W)
    Y2_eps1_0 = np.zeros_like(W)
    alpha1 = np.zeros_like(W)
    alpha2 = np.zeros_like(W)
    alpha1_0 = np.zeros_like(W)
    alpha2_0 = np.zeros_like(W)
    for s in range(nxct):
        energyDif = excited_energy[s]
        #E = (ME[s, 0] + 1j * ME[s, 1]) / 2.17 / 2 ** 0.5
        E_L = (ME[s, 0] + 1j * ME[s, 1])
        #M = (-1j * MM[s, 0] + MM[s, 1]) / 2 ** 0.5/(-2)/5
        M_L = (-1j*MM[s, 0] + MM[s, 1])/2/nk * W/RYD/light_speed/epsilon_r
        #Y1 += (abs(M + E)) ** 2 * np.exp(-alpha * (W - energyDif) ** 2)

        #Y1 += np.imag(E)* delta_lorentzian(W / RYD, energyDif / RYD, eta / RYD)

        #E = (ME[s, 0] - 1j * ME[s, 1]) / 2.17 / 2 ** 0.5
        E_R = (ME[s, 0] - 1j * ME[s, 1])
        #M = (1j * MM[s, 0] + MM[s, 1]) / 2 ** 0.5/(-2)/5
        M_R = (1j*MM[s, 0] + MM[s, 1])/2/nk * W/RYD/light_speed/epsilon_r
        #Y2 += (abs(M + E)) ** 2 * np.exp(-alpha * (W - energyDif) ** 2)
        Y1_eps2 += (np.abs(M_L + E_L)) ** 2 * delta_gauss(W / RYD, energyDif / RYD, eta / RYD) / (
                    energyDif / RYD) ** 2 / 2 / np.sqrt(2)
        Y2_eps2 += (np.abs(M_R + E_R)) ** 2 * delta_gauss(W/RYD, energyDif/RYD, eta/RYD)/(energyDif/RYD)**2/2/np.sqrt(2)

        Y1_eps1 += (np.abs(M_L + E_L)) ** 2 * delta_lorentzian(W / RYD, energyDif / RYD, eta / RYD) / (
                    energyDif / RYD) ** 2 / 2 / np.sqrt(2) *(energyDif-W)/eta

        Y2_eps1 += (np.abs(M_R + E_R)) ** 2 * delta_lorentzian(W / RYD, energyDif / RYD, eta / RYD) / (
                    energyDif / RYD) ** 2 / 2 / np.sqrt(2) *(energyDif-W)/eta

        Y1_eps2_0 += (np.abs(E_L)) ** 2 * delta_gauss(W / RYD, energyDif / RYD, eta / RYD) / (
                    energyDif / RYD) ** 2 / 2 / np.sqrt(2)
        Y2_eps2_0 += (np.abs(E_R)) ** 2 * delta_gauss(W/RYD, energyDif/RYD, eta/RYD)/(energyDif/RYD)**2/2/np.sqrt(2)

        Y1_eps1_0 += (np.abs(E_L)) ** 2 * delta_lorentzian(W / RYD, energyDif / RYD, eta / RYD) / (
                    energyDif / RYD) ** 2 / 2 / np.sqrt(2) *(energyDif-W)/eta

        Y2_eps1_0 += (np.abs(E_R)) ** 2 * delta_lorentzian(W / RYD, energyDif / RYD, eta / RYD) / (
                    energyDif / RYD) ** 2 / 2 / np.sqrt(2) *(energyDif-W)/eta


        #Y += ((np.abs(M_L + E_L)) ** 2-np.abs(E_L)**2-(np.abs(M_R + E_R)) ** 2 + np.abs(E_R)**2)* delta_gauss(W/RYD, energyDif/RYD, eta/RYD)/(energyDif/RYD)**2/2/np.sqrt(2)
        #Y += ((np.abs(M_L + E_L)) ** 2 - (np.abs(M_R + E_R)) ** 2) * delta_gauss(
        #    W / RYD, energyDif / RYD, eta / RYD) / (energyDif / RYD) ** 2 / 2 / np.sqrt(2)
        #Y2 += np.imag(E) ** 2 * delta_lorentzian(W / RYD, energyDif / RYD, eta / RYD)

        YY1 += np.real(E_L*np.conj(M_L)+M_L*np.conj(E_L)-E_R*np.conj(M_R)-M_R*np.conj(E_R)) * delta_gauss(W / RYD, energyDif / RYD, eta / RYD) / (
                    energyDif / RYD) ** 2 / 2 / np.sqrt(2)
        YY2 += ((np.abs(M_L + E_L)) ** 2-(np.abs(M_R + E_R)) ** 2-(np.abs(E_L)) ** 2+(np.abs(E_R)) ** 2) * delta_gauss(
            W / RYD, energyDif / RYD, eta / RYD) / (
                      energyDif / RYD) ** 2 / 2 / np.sqrt(2)
        # print('test')

    Y = Y1_eps2-Y2_eps2 - (Y1_eps2_0-Y2_eps2_0)
    Y *= pref
    YY1 *= pref
    YY2 *= pref
    Y1_eps2 *= pref
    Y2_eps2 *= pref
    Y1_eps1 = pref*Y1_eps1+1
    Y2_eps1 = pref*Y2_eps1+1

    Y1_eps2_0 *= pref
    Y2_eps2_0 *= pref
    Y1_eps1_0 = pref*Y1_eps1+1
    Y2_eps1_0 = pref*Y2_eps1+1

    alpha1 = W *np.sqrt(np.sqrt(Y1_eps1**2+Y1_eps2**2)-Y1_eps1)
    alpha2 = W * np.sqrt(np.sqrt(Y2_eps1 ** 2 + Y2_eps2 ** 2) - Y2_eps1)

    alpha1_0 = W * np.sqrt(np.sqrt(Y1_eps1_0 ** 2 + Y1_eps2_0 ** 2) - Y1_eps1_0)
    alpha2_0 = W * np.sqrt(np.sqrt(Y2_eps1_0 ** 2 + Y2_eps2_0 ** 2) - Y2_eps1_0)

    # alpha1 = W*Y1_eps2/np.sqrt(Y1_eps1)
    # alpha2 = W * Y2_eps2 / np.sqrt(Y2_eps1)
    # alpha1_0 = W * Y1_eps2_0 / np.sqrt(Y1_eps1_0)
    # alpha2_0 = W * Y2_eps2_0 / np.sqrt(Y1_eps1_0)

    CD = alpha1 - alpha2 - (alpha1_0 - alpha2_0)
    #deps2 = Y1_eps2 - Y2_eps2 - (Y1_eps2_0 - Y2_eps2_0)



    # Y[1:] /= (W[1:] ** 2 * (1.6E-19) ** 2)
    # Y1[1:] /= (W[1:] ** 2 * (1.6E-19) ** 2)
    # Y2[1:] /= (W[1:] ** 2 * (1.6E-19) ** 2)
    # Y[1:] /= ((W[1:]/RYD) ** 2)
    # Y1[1:] /= ((W[1:]/RYD) ** 2)
    # Y2[1:] /= ((W[1:]/RYD) ** 2)

    plt.figure()
    plt.plot(W, Y, 'r', label = 'L-R')
    plt.plot(W, YY1, 'g', label='YY1')
    plt.plot(W, YY2, 'b', label='YY2')
    #plt.plot(W, CD, 'r', label='CD')
    # plt.plot(W, alpha1_0, 'b', label='alpha1_0')
    # plt.plot(W, alpha2_0, 'g', label='alpha2_0')
    # plt.plot(W, alpha1, 'b', label='alpha1')
    # plt.plot(W, alpha2, 'g', label='alpha2')

    # plt.plot(W, Y1_eps2, 'b', label = 'L_eps2')
    # plt.plot(W, Y2_eps2, 'g', label = 'R_eps2')
    # plt.plot(W, Y1_eps1, 'k', label = 'L_eps1')
    # plt.plot(W, Y2_eps1, 'y', label = 'R_eps1')
    plt.legend()
    plt.show()

    data = np.array([W, CD])
    np.savetxt('CD.dat', data.T)

    return

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
        eps_2 += (abs(E)) ** 2 * delta_gauss(W/RYD, energyDif/RYD, eta/RYD)/np.sqrt(2)
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

                eps_2 += np.abs(noeh_dipole[ik,iv,ic+nv,0])**2 * (delta_gauss(W/RYD, energyDif2, eta/RYD))/(energyDif**2)/2
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
    #plt.plot(W, eps_1, 'r')
    #plt.plot(W, Y2, 'g')
    plt.show()

    data = np.array([W, eps_2, eps_1])
    np.savetxt('absp.dat', data.T)

    return

def calculate_m(nk,MM, ME, excited_energy, nxct, W, eta, volume):
    pref = 16.0 * np.pi ** 2 / volume / nk
    RYD = 13.6057039763
    # Y = np.zeros_like(W)
    eps_2_x = np.zeros_like(W)
    eps_2_z = np.zeros_like(W)
    eps_2_x_z = np.zeros_like(W)

    eps_1 = np.zeros_like(W)
    # Y2 = np.zeros_like(W)
    for s in range(nxct):
        energyDif = excited_energy[s]
        # E = (ME[s, 0] + 1j * ME[s, 1]) / 2.17 / 2 ** 0.5
        #M = (ME[s, 2])
        #M_z = (ME[s, 2])
        M_x = (MM[s, 0])
        eps_2_x += (abs(M_x)) ** 2 * delta_gauss(W / RYD, energyDif / RYD, eta / RYD) / np.sqrt(2)
        #eps_2_z += (abs(M_z)) ** 2 * delta_gauss(W / RYD, energyDif / RYD, eta / RYD) / np.sqrt(2)
        #eps_2_x_z += (abs(M_z)) *(abs(M_x)) * delta_gauss(W / RYD, energyDif / RYD, eta / RYD) / np.sqrt(2)
        # eps_1 += (abs(M)) ** 2 * delta_lorentzian(W / RYD, energyDif / RYD, eta / RYD) * (energyDif - W) / eta / (
        #             energyDif / RYD) ** 2 / np.sqrt(2)
        # eps_1 += -(abs(E)) ** 2 * delta_lorentzian(-W / RYD, energyDif / RYD, eta / RYD) * (energyDif + W) / eta / RYD
        # Y1 += np.imag(E)* delta_lorentzian(W / RYD, energyDif / RYD, eta / RYD)

        # E = (ME[s, 0] - 1j * ME[s, 1]) / 2.17 / 2 ** 0.5
        # E = (ME[s, 1])
        # Y2 += (abs(M + E)) ** 2 * np.exp(-alpha * (W - energyDif) ** 2)
        # Y2 += (abs(E)) ** 2 * delta_lorentzian(W/RYD, energyDif/RYD, eta/RYD)
        # Y2 += np.imag(E) ** 2 * delta_lorentzian(W / RYD, energyDif / RYD, eta / RYD)
    # Y = Y1 - Y2

    eps_2_x[1:] /= ((W[1:] / RYD) ** 2)
    #eps_2_z[1:] /= ((W[1:] / RYD) ** 2)
    #eps_2_x_z[1:] /= ((W[1:] / RYD) ** 2)
    # eps_1[1:] /= ((W[1:] / RYD) ** 2)
    # Y *= pref
    eps_2_x *= pref
    #eps_2_z *= pref
    #eps_2_x_z *= pref

    # eps_1 = 1 + pref * eps_1
    # Y2 *= pref

    # Y[1:] /= (W[1:] ** 2 * (1.6E-19) ** 2)
    # Y1[1:] /= (W[1:] ** 2 * (1.6E-19) ** 2)
    # Y2[1:] /= (W[1:] ** 2 * (1.6E-19) ** 2)
    # Y[1:] /= ((W[1:]/RYD) ** 2)

    # Y2[1:] /= ((W[1:]/RYD) ** 2)

    plt.figure()
    # plt.plot(W, Y, 'r')
    plt.plot(W, eps_2_x, 'b', label='eps2_x')
    #plt.plot(W, eps_2_z, 'r', label='eps2_z')
    #plt.plot(W, eps_2_x_z, 'y', label='eps2_x_z')
    #plt.plot(W, eps_1, 'r', label='eps1')
    # plt.plot(W, Y2, 'g')
    plt.legend()
    plt.show()
    #
    # data = np.array([W, eps_2, eps_1])
    # np.savetxt('absp.dat', data.T)

    return
