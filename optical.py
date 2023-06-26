import numpy as np
from math_function import *
import matplotlib.pyplot as plt

def calculate_epsR_epsL_noeh(main_class):
    energy_shift=main_class.energy_shift
    eps1_correction=main_class.eps1_correction
    volume = main_class.volume
    nk = main_class.nk
    W = main_class.W
    E_kvc = main_class.E_kvc
    L_kvc = main_class.L_kvc
    nv = main_class.nv
    nc = main_class.nc
    energy_dft = main_class.energy_dft
    use_eqp = main_class.use_eqp
    if use_eqp:
        eqp_corr= main_class.eqp_corr
    eta = main_class.eta



    pref = 16.0 * np.pi ** 2 / volume / nk
    RYD = 13.6057039763  # W=W/RYD
    light_speed = 137
    epsilon_r = 1
    Y = np.zeros_like(W)
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

    E1 = (E_kvc[:, :, :, 0] + 1j * E_kvc[:, :, :, 1])
    # M = (-1j * MM[s, 0] + MM[s, 1]) / 2 ** 0.5/(-2)/5
    M1 = (1 * L_kvc[:, :, :, 0] + 1j * L_kvc[:, :, :, 1]) / 2 / nk * (1j)

    E2 = (E_kvc[:, :, :, 0] - 1j * E_kvc[:, :, :, 1])

    dE = np.abs(E1)**2 - np.abs(E2)**2

    # M = (-1j * MM[s, 0] + MM[s, 1]) / 2 ** 0.5/(-2)/5
    E1_2 = np.abs(E1)**2
    E2_2 = np.abs(E2)**2
    dE_2 = E1_2 - E2_2
    M2 = (- L_kvc[:, :, :, 0] + 1j * L_kvc[:, :, :, 1]) / 2 / nk * (1j)
    for ik in range(nk):
        for iv in range(nv): #range(nv)
            for ic in range(nc): #range(nc)
                energyDif = energy_dft[ik, ic + nv] - energy_dft[ik, iv]+energy_shift/RYD
                if use_eqp:
                    energyDif2 = energyDif + eqp_corr[ik, nv + ic] / RYD - eqp_corr[ik, iv] / RYD
                else:
                    energyDif2 = energyDif
                Y1_eps2 += np.abs(E1[ik, iv, ic] + M1[ik, iv, ic] * W / RYD / light_speed / epsilon_r) ** 2 \
                           * delta_gauss(W / RYD, energyDif2, eta / RYD) / (energyDif ** 2) / 2 / 2
                Y2_eps2 += np.abs(E2[ik, iv, ic] + M2[ik, iv, ic] * W / RYD / light_speed / epsilon_r) ** 2 \
                           * delta_gauss(W / RYD, energyDif2, eta / RYD) / (energyDif ** 2) / 2 / 2
                Y1_eps1 += np.abs(E1[ik, iv, ic] + M1[ik, iv, ic] * W / RYD / light_speed / epsilon_r) ** 2 \
                           * delta_gauss(W / RYD, energyDif2, eta / RYD) / (energyDif ** 2) / 2 / 2 * (
                                       energyDif - W / RYD) / eta
                Y2_eps1 += np.abs(E2[ik, iv, ic] + M2[ik, iv, ic] * W / RYD / light_speed / epsilon_r) ** 2 \
                           * delta_gauss(W / RYD, energyDif2, eta / RYD) / (energyDif ** 2) / 2 / 2 * (
                                       energyDif - W / RYD) / eta

                Y1_eps2_0 += np.abs(E1[ik, iv, ic]) ** 2 \
                             * delta_gauss(W / RYD, energyDif2, eta / RYD) / (energyDif ** 2) / 2 / 2
                Y2_eps2_0 += np.abs(E2[ik, iv, ic]) ** 2 \
                             * delta_gauss(W / RYD, energyDif2, eta / RYD) / (energyDif ** 2) / 2 / 2
                Y1_eps1_0 += np.abs(E1[ik, iv, ic]) ** 2 \
                             * delta_gauss(W / RYD, energyDif2, eta / RYD) / (energyDif ** 2) / 2 / 2 * (
                                     energyDif - W / RYD) / eta
                Y2_eps1_0 += np.abs(E2[ik, iv, ic]) ** 2 \
                             * delta_gauss(W / RYD, energyDif2, eta / RYD) / (energyDif ** 2) / 2 / 2 * (
                                     energyDif - W / RYD) / eta

                # Y += (np.abs(E1[ik,iv,ic]+M1[ik,iv,ic]* W/RYD/light_speed/epsilon_r)**2-
                #       np.abs(E2[ik, iv, ic] + M2[ik, iv, ic]* W/RYD/light_speed/epsilon_r) ** 2-
                #       np.abs(E1[ik,iv,ic])**2+
                #       np.abs(E2[ik,iv,ic])**2)\
                #       *delta_gauss(W/RYD, energyDif2, eta/RYD)/(energyDif**2)/2/2

    Y = Y1_eps2 - Y2_eps2 - (Y1_eps2_0 - Y2_eps2_0)
    Y *= pref
    Y1_eps2 *= pref
    Y2_eps2 *= pref
    Y1_eps1 = pref * Y1_eps1 + 1 +eps1_correction
    Y2_eps1 = pref * Y2_eps1 + 1 +eps1_correction

    Y1_eps2_0 *= pref
    Y2_eps2_0 *= pref
    Y1_eps1_0 = pref * Y1_eps1_0 + 1 +eps1_correction
    Y2_eps1_0 = pref * Y2_eps1_0 + 1 +eps1_correction

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
    plt.plot(W, Y1_eps2_0, 'b', label='L')
    plt.plot(W, Y2_eps2_0, 'g', label='R')
    plt.plot(W, Y2_eps2_0-Y1_eps2_0, 'r', label='R-L')

    # plt.plot(W, CD, 'r', label='CD')
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

    data = np.array([W, Y1_eps2_0,Y1_eps2_0])
    np.savetxt(main_class.input_folder+'CD0.dat', data.T)

    return

def calculate_epsR_epsL_eh(main_class):
    #nk,MM, ME, excited_energy, nxct, W, eta, volume
    energy_shift=main_class.energy_shift
    eps1_correction=main_class.eps1_correction

    nk = main_class.nk
    MM = main_class.MM
    ME = main_class.ME
    excited_energy = main_class.excited_energy
    nxct = main_class.nxct
    W = main_class.W
    eta = main_class.eta
    volume = main_class.volume

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
        energyDif = excited_energy[s]+energy_shift
        #E = (ME[s, 0] + 1j * ME[s, 1]) / 2.17 / 2 ** 0.5
        #E_L = (ME[s, 0] + 1j * ME[s, 1])
        E_L = (ME[s, 0]+1j * ME[s, 1])
        #M = (-1j * MM[s, 0] + MM[s, 1]) / 2 ** 0.5/(-2)/5
        M_L = (MM[s, 0]*1 + 1j * MM[s, 1])/2/nk * W/RYD/light_speed/epsilon_r*(1j)
        #Y1 += (abs(M + E)) ** 2 * np.exp(-alpha * (W - energyDif) ** 2)

        #Y1 += np.imag(E)* delta_lorentzian(W / RYD, energyDif / RYD, eta / RYD)

        #E = (ME[s, 0] - 1j * ME[s, 1]) / 2.17 / 2 ** 0.5
        #E_R = (ME[s, 0] - 1j * ME[s, 1])
        E_R = (ME[s, 0]- 1j * ME[s, 1])
        #M = (1j * MM[s, 0] + MM[s, 1]) / 2 ** 0.5/(-2)/5
        M_R = (-MM[s, 0]*1 + 1j * MM[s, 1])/2/nk * W/RYD/light_speed/epsilon_r*(1j)
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
    Y1_eps1 = pref*Y1_eps1+1 + eps1_correction
    Y2_eps1 = pref*Y2_eps1+1 + eps1_correction

    Y1_eps2_0 *= pref
    Y2_eps2_0 *= pref
    Y1_eps1_0 = pref*Y1_eps1_0+1 + eps1_correction
    Y2_eps1_0 = pref*Y2_eps1_0+1 + eps1_correction

    alpha1 = W *np.sqrt(np.sqrt(Y1_eps1**2+Y1_eps2**2)-Y1_eps1)
    alpha2 = W * np.sqrt(np.sqrt(Y2_eps1 ** 2 + Y2_eps2 ** 2) - Y2_eps1)

    alpha1_0 = W * np.sqrt(np.sqrt(Y1_eps1_0 ** 2 + Y1_eps2_0 ** 2) - Y1_eps1_0)
    alpha2_0 = W * np.sqrt(np.sqrt(Y2_eps1_0 ** 2 + Y2_eps2_0 ** 2) - Y2_eps1_0)

    R1 = calculate_reflectivity(Y1_eps1,Y1_eps2)
    R2 = calculate_reflectivity(Y2_eps1, Y2_eps2)
    R1_0 = calculate_reflectivity(Y1_eps1_0,Y1_eps2_0)
    R2_0 = calculate_reflectivity(Y2_eps1_0, Y2_eps2_0)
    #
    #dR1 = (R1-R1_0)/R1
    #dR2 = (R2-R2_0)/R2
    #dR1 = (R1-R2)/R1
    #dR2 = (R2-R1)/R2



    # alpha1 = W*Y1_eps2/np.sqrt(Y1_eps1)
    # alpha2 = W * Y2_eps2 / np.sqrt(Y2_eps1)
    # alpha1_0 = W * Y1_eps2_0 / np.sqrt(Y1_eps1_0)
    # alpha2_0 = W * Y2_eps2_0 / np.sqrt(Y1_eps1_0)

    #CD = alpha1 - alpha2 - (alpha1_0 - alpha2_0)
    #deps2 = Y1_eps2 - Y2_eps2 - (Y1_eps2_0 - Y2_eps2_0)



    # Y[1:] /= (W[1:] ** 2 * (1.6E-19) ** 2)
    # Y1[1:] /= (W[1:] ** 2 * (1.6E-19) ** 2)
    # Y2[1:] /= (W[1:] ** 2 * (1.6E-19) ** 2)
    # Y[1:] /= ((W[1:]/RYD) ** 2)
    # Y1[1:] /= ((W[1:]/RYD) ** 2)
    # Y2[1:] /= ((W[1:]/RYD) ** 2)

    deps_2_L = Y1_eps2 - Y1_eps2_0
    deps_2_R = Y2_eps2 - Y2_eps2_0
    deps_1_L = Y1_eps1 - Y1_eps1_0
    deps_1_R = Y2_eps1 - Y2_eps1_0


    plt.figure()

    #plt.plot(W, dR1, 'r', label='dR1')
    #plt.plot(W, dR2, 'g', label='dR2')
    # plt.plot(W, R1, 'r', label='R1')
    # plt.plot(W, R2, 'g', label='R2')
    # plt.plot(W, R1_0, 'b', label='R1_0')
    # plt.plot(W, R2_0, 'k', label='R2_0')



    #plt.plot(W, Y, 'r', label = 'L-R')

    # plt.plot(W, deps_1_L, 'r', label = 'deps_2_L')
    # plt.plot(W, deps_1_R, 'b', label='deps_2_R')
    plt.plot(W, Y2_eps2_0-Y1_eps2_0, 'r', label = 'R-L')
    # plt.plot(W, deps_1_L, 'r', label = 'deps_1_L')
    # plt.plot(W, deps_1_R, 'b', label = 'deps_1_R')
    #plt.plot(W, Y1_eps2_0 , 'g', label='Y1_eps2_0 ')
    #plt.plot(W, Y2_eps2_0 , 'b', label='Y2_eps2_0 ')
    # plt.plot(W, CD, 'r', label='CD')
    # plt.plot(W, alpha1_0, 'b', label='alpha1_0')
    # plt.plot(W, alpha2_0, 'g', label='alpha2_0')
    # plt.plot(W, alpha1, 'b', label='alpha1')
    # plt.plot(W, alpha2, 'g', label='alpha2')

    #plt.plot(W, Y1_eps2_0, 'b', label = 'L_eps1')
    #plt.plot(W, Y2_eps2_0, 'g', label = 'R_eps1')
    #plt.plot(W, Y1_eps1, 'k', label = 'L_eps1')
    #plt.plot(W, Y2_eps1, 'y', label = 'R_eps1')
    plt.legend()
    plt.show()

    #data = np.array([W, deps_2_L, deps_2_R])
    data = np.array([W, deps_2_L-deps_2_R])
    data2 = np.array([W,Y1_eps2_0,Y2_eps2_0])
    np.savetxt(main_class.input_folder+'CD.dat', data.T)
    np.savetxt(main_class.input_folder+'eps1_LR.dat', data2.T)


    return

def calculate_absorption_eh(main_class):
    #nk,MM, ME, excited_energy, nxct, W, eta, volume
    nk = main_class.nk
    ME = main_class.ME
    excited_energy = main_class.excited_energy
    nxct = main_class.nxct
    W = main_class.W
    eta = main_class.eta
    volume = main_class.volume
    energy_shift=main_class.energy_shift
    eps1_correction=main_class.eps1_correction


    pref = 16.0 * np.pi**2/volume/nk/main_class.spinor
    RYD = 13.6057039763
    # Y = np.zeros_like(W)
    eps_2 = np.zeros_like(W)
    eps_1 = np.zeros_like(W)
    # Y2 = np.zeros_like(W)
    for s in range(nxct):
        energyDif = excited_energy[s]+energy_shift
        #E = (ME[s, 0] + 1j * ME[s, 1]) / 2.17 / 2 ** 0.5
        E = (ME[s, 0])
        eps_2 += (abs(E)) ** 2 * delta_gauss(W/RYD, energyDif/RYD, eta/RYD)/(energyDif/RYD)**2
        eps_1 += (abs(E)) ** 2 * delta_lorentzian(W/RYD, energyDif/RYD, eta/RYD)*(energyDif-W)/eta /(energyDif/RYD)**2
        #eps_1 += -(abs(E)) ** 2 * delta_lorentzian(-W / RYD, energyDif / RYD, eta / RYD) * (energyDif + W) / eta / RYD
        #Y1 += np.imag(E)* delta_lorentzian(W / RYD, energyDif / RYD, eta / RYD)

        #E = (ME[s, 0] - 1j * ME[s, 1]) / 2.17 / 2 ** 0.5
        # E = (ME[s, 1])
        #Y2 += (abs(M + E)) ** 2 * np.exp(-alpha * (W - energyDif) ** 2)
        # Y2 += (abs(E)) ** 2 * delta_lorentzian(W/RYD, energyDif/RYD, eta/RYD)
        #Y2 += np.imag(E) ** 2 * delta_lorentzian(W / RYD, energyDif / RYD, eta / RYD)
    # Y = Y1 - Y2

    #eps_2[1:] /= ((W[1:] / RYD) ** 2)
    #eps_1[1:] /= ((W[1:] / RYD) ** 2)
    # Y *= pref
    eps_2 *= pref
    eps_1 = 1 + pref*eps_1+eps1_correction
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

    data = np.array([W, eps_1, eps_2])
    np.savetxt(main_class.input_folder+'absp.dat', data.T)

    return

def calculate_absorption_noeh(main_class):
    # noeh_dipole, nk, nv, nc, energy_dft, W, eta, volume, use_eqp=False, eqp_corr = None
    RYD = 13.6057039763
    energy_shift=main_class.energy_shift
    eps1_correction=main_class.eps1_correction

    noeh_dipole = main_class.noeh_dipole
    nk = main_class.nk
    nv = main_class.nv
    nc = main_class.nc
    energy_dft = main_class.energy_dft
    W = main_class.W
    eta = main_class.eta
    volume = main_class.volume
    use_eqp = main_class.use_eqp
    if use_eqp:
        eqp_corr = main_class.eqp_corr

    pref = 16.0 * np.pi**2/volume/nk/main_class.spinor

    #Y = np.zeros_like(W)
    eps_2 = np.zeros_like(W)
    eps_1 = np.zeros_like(W)
    #Y2 = np.zeros_like(W)

    for ik in range(nk):
        for iv in range(nv):
            for ic in range(nc):
                energyDif = energy_dft[ik,ic+nv]-energy_dft[ik,iv]+energy_shift/RYD
                if use_eqp:
                    energyDif2 =energyDif + eqp_corr[ik,nv+ic]/RYD-eqp_corr[ik,iv]/RYD
                else:
                    energyDif2 =energyDif

                eps_2 += np.abs(noeh_dipole[ik,iv,ic+nv,0])**2 * (delta_gauss(W/RYD, energyDif2, eta/RYD))/(energyDif**2)
                eps_1 += np.abs(noeh_dipole[ik, iv, ic + nv, 0]) ** 2 * (
                    delta_lorentzian(W / RYD, energyDif2, eta / RYD)) / (energyDif ** 2)*(energyDif2-W/RYD)/eta * RYD
                #Y2 += np.abs(noeh_dipole[ik, iv, ic + nv, 1]) ** 2 * (delta_lorentzian(W / RYD, energyDif,
                #                                                                      eta / RYD)-delta_lorentzian(-W/RYD, energyDif, eta/RYD))/(energyDif**2)


    #Y = Y1 - Y2

    #Y *= pref
    eps_2 *= pref
    eps_1 = pref*eps_1 + 1 + eps1_correction
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
    np.savetxt(main_class.input_folder+'absp0.dat', data.T)

    return

def calculate_m(main_class):
    #nk,MM, ME, excited_energy, nxct, W, eta, volume
    energy_shift=main_class.energy_shift
    eps1_correction=main_class.eps1_correction

    nk = main_class.nk
    MM = main_class.MM
    ME = main_class.ME
    excited_energy = main_class.excited_energy
    nxct = main_class.nxct
    W = main_class.W
    eta = main_class.eta
    volume = main_class.volume

    pref = 16.0 * np.pi ** 2 / volume / nk
    RYD = 13.6057039763
    # Y = np.zeros_like(W)
    eps_2_x_real = np.zeros_like(W)
    eps_2_y_real = np.zeros_like(W)
    eps_2_x_image = np.zeros_like(W)
    eps_2_y_image = np.zeros_like(W)
    eps_2_z = np.zeros_like(W)
    eps_2_x_z = np.zeros_like(W)

    eps_1 = np.zeros_like(W)
    # Y2 = np.zeros_like(W)
    for s in range(nxct):
        energyDif = excited_energy[s]+energy_shift
        # E = (ME[s, 0] + 1j * ME[s, 1]) / 2.17 / 2 ** 0.5
        #M = (ME[s, 2])
        #M_z = (ME[s, 2])
        M_x = (MM[s, 0])
        M_y = (MM[s, 1])
        eps_2_x_real += ((np.real(M_x)))  * delta_gauss(W / RYD, energyDif / RYD, eta / RYD) / np.sqrt(2)
        eps_2_y_real += ((np.real(M_y)))  * delta_gauss(W / RYD, energyDif / RYD, eta / RYD) / np.sqrt(2)
        eps_2_x_image += ((np.imag(M_x)))  * delta_gauss(W / RYD, energyDif / RYD, eta / RYD) / np.sqrt(2)
        eps_2_y_image += ((np.imag(M_y)))  * delta_gauss(W / RYD, energyDif / RYD, eta / RYD) / np.sqrt(2)
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

    eps_2_x_real[1:] /= ((W[1:] / RYD) ** 2)
    eps_2_y_real[1:] /= ((W[1:] / RYD) ** 2)
    eps_2_x_image[1:] /= ((W[1:] / RYD) ** 2)
    eps_2_y_image[1:] /= ((W[1:] / RYD) ** 2)
    #eps_2_z[1:] /= ((W[1:] / RYD) ** 2)
    #eps_2_x_z[1:] /= ((W[1:] / RYD) ** 2)
    # eps_1[1:] /= ((W[1:] / RYD) ** 2)
    # Y *= pref
    eps_2_x_real *= pref
    eps_2_y_real *= pref
    eps_2_x_image *= pref
    eps_2_y_image *= pref
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
    plt.plot(W, eps_2_x_real, 'r', label='eps2_x_real')
    plt.plot(W, eps_2_y_real, 'g', label='eps2_y_real')
    plt.plot(W, eps_2_x_image, 'b', label='eps2_x_image')
    plt.plot(W, eps_2_y_image, 'k', label='eps2_y_image')
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
