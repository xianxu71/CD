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



    pref = 16.0 * np.pi ** 2 / volume / nk/main_class.spinor
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
    E2 = (E_kvc[:, :, :, 0] - 1j * E_kvc[:, :, :, 1])
    M1 = (1 * L_kvc[:, :, :, 0] + 1j * L_kvc[:, :, :, 1])
    M2 = (- L_kvc[:, :, :, 0] + 1j * L_kvc[:, :, :, 1])

    # dE = np.abs(E1)**2 - np.abs(E2)**2
    # E1_2 = np.abs(E1)**2
    # E2_2 = np.abs(E2)**2
    # dE_2 = E1_2 - E2_2

    for ik in range(nk):
        for iv in range(nv): #range(nv)
            for ic in range(nc): #range(nc)
                energyDif = energy_dft[ik, ic + nv] - energy_dft[ik, iv]+energy_shift/RYD
                if use_eqp:
                    energyDif2 = energyDif + eqp_corr[ik, nv + ic] / RYD - eqp_corr[ik, iv] / RYD
                else:
                    energyDif2 = energyDif
                Y1_eps2 += np.abs(E1[ik, iv, ic] + M1[ik, iv, ic] * W / RYD / light_speed / epsilon_r) ** 2 \
                           * delta_gauss(W / RYD, energyDif2, eta / RYD)/2
                Y2_eps2 += np.abs(E2[ik, iv, ic] + M2[ik, iv, ic] * W / RYD / light_speed / epsilon_r) ** 2 \
                           * delta_gauss(W / RYD, energyDif2, eta / RYD)/2
                Y1_eps1 += np.abs(E1[ik, iv, ic] + M1[ik, iv, ic] * W / RYD / light_speed / epsilon_r) ** 2 \
                           * delta_gauss(W / RYD, energyDif2, eta / RYD)/2 * (
                                       energyDif - W / RYD) / eta
                Y2_eps1 += np.abs(E2[ik, iv, ic] + M2[ik, iv, ic] * W / RYD / light_speed / epsilon_r) ** 2 \
                           * delta_gauss(W / RYD, energyDif2, eta / RYD)/2 * (
                                       energyDif - W / RYD) / eta

                Y1_eps2_0 += np.abs(E1[ik, iv, ic]) ** 2 \
                             * delta_gauss(W / RYD, energyDif2, eta / RYD)/ 2
                Y2_eps2_0 += np.abs(E2[ik, iv, ic]) ** 2 \
                             * delta_gauss(W / RYD, energyDif2, eta / RYD)/ 2
                Y1_eps1_0 += np.abs(E1[ik, iv, ic]) ** 2 \
                             * delta_gauss(W / RYD, energyDif2, eta / RYD)/ 2 * (
                                     energyDif - W / RYD) / eta
                Y2_eps1_0 += np.abs(E2[ik, iv, ic]) ** 2 \
                             * delta_gauss(W / RYD, energyDif2, eta / RYD)/ 2 * (
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
    # Y1_eps1 = pref * Y1_eps1 + 1 +eps1_correction
    # Y2_eps1 = pref * Y2_eps1 + 1 +eps1_correction

    Y1_eps2_0 *= pref
    Y2_eps2_0 *= pref
    # Y1_eps1_0 = pref * Y1_eps1_0 + 1 +eps1_correction
    # Y2_eps1_0 = pref * Y2_eps1_0 + 1 +eps1_correction

    # alpha1 = W * np.sqrt(np.sqrt(Y1_eps1 ** 2 + Y1_eps2 ** 2) - Y1_eps1)
    # alpha2 = W * np.sqrt(np.sqrt(Y2_eps1 ** 2 + Y2_eps2 ** 2) - Y2_eps1)
    #
    # alpha1_0 = W * np.sqrt(np.sqrt(Y1_eps1_0 ** 2 + Y1_eps2_0 ** 2) - Y1_eps1_0)
    # alpha2_0 = W * np.sqrt(np.sqrt(Y2_eps1_0 ** 2 + Y2_eps2_0 ** 2) - Y2_eps1_0)

    # CD = alpha1 - alpha2 - (alpha1_0 - alpha2_0)

    # Y[1:] /= (W[1:] ** 2 * (1.6E-19) ** 2)
    # Y1[1:] /= (W[1:] ** 2 * (1.6E-19) ** 2)
    # Y2[1:] /= (W[1:] ** 2 * (1.6E-19) ** 2)
    # Y[1:] /= ((W[1:]/RYD) ** 2)
    # Y1[1:] /= ((W[1:]/RYD) ** 2)
    # Y2[1:] /= ((W[1:]/RYD) ** 2)

    plt.figure()
    plt.plot(W, (Y2_eps2 - Y1_eps2)*10, 'r', label='R-L')
    plt.plot(W, Y1_eps2, 'b', label='L')
    plt.plot(W, Y2_eps2, 'g', label='R')


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

    data = np.array([W, Y1_eps2,Y1_eps2])
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

    pref = 16.0 * np.pi**2/volume/nk/main_class.spinor
    RYD = 13.6057039763
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
    for s in range(nxct):
        energyDif = excited_energy[s]+energy_shift
        E_L = (ME[s, 0]+1j * ME[s, 1])
        M_L = (MM[s, 0]*1 + 1j * MM[s, 1])/2 * W/RYD/light_speed/epsilon_r
        E_R = (ME[s, 0]- 1j * ME[s, 1])
        M_R = (-MM[s, 0]*1 + 1j * MM[s, 1])/2 * W/RYD/light_speed/epsilon_r

        Y1_eps2 += (np.abs(M_L + E_L)) ** 2 * delta_gauss(W / RYD, energyDif / RYD, eta / RYD) /2
        Y2_eps2 += (np.abs(M_R + E_R)) ** 2 * delta_gauss(W/RYD, energyDif/RYD, eta/RYD)/2

        Y1_eps1 += (np.abs(M_L + E_L)) ** 2 * delta_lorentzian(W / RYD, energyDif / RYD, eta / RYD)/ 2  *(energyDif-W)/eta

        Y2_eps1 += (np.abs(M_R + E_R)) ** 2 * delta_lorentzian(W / RYD, energyDif / RYD, eta / RYD)/ 2 *(energyDif-W)/eta

        Y1_eps2_0 += (np.abs(E_L)) ** 2 * delta_gauss(W / RYD, energyDif / RYD, eta / RYD) / 2
        Y2_eps2_0 += (np.abs(E_R)) ** 2 * delta_gauss(W/RYD, energyDif/RYD, eta/RYD)/2

        Y1_eps1_0 += (np.abs(E_L)) ** 2 * delta_lorentzian(W / RYD, energyDif / RYD, eta / RYD)/ 2  *(energyDif-W)/eta

        Y2_eps1_0 += (np.abs(E_R)) ** 2 * delta_lorentzian(W / RYD, energyDif / RYD, eta / RYD) / 2 *(energyDif-W)/eta
        # print('test')

    Y = Y1_eps2-Y2_eps2 - (Y1_eps2_0-Y2_eps2_0)
    Y *= pref
    Y1_eps2 *= pref
    Y2_eps2 *= pref
    Y1_eps1 = pref*Y1_eps1+1 + eps1_correction
    Y2_eps1 = pref*Y2_eps1+1 + eps1_correction

    Y1_eps2_0 *= pref
    Y2_eps2_0 *= pref
    Y1_eps1_0 = pref*Y1_eps1_0+1 + eps1_correction
    Y2_eps1_0 = pref*Y2_eps1_0+1 + eps1_correction

    deps_2_L = Y1_eps2 - Y1_eps2_0
    deps_2_R = Y2_eps2 - Y2_eps2_0



    plt.figure()

    plt.plot(W, Y1_eps2+ (Y2_eps2_0 - Y1_eps2_0)/2 , 'g', label='L', linewidth=0.5)
    plt.plot(W, Y2_eps2- (Y2_eps2_0 - Y1_eps2_0)/2 , 'b', label='R', linewidth=0.5)
    plt.plot(W, ((Y2_eps2 - Y1_eps2) - (Y2_eps2_0 - Y1_eps2_0))*100, 'r', label='CD', linewidth=0.5)

    #plt.plot(W, Y1_eps2-Y1_eps2_0 , 'g', label='L', linewidth=0.5)![](../../../../../var/folders/19/wr_kxbh13vz607g6wqhwk96m0000gn/T/TemporaryItems/NSIRD_screencaptureui_O911PG/Screen Shot 2023-07-10 at 4.19.48 PM.png)
    #plt.plot(W, Y2_eps2-Y2_eps2_0  , 'b', label='R', linewidth=0.5)
    #plt.plot(W, ((Y2_eps2 - Y1_eps2) - (Y2_eps2_0 - Y1_eps2_0)), 'r',  linewidth=0.5)
    #plt.plot(W, ((Y2_eps2 - Y1_eps2) - (Y2_eps2_0 - Y1_eps2_0)), 'r', label='R-L', linewidth=0.5)
    #plt.plot(W, ((Y2_eps1 - Y1_eps1) - (Y2_eps1_0 - Y1_eps1_0)), 'k', label='R-L', linewidth=0.5)


    plt.legend()
    plt.show()
    plt.close()

    #data = np.array([W, deps_2_L, deps_2_R])
    data = np.array([W,Y1_eps2+ (Y2_eps2_0 - Y1_eps2_0)/2, ((Y2_eps2 - Y1_eps2) - (Y2_eps2_0 - Y1_eps2_0)), ((Y2_eps1 - Y1_eps1) - (Y2_eps1_0 - Y1_eps1_0))])
    #data2 = np.array([W,Y1_eps2_0,Y2_eps2_0])
    np.savetxt(main_class.input_folder+'CD.dat', data.T)

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
    #test#
    EE = ME[:,0]

    for s in range(nxct):
        energyDif = excited_energy[s]+energy_shift
        #E = (ME[s, 0] + 1j * ME[s, 1]) / 2.17 / 2 ** 0.5
        #E = (1*ME[s, 0]+1j*ME[s, 1])/np.sqrt(2)
        E = (1 * ME[s, 0] + 1j* ME[s, 1])/np.sqrt(2)
        eps_2 += (abs(E)) ** 2 * delta_gauss(W, energyDif, eta)*RYD
        eps_1 += (abs(E)) ** 2 * delta_lorentzian(W, energyDif, eta)*(energyDif-W)/eta*RYD
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

                eps_2 += np.abs(noeh_dipole[ik,iv,ic+nv,0])**2 * (delta_gauss(W/RYD, energyDif2, eta/RYD))
                eps_1 += np.abs(noeh_dipole[ik, iv, ic + nv, 0]) ** 2 * (
                    delta_lorentzian(W / RYD, energyDif2, eta / RYD))*(energyDif2-W/RYD)/eta * RYD
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

def calculate_m_eh(main_class):
    energy_shift = main_class.energy_shift
    eps1_correction = main_class.eps1_correction

    nk = main_class.nk
    MM = main_class.MM
    excited_energy = main_class.excited_energy
    nxct = main_class.nxct
    W = main_class.W
    eta = main_class.eta
    volume = main_class.volume

    pref = 16.0 * np.pi ** 2 / volume / nk / main_class.spinor
    RYD = 13.6057039763
    light_speed = 137
    epsilon_r = 1
    Y1_eps2 = np.zeros_like(W)
    Y2_eps2 = np.zeros_like(W)
    for s in range(nxct):
        energyDif = excited_energy[s] + energy_shift
        M_0 = (MM[s, 0]) / 2 * W / RYD / light_speed / epsilon_r
        M_1 = (MM[s, 1]) / 2 * W / RYD / light_speed / epsilon_r
        Y1_eps2 += (np.abs(M_0+1j*M_1)) * delta_gauss(W / RYD, energyDif / RYD, eta / RYD) / 2
        Y2_eps2 += (np.abs(M_1)) * delta_gauss(W / RYD, energyDif / RYD, eta / RYD) / 2

    Y1_eps2 *= pref
    Y2_eps2 *= pref


    plt.figure()

    plt.plot(W, Y1_eps2, 'b', label='0')
    #plt.plot(W, Y2_eps2, 'g', label='1')
    #plt.plot(W, ((Y2_eps2 - Y1_eps2)) * 10, 'r', label='R-L')
    plt.legend()
    plt.show()

    return
def calculate_m_noeh(main_class):
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



    pref = 16.0 * np.pi ** 2 / volume / nk/main_class.spinor
    RYD = 13.6057039763  # W=W/RYD
    light_speed = 137
    epsilon_r = 1
    Y = np.zeros_like(W)
    Y1_eps2 = np.zeros_like(W)
    Y2_eps2 = np.zeros_like(W)

    M1 = (L_kvc[:, :, :, 0])
    M2 = (L_kvc[:, :, :, 1])

    # dE = np.abs(E1)**2 - np.abs(E2)**2
    # E1_2 = np.abs(E1)**2
    # E2_2 = np.abs(E2)**2
    # dE_2 = E1_2 - E2_2

    for ik in range(nk):
        for iv in range(nv): #range(nv)
            for ic in range(nc): #range(nc)
                energyDif = energy_dft[ik, ic + nv] - energy_dft[ik, iv]+energy_shift/RYD
                if use_eqp:
                    energyDif2 = energyDif + eqp_corr[ik, nv + ic] / RYD - eqp_corr[ik, iv] / RYD
                else:
                    energyDif2 = energyDif
                Y1_eps2 += np.abs(M1[ik, iv, ic] * W / RYD / light_speed / epsilon_r) ** 2 \
                           * delta_gauss(W / RYD, energyDif2, eta / RYD)/2
                Y2_eps2 += np.abs(M2[ik, iv, ic] * W / RYD / light_speed / epsilon_r) ** 2 \
                           * delta_gauss(W / RYD, energyDif2, eta / RYD)/2


    Y = Y1_eps2 - Y2_eps2
    Y *= pref
    Y1_eps2 *= pref
    Y2_eps2 *= pref


    plt.figure()
    plt.plot(W, (Y2_eps2 - Y1_eps2), 'r', label='diff')
    plt.plot(W, Y1_eps2, 'b', label='0')
    plt.plot(W, Y2_eps2, 'g', label='1')

    plt.legend()
    plt.show()

    data = np.array([W, Y1_eps2,Y1_eps2])
    np.savetxt(main_class.input_folder+'CD0.dat', data.T)

    return