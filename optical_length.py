from math_function import *
import matplotlib.pyplot as plt

def calculate_absorption_noeh(main_class):
    nb = main_class.nv + main_class.nc
    energy_dft = main_class.energy_dft
    nk = main_class.nk
    nv = main_class.nv
    nc = main_class.nc
    eta = main_class.eta
    W = main_class.W
    volume = main_class.volume
    RYD = 13.6057039763
    pref = 16.0 * np.pi ** 2 / volume / nk / main_class.spinor
    a2bohr = 1.88973

    eps_2 = np.zeros_like(W)
    eps_1 = np.zeros_like(W)

    E = main_class.E_kvc
    for ic in range(nc):
        for iv in range(nv):
            for ik in range(nk):
                energydif = energy_dft[ik,ic+nv]-energy_dft[ik,iv]
                eps_2 += np.abs(E[ik, iv, ic,0])**2 * (delta_gauss(W/RYD, energydif, eta/RYD))
                eps_1 += np.abs(E[ik, iv, ic, 0]) ** 2 * (delta_lorentzian(W / RYD, energydif, eta / RYD))\
                         *(energydif-W/RYD)/eta * RYD
    eps_2 *= pref
    eps_1 = pref * eps_1 + 1
    plt.figure()
    # plt.plot(W, Y, 'r')
    plt.plot(W, eps_2, 'b')
    plt.plot(W, eps_1, 'r')
    # plt.plot(W, Y2, 'g')
    plt.show()

def calculate_cd_noeh(main_class):
    nb = main_class.nv + main_class.nc
    energy_dft = main_class.energy_dft
    nk = main_class.nk
    nv = main_class.nv
    nc = main_class.nc
    eta = main_class.eta
    W = main_class.W
    E_kvc = main_class.E_kvc
    L_kvc = main_class.L_kvc*100
    volume = main_class.volume
    RYD = 13.6057039763
    pref = 16.0 * np.pi ** 2 / volume / nk / main_class.spinor
    RYD = 13.6057039763  # W=W/RYD
    light_speed = 274
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



    E1 = (E_kvc[:, :, :, 0] + 1j * E_kvc[:, :, :, 1])
    E2 = (E_kvc[:, :, :, 0] - 1j * E_kvc[:, :, :, 1])
    M1 = (1 * L_kvc[:, :, :, 0] + 1j * L_kvc[:, :, :, 1])
    M2 = (- L_kvc[:, :, :, 0] + 1j * L_kvc[:, :, :, 1])



    for ic in range(nc):
        for iv in range(nv):
            for ik in range(nk):
                energydif = energy_dft[ik,ic+nv]-energy_dft[ik,iv]
                Y1_eps2 += np.abs(E1[ik, iv, ic] + M1[ik, iv, ic] * W / RYD / light_speed / epsilon_r) ** 2 \
                           * delta_gauss(W / RYD, energydif, eta / RYD) / 2
                Y2_eps2 += np.abs(E2[ik, iv, ic] + M2[ik, iv, ic] * W / RYD / light_speed / epsilon_r) ** 2 \
                           * delta_gauss(W / RYD, energydif, eta / RYD) / 2
                Y1_eps1 += np.abs(E1[ik, iv, ic] + M1[ik, iv, ic] * W / RYD / light_speed / epsilon_r) ** 2 \
                           * delta_gauss(W / RYD, energydif, eta / RYD) / 2 * (
                                   energydif - W / RYD) / eta
                Y2_eps1 += np.abs(E2[ik, iv, ic] + M2[ik, iv, ic] * W / RYD / light_speed / epsilon_r) ** 2 \
                           * delta_gauss(W / RYD, energydif, eta / RYD) / 2 * (
                                   energydif - W / RYD) / eta

                Y1_eps2_0 += np.abs(E1[ik, iv, ic]) ** 2 \
                             * delta_gauss(W / RYD, energydif, eta / RYD) / 2
                Y2_eps2_0 += np.abs(E2[ik, iv, ic]) ** 2 \
                             * delta_gauss(W / RYD, energydif, eta / RYD) / 2
                Y1_eps1_0 += np.abs(E1[ik, iv, ic]) ** 2 \
                             * delta_gauss(W / RYD, energydif, eta / RYD) / 2 * (
                                     energydif - W / RYD) / eta
                Y2_eps1_0 += np.abs(E2[ik, iv, ic]) ** 2 \
                             * delta_gauss(W / RYD, energydif, eta / RYD) / 2 * (
                                     energydif - W / RYD) / eta
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
    plt.figure()
    # plt.plot(W, Y, 'r')
    plt.plot(W, Y1_eps2, 'b', label='L')
    plt.plot(W, Y2_eps2, 'g', label='R')
    plt.plot(W, (Y2_eps2 - Y1_eps2), 'r', label='R-L')
    plt.legend()
    # plt.plot(W, Y2, 'g')
    plt.show()




def calculate_absorption_eh(main_class):
    nk = main_class.nk
    ME = main_class.ME
    excited_energy = main_class.excited_energy
    nxct = main_class.nxct
    W = main_class.W
    eta = main_class.eta
    volume = main_class.volume

    pref = 16.0 * np.pi ** 2 / volume / nk / main_class.spinor
    RYD = 13.6057039763
    eps_2 = np.zeros_like(W)
    eps_1 = np.zeros_like(W)

    for s in range(nxct):
        energyDif = excited_energy[s]
        # E = (ME[s, 0] + 1j * ME[s, 1]) / 2.17 / 2 ** 0.5
        E = (ME[s, 0])
        eps_2 += (abs(E)) ** 2 * delta_gauss(W, energyDif, eta) * RYD
        eps_1 += (abs(E)) ** 2 * delta_lorentzian(W, energyDif, eta) * (energyDif - W) / eta * RYD
    eps_2 *= pref
    eps_1 = 1 + pref * eps_1
    plt.figure()
    # plt.plot(W, Y, 'r')
    plt.plot(W, eps_2, 'b', label='eps2')
    plt.plot(W, eps_1, 'r', label='eps1')
    # plt.plot(W, Y2, 'g')
    plt.legend()
    plt.show()

    return



