import numpy as np


def vc_contribution(main_class):
    '''
    This script decomposes each exciton solution into the bands that participate in the v->c transisions
    '''
    ME = main_class.ME
    MM = main_class.MM

    dipole2_x = np.abs(main_class.ME[:,0])**2
    dipole2_y = np.abs(main_class.ME[:, 1]) ** 2
    dipole2_L = np.abs(main_class.ME[:, 0]+1j*main_class.ME[:, 1]) ** 2/2
    dipole2_R = np.abs(main_class.ME[:, 0] - 1j * main_class.ME[:, 1]) ** 2 / 2

    E_L = (ME[:, 0] + 1j * ME[:, 1])
    M_L = (MM[:, 0] * 1 + 1j * MM[:, 1]) / 10000
    E_R = (ME[:, 0] - 1j * ME[:, 1])
    M_R = (-MM[:, 0] * 1 + 1j * MM[:, 1]) / 10000

    CD = (np.abs(E_L+M_L)**2 - np.abs(E_R+M_R)**2)-(np.abs(E_L)**2 - np.abs(E_R)**2)

    excited_energy = main_class.excited_energy + main_class.energy_shift
    Akvcs = main_class.avck

    c_res0 = np.zeros([main_class.nc*main_class.nxct,3])
    v_res0 = np.zeros([main_class.nv * main_class.nxct, 3])
    c_resx = np.zeros([main_class.nc*main_class.nxct,3])
    v_resx = np.zeros([main_class.nv * main_class.nxct, 3])
    c_resy = np.zeros([main_class.nc*main_class.nxct,3])
    v_resy = np.zeros([main_class.nv * main_class.nxct, 3])
    c_resL = np.zeros([main_class.nc*main_class.nxct,3])
    v_resL = np.zeros([main_class.nv * main_class.nxct, 3])
    c_resR = np.zeros([main_class.nc*main_class.nxct,3])
    v_resR = np.zeros([main_class.nv * main_class.nxct, 3])

    c_rescd = np.zeros([main_class.nv * main_class.nxct, 3])
    v_rescd = np.zeros([main_class.nv * main_class.nxct, 3])



    for ixct in range(main_class.nxct):
        temp_contrib_vc = np.sum(abs(Akvcs[:,:,:,ixct]) ** 2,axis=0)  # (ic,iv)
        c_list = np.arange(1,main_class.nc+1,1)
        v_list = np.arange(-1, -main_class.nv - 1, -1)
        ex_c_list = np.ones(main_class.nc) * excited_energy[ixct]
        ex_v_list = np.ones(main_class.nv) * excited_energy[ixct]

        temp_contrib_v_0 = np.sum(temp_contrib_vc, axis=1)
        temp_contrib_c_0 = np.sum(temp_contrib_vc, axis=0)

        temp_contrib_v_x = np.sum(temp_contrib_vc, axis=1) * dipole2_x[ixct]
        temp_contrib_c_x = np.sum(temp_contrib_vc, axis=0) * dipole2_x[ixct]

        temp_contrib_v_y = np.sum(temp_contrib_vc, axis=1) * dipole2_y[ixct]
        temp_contrib_c_y = np.sum(temp_contrib_vc, axis=0) * dipole2_y[ixct]

        temp_contrib_v_L = np.sum(temp_contrib_vc, axis=1) * dipole2_L[ixct]
        temp_contrib_c_L = np.sum(temp_contrib_vc, axis=0) * dipole2_L[ixct]

        temp_contrib_v_R = np.sum(temp_contrib_vc, axis=1) * dipole2_R[ixct]
        temp_contrib_c_R = np.sum(temp_contrib_vc, axis=0) * dipole2_R[ixct]

        temp_contrib_v_cd = np.sum(temp_contrib_vc, axis=1) * CD[ixct]
        temp_contrib_c_cd = np.sum(temp_contrib_vc, axis=0) * CD[ixct]

        temp_c_res_0 = np.vstack((ex_c_list, c_list, temp_contrib_c_0)).T
        temp_v_res_0 = np.vstack((ex_v_list, v_list, temp_contrib_v_0)).T

        temp_c_res_x = np.vstack((ex_c_list, c_list, temp_contrib_c_x)).T
        temp_v_res_x = np.vstack((ex_v_list, v_list, temp_contrib_v_x)).T

        temp_c_res_y = np.vstack((ex_c_list, c_list, temp_contrib_c_y)).T
        temp_v_res_y = np.vstack((ex_v_list, v_list, temp_contrib_v_y)).T

        temp_c_res_L = np.vstack((ex_c_list, c_list, temp_contrib_c_L)).T
        temp_v_res_L = np.vstack((ex_v_list, v_list, temp_contrib_v_L)).T

        temp_c_res_R = np.vstack((ex_c_list, c_list, temp_contrib_c_R)).T
        temp_v_res_R = np.vstack((ex_v_list, v_list, temp_contrib_v_R)).T

        temp_c_res_cd = np.vstack((ex_c_list, c_list, temp_contrib_c_cd)).T
        temp_v_res_cd = np.vstack((ex_v_list, v_list, temp_contrib_v_cd)).T

        c_res0[ixct * main_class.nc:(ixct + 1) * main_class.nc, :] = temp_c_res_0
        v_res0[ixct * main_class.nv:(ixct + 1) * main_class.nv, :] = temp_v_res_0

        c_resx[ixct * main_class.nc:(ixct + 1) * main_class.nc, :] = temp_c_res_x
        v_resx[ixct * main_class.nv:(ixct + 1) * main_class.nv, :] = temp_v_res_x

        c_resy[ixct * main_class.nc:(ixct + 1) * main_class.nc, :] = temp_c_res_y
        v_resy[ixct * main_class.nv:(ixct + 1) * main_class.nv, :] = temp_v_res_y

        c_resL[ixct * main_class.nc:(ixct + 1) * main_class.nc, :] = temp_c_res_L
        v_resL[ixct * main_class.nv:(ixct + 1) * main_class.nv, :] = temp_v_res_L

        c_resR[ixct * main_class.nc:(ixct + 1) * main_class.nc, :] = temp_c_res_R
        v_resR[ixct * main_class.nv:(ixct + 1) * main_class.nv, :] = temp_v_res_R

        c_rescd[ixct * main_class.nc:(ixct + 1) * main_class.nc, :] = temp_c_res_cd
        v_rescd[ixct * main_class.nv:(ixct + 1) * main_class.nv, :] = temp_v_res_cd

    np.savetxt(main_class.input_folder+'con_c_0.dat', c_res0)
    np.savetxt(main_class.input_folder+'con_v_0.dat', v_res0)

    np.savetxt(main_class.input_folder+'con_c_x.dat', c_resx)
    np.savetxt(main_class.input_folder+'con_v_x.dat', v_resx)

    np.savetxt(main_class.input_folder+'con_c_y.dat', c_resy)
    np.savetxt(main_class.input_folder+'con_v_y.dat', v_resy)

    np.savetxt(main_class.input_folder+'con_c_L.dat', c_resL)
    np.savetxt(main_class.input_folder+'con_v_L.dat', v_resL)

    np.savetxt(main_class.input_folder+'con_c_R.dat', c_resR)
    np.savetxt(main_class.input_folder+'con_v_R.dat', v_resR)

    np.savetxt(main_class.input_folder + 'con_c_cd.dat', c_rescd)
    np.savetxt(main_class.input_folder + 'con_v_cd.dat', v_rescd)

    return 0

def dipole_k_L(main_class):
    idx = list(range(main_class.nv - 1, -1, -1))

    inds = np.ix_(range(main_class.nk), idx, range(main_class.nc), range(3))
    E_temp = main_class.E_kvc[inds]
    L_temp = main_class.L_kvc[inds]
    acvk = main_class.avck

    acvk0 = acvk[:,0,0,0]
    kpt = main_class.rk

    exc_EL = open('phase_EL_wfn_1-1.dat' , 'w')
    exc_LL = open('phase_LL_wfn_1-1.dat', 'w')


    EL = (E_temp[:,0,0,0]+1j*E_temp[:,0,0,1])*(acvk0)/np.abs(acvk0)
    EL_real = np.real(EL)
    EL_imag = np.imag(EL)

    for i in range(main_class.nk):
        exc_EL.write(str(kpt[i][0]) + ' ' + str(kpt[i][1]) + ' ' + str(kpt[i][2]) + ' ' + str(EL_real[i]) + ' ' + str(
            EL_imag[i]) + '\n')
        print("kpt:", i, '/', main_class.nk, '  EL(k):', EL_real[i])

    LL = (L_temp[:,0,0,0]+1j*L_temp[:,0,0,1])*(acvk0)/np.abs(acvk0)
    LL_real = np.real(LL)
    LL_imag = np.imag(LL)

    for i in range(main_class.nk):
        exc_LL.write(str(kpt[i][0]) + ' ' + str(kpt[i][1]) + ' ' + str(kpt[i][2]) + ' ' + str(LL_real[i]) + ' ' + str(
            LL_imag[i]) + '\n')
        print("kpt:", i, '/', main_class.nk, '  LL(k):', LL_real[i])


    exc_EL.close()
    exc_LL.close()
    # exc_EL2 = open('phase_EL2_wfn_exciton_3.dat', 'w')
    # exc_LL2 = open('phase_LL2_wfn_exciton_3.dat', 'w')
    # exc_ELLL2 = open('phase_ELLL2_wfn_exciton_3.dat', 'w')
    #
    # acvk0 = acvk[:, 0, 0, 0]
    # acvk1 = acvk[:, 0, 0, 2]
    #
    #
    # EL2 = (E_temp[:, 0, 0, 0] + 1j * E_temp[:, 0, 0, 1]) * (acvk1)
    # EL_real2 = np.real(EL2)
    # EL_imag2 = np.imag(EL2)
    #
    # for i in range(main_class.nk):
    #     exc_EL2.write(str(kpt[i][0]) + ' ' + str(kpt[i][1]) + ' ' + str(kpt[i][2]) + ' ' + str(EL_real2[i]) + ' ' + str(
    #         EL_imag2[i]) + '\n')
    #     print("kpt:", i, '/', main_class.nk, '  EL(k):', EL_real2[i])
    #
    # LL2 = (L_temp[:, 0, 0, 0] + 1j * L_temp[:, 0, 0, 1]) * (acvk1)
    # LL_real2 = np.real(LL2)
    # LL_imag2 = np.imag(LL2)
    #
    # for i in range(main_class.nk):
    #     exc_LL2.write(str(kpt[i][0]) + ' ' + str(kpt[i][1]) + ' ' + str(kpt[i][2]) + ' ' + str(LL_real2[i]) + ' ' + str(
    #         LL_imag2[i]) + '\n')
    #     print("kpt:", i, '/', main_class.nk, '  LL(k):', LL_real2[i])
    #
    # ELLL2 = np.real(np.abs(EL2+LL2/10000)**2 - np.abs(EL2)**2 -np.abs(LL2/10000)**2)
    #
    # for i in range(main_class.nk):
    #     exc_ELLL2.write(str(kpt[i][0]) + ' ' + str(kpt[i][1]) + ' ' + str(kpt[i][2]) + ' ' + str(
    #         ELLL2[i]) + '\n')
    #     print("kpt:", i, '/', main_class.nk, '  EL(k):', ELLL2[i])
    # exc_ELLL2.close()
    #
    # diff = np.real(np.abs(np.sum(EL2 + LL2 / 10000)) ** 2 - np.abs(np.sum(EL2)) ** 2 - np.abs(np.sum(LL2 / 10000)) ** 2)
    # print(diff)
    #
    # exc_EL2.close()
    # exc_LL2.close()

    return 0
def dipole_k_R(main_class):
    idx = list(range(main_class.nv - 1, -1, -1))

    inds = np.ix_(range(main_class.nk), idx, range(main_class.nc), range(3))
    E_temp = main_class.E_kvc[inds]
    L_temp = main_class.L_kvc[inds]
    acvk = main_class.avck

    acvk0 = acvk[:,0,0,0]
    kpt = main_class.rk

    exc_ER = open('phase_ER_wfn_1-1.dat' , 'w')
    exc_LR = open('phase_LR_wfn_1-1.dat', 'w')


    ER = (E_temp[:,0,0,0]-1j*E_temp[:,0,0,1])*(acvk0)/np.abs(acvk0)
    ER_real = np.real(ER)
    ER_imag = np.imag(ER)

    for i in range(main_class.nk):
        exc_ER.write(str(kpt[i][0]) + ' ' + str(kpt[i][1]) + ' ' + str(kpt[i][2]) + ' ' + str(ER_real[i]) + ' ' + str(
            ER_imag[i]) + '\n')
        print("kpt:", i, '/', main_class.nk, '  ER(k):', ER_real[i])

    LR = (-L_temp[:,0,0,0]+1j*L_temp[:,0,0,1])*(acvk0)/np.abs(acvk0)
    LR_real = np.real(LR)
    LR_imag = np.imag(LR)

    for i in range(main_class.nk):
        exc_LR.write(str(kpt[i][0]) + ' ' + str(kpt[i][1]) + ' ' + str(kpt[i][2]) + ' ' + str(LR_real[i]) + ' ' + str(
            LR_imag[i]) + '\n')
        print("kpt:", i, '/', main_class.nk, '  LR(k):', LR_real[i])


    exc_ER.close()
    exc_LR.close()

def dipole_k_L2(main_class):
    E_temp = main_class.ME_k
    L_temp = main_class.MM_k
    kpt = main_class.rk


    EL = (E_temp[0,:,0]+1j*E_temp[0,:,1])
    EL_real = np.real(EL)
    EL_imag = np.imag(EL)

    exc_EL = open('phase_EL_wfn_allncnv_ex1.dat', 'w')
    exc_LL = open('phase_LL_wfn_allncnv_ex1.dat', 'w')

    for i in range(main_class.nk):
        exc_EL.write(str(kpt[i][0]) + ' ' + str(kpt[i][1]) + ' ' + str(kpt[i][2]) + ' ' + str(EL_real[i]) + ' ' + str(
            EL_imag[i]) + '\n')
        print("kpt:", i, '/', main_class.nk, '  EL(k):', EL_real[i])

    LL = (L_temp[0,:,0]+1j*L_temp[0,:,1])
    LL_real = np.real(LL)
    LL_imag = np.imag(LL)

    for i in range(main_class.nk):
        exc_LL.write(str(kpt[i][0]) + ' ' + str(kpt[i][1]) + ' ' + str(kpt[i][2]) + ' ' + str(LL_real[i]) + ' ' + str(
            LL_imag[i]) + '\n')
        print("kpt:", i, '/', main_class.nk, '  LL(k):', LL_real[i])

    def dipole_k_L2(main_class):
        E_temp = main_class.ME_k
        L_temp = main_class.MM_k
        kpt = main_class.rk

        EL = (E_temp[0, :, 0] + 1j * E_temp[0, :, 1])
        EL_real = np.real(EL)
        EL_imag = np.imag(EL)

        exc_EL = open('phase_EL_wfn_allncnv_ex1.dat', 'w')
        exc_LL = open('phase_LL_wfn_allncnv_ex1.dat', 'w')

        for i in range(main_class.nk):
            exc_EL.write(
                str(kpt[i][0]) + ' ' + str(kpt[i][1]) + ' ' + str(kpt[i][2]) + ' ' + str(EL_real[i]) + ' ' + str(
                    EL_imag[i]) + '\n')
            print("kpt:", i, '/', main_class.nk, '  EL(k):', EL_real[i])

        LL = (L_temp[0, :, 0] + 1j * L_temp[0, :, 1])
        LL_real = np.real(LL)
        LL_imag = np.imag(LL)

        for i in range(main_class.nk):
            exc_LL.write(
                str(kpt[i][0]) + ' ' + str(kpt[i][1]) + ' ' + str(kpt[i][2]) + ' ' + str(LL_real[i]) + ' ' + str(
                    LL_imag[i]) + '\n')
            print("kpt:", i, '/', main_class.nk, '  LL(k):', LL_real[i])




    exc_EL.close()
    exc_LL.close()
    return 0


def dipole_k_xy(main_class):
    idx = list(range(main_class.nv - 1, -1, -1))

    inds = np.ix_(range(main_class.nk), idx, range(main_class.nc), range(3))
    E_temp = main_class.E_kvc[inds]
    L_temp = main_class.L_kvc[inds]
    acvk = main_class.avck

    acvk0 = acvk[:, 0, 0, 0]
    acvk_temp = acvk[:, 0, 0, 2]
    acvk_temp = acvk_temp * np.conj(acvk0) / np.abs(acvk0)
    evecs_real = np.real(acvk_temp)
    evecs_imag = np.imag(acvk_temp)
    kpt = main_class.rk

    exc = open('phase_exc_wfn_useless_2-2.dat', 'w')
    exc_Ex = open('phase_Ex_wfn.dat', 'w')
    exc_Lx = open('phase_Lx_wfn.dat', 'w')


    for i in range(main_class.nk):
        exc.write(
            str(kpt[i][0]) + ' ' + str(kpt[i][1]) + ' ' + str(kpt[i][2]) + ' ' + str(evecs_real[i]) + ' ' + str(
                evecs_imag[i]) + '\n')
        print("kpt:", i, '/', main_class.nk, '  A(k):', evecs_real[i])

    Ex = E_temp[:, 0, 0, 0] * (acvk0) / np.abs(acvk0)
    Ex_real = np.real(Ex)
    Ex_imag = np.imag(Ex)

    for i in range(main_class.nk):
        exc_Ex.write(
            str(kpt[i][0]) + ' ' + str(kpt[i][1]) + ' ' + str(kpt[i][2]) + ' ' + str(Ex_real[i]) + ' ' + str(
                Ex_imag[i]) + '\n')
        print("kpt:", i, '/', main_class.nk, '  Ex(k):', Ex_real[i])

    Lx = L_temp[:, 0, 0, 0] * (acvk0) / np.abs(acvk0)
    Lx_real = np.real(Lx)
    Lx_imag = np.imag(Lx)

    for i in range(main_class.nk):
        exc_Lx.write(
            str(kpt[i][0]) + ' ' + str(kpt[i][1]) + ' ' + str(kpt[i][2]) + ' ' + str(Lx_real[i]) + ' ' + str(
                Lx_imag[i]) + '\n')
        print("kpt:", i, '/', main_class.nk, '  Lx(k):', Lx_real[i])

    exc_Ex.close()
    exc_Lx.close()
    exc.close()

    exc_Ey = open('phase_Ey_wfn.dat', 'w')
    exc_Ly = open('phase_Ly_wfn.dat', 'w')

    Ey = E_temp[:, 0, 0, 1] * (acvk0) / np.abs(acvk0)
    Ey_real = np.real(Ey)
    Ey_imag = np.imag(Ey)

    for i in range(main_class.nk):
        exc_Ey.write(
            str(kpt[i][0]) + ' ' + str(kpt[i][1]) + ' ' + str(kpt[i][2]) + ' ' + str(Ey_real[i]) + ' ' + str(
                Ey_imag[i]) + '\n')
        print("kpt:", i, '/', main_class.nk, '  Ey(k):', Ey_real[i])

    Ly = L_temp[:, 0, 0, 1] * (acvk0) / np.abs(acvk0)
    Ly_real = np.real(Ly)
    Ly_imag = np.imag(Ly)

    for i in range(main_class.nk):
        exc_Ly.write(
            str(kpt[i][0]) + ' ' + str(kpt[i][1]) + ' ' + str(kpt[i][2]) + ' ' + str(Ly_real[i]) + ' ' + str(
                Ly_imag[i]) + '\n')
        print("kpt:", i, '/', main_class.nk, '  Ly(k):', Ly_real[i])

    exc_Ey.close()
    exc_Ly.close()
    return 0

