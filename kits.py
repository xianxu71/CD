import numpy as np


def vc_contribution(main_class):
    '''
    This script decomposes each exciton solution into the bands that participate in the v->c transisions
    '''
    dipole2_x = np.abs(main_class.ME[:,0])**2
    dipole2_y = np.abs(main_class.ME[:, 1]) ** 2
    dipole2_L = np.abs(main_class.ME[:, 0]+1j*main_class.ME[:, 1]) ** 2/2
    dipole2_R = np.abs(main_class.ME[:, 0] - 1j * main_class.ME[:, 1]) ** 2 / 2

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

    return 0