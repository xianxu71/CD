import numpy as np

def calculate_diag_L(main_class):
    fact = 1
    datax = main_class.noeh_dipole_full[:, :, :, 0]
    datay = main_class.noeh_dipole_full[:, :, :, 1]
    dataz = main_class.noeh_dipole_full[:, :, :, 2]


    Ekv = np.einsum('kb,m->kbm', main_class.energy_dft_full,
                    np.ones(main_class.nv_for_r + main_class.nc_for_r))
    Ekm = np.einsum('km,b->kbm', main_class.energy_dft_full, np.ones(main_class.nv_for_r + main_class.nc_for_r))

    energy_diff = (Ekv - Ekm)  # e_diff(k,v,m) = [E(k,v) - E(k,m)]^-1
    with np.errstate(divide='ignore'):
        energy_diff_inverse = 1 / energy_diff
        energy_diff_inverse[abs(energy_diff) < main_class.degeneracy_remover] = 0

    totx = np.einsum('kbm,kbm,kmb-> kb', datay, energy_diff_inverse,
                     dataz) - \
           np.einsum('kbm,kbm,kmb-> kb', dataz, energy_diff_inverse,
                     datay)
    toty = np.einsum('kbm,kbm,kmb-> kb', dataz, energy_diff_inverse,
                     datax) - \
           np.einsum('kbm,kbm,kmb-> kb', datax, energy_diff_inverse,
                     dataz)
    totz = np.einsum('kbm,kbm,kmb-> kb', datax, energy_diff_inverse,
                     datay) - \
           np.einsum('kbm,kbm,kmb-> kb', datay, energy_diff_inverse,
                     datax)
    print('finish testing L_kvc')

    return 0