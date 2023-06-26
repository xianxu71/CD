import numpy as np
import h5py as h5

def calculate_diag_L(main_class):



    f = h5.File(main_class.input_folder+'eigenvectors.h5','r')
    rk = a = f['mf_header/kpoints/rk'][()]
    use_k = np.zeros(main_class.nk)
    threshhold = 0.15
    kx0 = 0.333333
    ky0 = 0.333333
    kz0 = 0
    for i in range(main_class.nk):
        kx = rk[i,0]
        ky = rk[i,1]
        kz = rk[i,2]
        if kx > 0.5:
            kx = kx
        if ky > 0.5:
            ky = ky
        if kz > 0.5:
            kz = kz
        dd = (kx-kx0)**2+(ky-ky0)**2+(kz-kz0)**2
        if dd < threshhold**2:
            use_k[i] = 1

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
    totx2 = np.abs(totx)[:,0:4]
    toty2 = np.abs(toty)[:,0:4]
    totz2 = np.abs(totz)[:,0:4]
    for i in range(main_class.nk):
        totx2[i, :] = totx2[i, :] * use_k[i]
        toty2[i, :] = toty2[i, :] * use_k[i]
        totz2[i,:] = totz2[i,:]*use_k[i]


    x_diag = np.zeros([main_class.nk, main_class.nv_for_r + main_class.nc_for_r])
    y_diag = np.zeros([main_class.nk, main_class.nv_for_r + main_class.nc_for_r])
    z_diag = np.zeros([main_class.nk, main_class.nv_for_r + main_class.nc_for_r])

    for i in range(main_class.nk):
        for j in range(main_class.nv_for_r + main_class.nc_for_r):
            x_diag[i, j] = np.abs(datax[i,j,j])
            y_diag[i, j] = np.abs(datay[i,j,j])
            z_diag[i, j] = np.abs(dataz[i, j, j])
    x_diag = np.abs(x_diag[:,0:4])
    y_diag = np.abs(y_diag[:, 0:4])
    z_diag = np.abs(z_diag[:, 0:4])
    #
    for i in range(main_class.nk):
        x_diag[i, :] = x_diag[i, :] * use_k[i]
        y_diag[i, :] = y_diag[i, :] * use_k[i]
        z_diag[i,:] = z_diag[i,:]*use_k[i]


    print('finish testing L_kvc')

    return 0