import numpy as np
import h5py as h5

class electromagnetic:
    def __init__(self, main_class):
        if main_class.read_temp:
            h5_file_r = h5.File(main_class.input_folder+'temp.h5', 'r')
            main_class.L_kvc = np.array(h5_file_r.get('L_kvc'))
            main_class.E_kvc = np.array(h5_file_r.get('E_kvc'))
            main_class.MM = np.array(h5_file_r.get('MM'))
            main_class.ME = np.array(h5_file_r.get('ME'))
            h5_file_r.close()
            print('read temp matrices to temp.h5')
        else:
            main_class.L_kvc = self.calculate_L(main_class) #calculated orbital angular momentum
            main_class.E_kvc = self.calculate_E(main_class) #calculated electric dipole
            main_class.MM, main_class.ME = self.calculate_MM_ME(main_class) #calculate orbital angular momentum and electric dipole with exciton representation



    def calculate_E(self,main_class):
        fact = 1  # test
        E = main_class.noeh_dipole[:, 0:main_class.nv, main_class.nv:main_class.nc + main_class.nv, :] * fact
        print('finish calculating E_kvc')
        return E

    def calculate_MM_ME(self,main_class):
        idx = list(range(main_class.nv - 1, -1, -1))

        inds = np.ix_(range(main_class.nk), idx, range(main_class.nv), range(3))
        E_temp = main_class.E_kvc[inds]
        L_temp = main_class.L_kvc[inds]

        MM = np.einsum('kvcs,kvcd->sd', main_class.avck, L_temp)
        ME = np.einsum('kvcs,kvcd->sd', main_class.avck, E_temp)
        print('finish calculating MM ME')
        return MM, ME

    def calculate_L(self, main_class):
        '''
        calculate orbital angular momentum
        '''
        '''
            # dim [nk,nv,nc,3], orbital angular momentum
            calculate orbital angular momentum

            noeh_dipole: dim: [nk, nb, nb, 3]

        '''

        # m0 = 9.10938356e-31 # mass of electrons
        # eVtoJ = 1.602177e-19 # electron volt = ？Joule
        # p_ry_to_SI = 1.9928534e-24 # convert momentum operator to SI unit # ??
        # fact = (p_ry_to_SI ** 2) / m0 / eVtoJ # pre factor
        fact = 1
        datax = main_class.noeh_dipole_full[:, :, :, 0]
        datay = main_class.noeh_dipole_full[:, :, :, 1]
        dataz = main_class.noeh_dipole_full[:, :, :, 2]

        L = np.zeros([main_class.nk, main_class.nv_for_r, main_class.nc_for_r, 3], dtype=np.complex)

        Ekv = np.einsum('kv,m->kvm', main_class.energy_dft_full[:, 0:main_class.nv_for_r], np.ones(main_class.nv_for_r + main_class.nc_for_r))
        Ekm = np.einsum('km,v->kvm', main_class.energy_dft_full, np.ones(main_class.nv_for_r))

        energy_diff = (Ekv - Ekm)  # e_diff(k,v,m) = [E(k,v) - E(k,m)]^-1
        with np.errstate(divide='ignore'):
            energy_diff_inverse = 1 / energy_diff
            energy_diff_inverse[abs(energy_diff) < 0.002] = 0

        totx = np.einsum('kvm,kvm,kmc-> kvc', datay[:, 0:main_class.nv_for_r, :], energy_diff_inverse,
                         dataz[:, :, main_class.nv_for_r:main_class.nv_for_r + main_class.nc_for_r]) - \
               np.einsum('kvm,kvm,kmc-> kvc', dataz[:, 0:main_class.nv_for_r, :], energy_diff_inverse,
                         datay[:, :, main_class.nv_for_r:main_class.nv_for_r + main_class.nc_for_r])
        toty = np.einsum('kvm,kvm,kmc-> kvc', dataz[:, 0:main_class.nv_for_r, :], energy_diff_inverse,
                         datax[:, :, main_class.nv_for_r:main_class.nv_for_r + main_class.nc_for_r]) - \
               np.einsum('kvm,kvm,kmc-> kvc', datax[:, 0:main_class.nv_for_r, :], energy_diff_inverse,
                         dataz[:, :, main_class.nv_for_r:main_class.nv_for_r + main_class.nc_for_r])
        totz = np.einsum('kvm,kvm,kmc-> kvc', datax[:, 0:main_class.nv_for_r, :], energy_diff_inverse,
                         datay[:, :, main_class.nv_for_r:main_class.nv_for_r + main_class.nc_for_r]) - \
               np.einsum('kvm,kvm,kmc-> kvc', datay[:, 0:main_class.nv_for_r, :], energy_diff_inverse,
                         datax[:, :, main_class.nv_for_r:main_class.nv_for_r + main_class.nc_for_r])
        L[:, :, :, 0] = totx * fact
        L[:, :, :, 1] = toty * fact
        L[:, :, :, 2] = totz * fact
        print('finish calculating L_kvc')

        return L[:, main_class.nv_for_r - main_class.nv:main_class.nv_for_r, 0:main_class.nc, :]
