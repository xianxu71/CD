import numpy as np
import h5py as h5

class electromagnetic:
    def __init__(self, main_class):
        #main_class.L_kvc = self.calculate_L(main_class)  # calculated orbital angular momentum
        main_class.E_kvc = self.calculate_E(main_class)  # calculated electric dipole
        main_class.MM, main_class.ME = self.calculate_MM_ME(main_class)  # calculate orbital angular momentum and electric dipole with exciton representation
        main_class.L_kvc = self.calculate_L(main_class)
    def calculate_E(self,main_class):
        a2bohr = 1.88973
        nv = main_class.nv
        nc = main_class.nc
        E = main_class.mydipole_W90[:, main_class.nv_for_length - nv: main_class.nv_for_length,
                        main_class.nv_for_length : main_class.nv_for_length + nc, :] * a2bohr
        print('finish calculating E_kvc')
        return E

    def calculate_MM_ME(self,main_class):
        idx = list(range(main_class.nv - 1, -1, -1))

        inds = np.ix_(range(main_class.nk), idx, range(main_class.nc), range(3))
        E_temp = main_class.E_kvc[inds]
        #L_temp = main_class.L_kvc[inds]

        #MM = np.einsum('kvcs,kvcd->sd', main_class.avck, L_temp)
        MM = 0
        ME = np.einsum('kvcs,kvcd->sd', main_class.avck, E_temp)
        print('finish calculating MM ME')
        return MM, ME
    def calculate_L(self,main_class):
        fact = 1j/2
        datax = main_class.mydipole_W90[:, :, :, 0]
        datay = main_class.mydipole_W90[:, :, :, 1]
        dataz = main_class.mydipole_W90[:, :, :, 2]

        L = np.zeros([main_class.nk, main_class.nv_for_length, main_class.nc_for_length, 3], dtype=np.complex)
        Lx = []
        Ly = []
        Lz = []
        for k in range(main_class.nk):
            Lx.append([])
            Ly.append([])
            Lz.append([])
            for v in range(main_class.nv_for_length):
                Lx[k].append([])
                Ly[k].append([])
                Lz[k].append([])
                for c in range(main_class.nv_for_length,main_class.nv_for_length+main_class.nc_for_length):
                    totx = 0
                    toty = 0
                    totz = 0
                    for m in range(main_class.nv_for_length+main_class.nc_for_length):
                        curx = datay[k][v][m] * dataz[k][m][c] - dataz[k][v][m] * datay[k][m][c]
                        curx = curx * (main_class.energy_dft_full[k][m] - main_class.energy_dft_full[k][c])
                        totx += curx * fact

                        cury = dataz[k][v][m] * datax[k][m][c] - datax[k][v][m] * dataz[k][m][c]
                        cury = cury * (main_class.energy_dft_full[k][m] - main_class.energy_dft_full[k][c])
                        toty += cury * fact

                        curz = datax[k][v][m] * datay[k][m][c] - datay[k][v][m] * datax[k][m][c]
                        curz = curz * (main_class.energy_dft_full[k][m] - main_class.energy_dft_full[k][c])
                        totz += curz * fact
                    Lx[k][v].append(totx)
                    Ly[k][v].append(toty)
                    Lz[k][v].append(totz)
        Lx = np.array(Lx)
        Ly = np.array(Ly)
        Lz = np.array(Lz)
        for ik in range(main_class.nk):
            for iv in range(main_class.nv_for_length):
                for ic in range(main_class.nv_for_length):
                    energy_diff_for_cancel_diple = main_class.energy_dft_full[ik,iv] - main_class.energy_dft_full[ik,ic+main_class.nv_for_length]
                    energy_diff_for_cancel_diple_inv = 1/energy_diff_for_cancel_diple
                    Lx[ik,iv,ic] = Lx[ik,iv,ic]*energy_diff_for_cancel_diple_inv
                    Ly[ik, iv, ic] = Ly[ik, iv, ic] * energy_diff_for_cancel_diple_inv
                    Lz[ik, iv, ic] = Lz[ik, iv, ic] * energy_diff_for_cancel_diple_inv
        L[:, :, :, 0] = Lx * fact
        L[:, :, :, 1] = Ly * fact
        L[:, :, :, 2] = Lz * fact
        print('finish calculating L_kvc')
        return L[:, main_class.nv_for_length - main_class.nv:main_class.nv_for_length, 0:main_class.nc, :]






