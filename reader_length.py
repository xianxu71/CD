import numpy as np
from mpi import MPI, comm, size, rank
import h5py as h5
import math_function

class reader:
    def __init__(self, main_class):
        self.nband_for_length = main_class.nv_for_length+main_class.nc_for_length
        main_class.energy_dft_full, main_class.energy_dft = self.read_dft_energy(main_class)
        main_class.avck = self.avck_reader(main_class)


        self.read_eigenvectors_h5(main_class)
        self.read_dipole(main_class)
        self.dipole_match_k(main_class)

    def read_dft_energy(self, main_class):
        input_file = main_class.input_folder + 'eigenvectors.h5'
        f = h5.File(input_file, 'r')
        energy_full = f['mf_header/kpoints/el'][()]
        energy = energy_full[0, :, main_class.hovb - main_class.nv: main_class.hovb + main_class.nc]
        f.close()
        print('finish reading dft energy')
        return energy_full[0, :, main_class.hovb - main_class.nv_for_length: main_class.hovb + main_class.nc_for_length], energy

    def read_eigenvectors_h5(self,main_class):
        fname = main_class.input_folder + "eigenvectors.h5"
        f = h5.File(fname,'r')
        main_class.rk = f['mf_header/kpoints/rk'][()]
        main_class.volume = f['mf_header/crystal/celvol'][()]
        main_class.excited_energy = f['exciton_data/eigenvalues'][0:main_class.nxct]
        f.close()
    def avck_reader(self, main_class):
        input_file = main_class.input_folder + 'eigenvectors.h5'
        f = h5.File(input_file, 'r')
        avck = f['exciton_data/eigenvectors'][()]
        f.close()
        avck = np.transpose(avck, (
        0, 1, 2, 4, 3, 5, 6))  # eigenvectors in the h5 file is [..., c , v ...], we convert it to [..., v , c ...]
        avck = avck[0, 0:main_class.nxct, :, :, :, 0, 0] + 1j * avck[0, 0:main_class.nxct, :, :, :, 0, 1]
        avck = np.transpose(avck, (1, 2, 3, 0))
        print('finish reading Acvk from eigenvectors.h5')
        return avck


    def read_dipole(self,main_class):
        fname = main_class.input_folder + "all_r_mn_all.dat"
        with open(fname, 'r') as f:
            print('\n Reading dipole from', fname)
            mydipole_W90 = np.zeros([main_class.nk, self.nband_for_length, self.nband_for_length, 3], dtype=np.complex)
            rk_W90 = np.zeros([main_class.nk,3])
            for i in range(main_class.nk):
                rk_W90[i,:] = list(map(float, f.readline().split()))
                for mb in range(self.nband_for_length):
                    for nb in range(self.nband_for_length):
                        for direction in range(3):
                            xx, yy = list(map(float, f.readline()[2:-2].split(",")))
                            mydipole_W90[i,mb,nb,direction] = xx + 1j*yy
        self.rk_W90 = rk_W90
        self.mydipole_W90 = mydipole_W90

        return 0
    def dipole_match_k(self, main_class):
        len_k = main_class.rk.shape[0]
        len_W90_k = self.rk_W90.shape[0]
        new_dipole = np.zeros([len_k, self.nband_for_length, self.nband_for_length, 3], dtype=np.complex)
        dict1 = {str(self.rk_W90[i, :].round(decimals = 3)): i for i in range(len_W90_k)}
        for i in range(len_k):
            j = dict1[str((main_class.rk[i, :]+0.00000000001).round(decimals = 3))]
            new_dipole[i, :, :, :] = self.mydipole_W90[j, :, :, :]
        main_class.mydipole_W90 = new_dipole * 1.00000
