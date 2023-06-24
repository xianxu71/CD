import numpy as np
from mpi import MPI, comm, size, rank
import h5py as h5
import math_function

class reader:
    """
    read data from input files
    """
    def __init__(self, main_class):
        if main_class.read_temp:
            h5_file_r = h5.File(main_class.input_folder+'temp.h5', 'r')
            main_class.noeh_dipole_full = np.array(h5_file_r.get('noeh_dipole_full'))
            main_class.noeh_dipole = np.array(h5_file_r.get('noeh_dipole'))
            h5_file_r.close()
        else:
            main_class.noeh_dipole_full, main_class.noeh_dipole = self.read_noeh_dipole(main_class)  #read dipole from vmtxel
        main_class.energy_dft_full, main_class.energy_dft = self.read_dft_energy(main_class) #read dft energy for each band
        main_class.avck = self.avck_reader(main_class) #read Acvk from eigenvectors.h5
        main_class.excited_energy = self.read_excited_energy(main_class) #read exciton energy
        main_class.volume = self.read_volume(main_class)
        #main_class.eqp_corr = self.read_eqp(main_class)
        if main_class.use_eqp:
            main_class.eqp_corr = self.read_eqp(main_class) #read quasiparticle energy correction from eqp.dat
    def read_volume(self,main_class):
        input_file = main_class.input_folder + 'eigenvectors.h5'
        f = h5.File(input_file, 'r')
        volume = f['mf_header/crystal/celvol'][()]
        f.close()
        return volume
    def read_excited_energy(self, main_class):
        input_file = main_class.input_folder + 'eigenvectors.h5'
        f = h5.File(input_file, 'r')
        energy = f['exciton_data/eigenvalues'][0:main_class.nxct]
        f.close()
        return energy
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

    def read_eqp(self, main_class):
        input_file = main_class.input_folder + 'eqp.dat'

        data = np.loadtxt(input_file)

        eqp_corr = np.zeros([main_class.nk, main_class.nc + main_class.nv])
        for ik in range(main_class.nk):
            for ib in range(main_class.nc + main_class.nv):
                eqp_corr[ik, ib] = data[ik * (main_class.nc + main_class.nv + 1) + ib + 1, 3] - data[ik * (main_class.nc + main_class.nv + 1) + ib + 1, 2]
        print('finish loading quasi-particle energy from {0:s}'.format('eqp.dat'))
        return eqp_corr

    def read_dft_energy(self, main_class):
        input_file = main_class.input_folder + 'eigenvectors.h5'
        f = h5.File(input_file, 'r')
        energy_full = f['mf_header/kpoints/el'][()]
        energy = energy_full[0, :, main_class.hovb - main_class.nv: main_class.hovb + main_class.nc]
        f.close()
        print('finish reading dft energy')
        return energy_full[0, :, main_class.hovb - main_class.nv_for_r: main_class.hovb + main_class.nc_for_r], energy


    def read_noeh_dipole(self, main_class):
        filename_1 = main_class.input_folder + "vmtxel_nl_b1.dat" #file names
        filename_2 = main_class.input_folder + "vmtxel_nl_b2.dat"
        filename_3 = main_class.input_folder + "vmtxel_nl_b3.dat"
        filename_1_sorted = main_class.input_folder + "vmtxel_nl_b1_sorted.dat" #sorted file names
        filename_2_sorted = main_class.input_folder + "vmtxel_nl_b2_sorted.dat"
        filename_3_sorted = main_class.input_folder + "vmtxel_nl_b3_sorted.dat"

        if rank == 0:
            self.vmtxel_sort(filename_1, filename_1_sorted) #get rid of the disorder in vmtxel*dat
            self.vmtxel_sort(filename_2, filename_2_sorted)
            self.vmtxel_sort(filename_3, filename_3_sorted)
        comm.Barrier()
        file_1 = open(filename_1_sorted)
        file_2 = open(filename_2_sorted)
        file_3 = open(filename_3_sorted)
        header = file_1.readline()
        header = file_2.readline()
        header = file_3.readline()
        nb = main_class.nv_in_file + main_class.nc_in_file
        nb2 = main_class.nv + main_class.nc
        noeh_dipole_full = np.zeros([main_class.nk, nb, nb, 3], dtype=np.complex)
        noeh_dipole_partial = np.zeros([main_class.nk, nb2, nb2, 3], dtype=np.complex)
        for ik in range(0, main_class.nk):
            for ib1 in range(0, nb):
                for ib2 in range(0, nb):
                    line_1 = file_1.readline()
                    line_2 = file_2.readline()
                    line_3 = file_3.readline()

                    v1_real, v1_imag = line_1.split(',')
                    v1_real = float(v1_real.strip('('))
                    v1_imag = float(v1_imag.strip(')\n'))

                    v2_real, v2_imag = line_2.split(',')
                    v2_real = float(v2_real.strip('('))
                    v2_imag = float(v2_imag.strip(')\n'))

                    v3_real, v3_imag = line_3.split(',')
                    v3_real = float(v3_real.strip('('))
                    v3_imag = float(v3_imag.strip(')\n'))

                    v1 = v1_real + 1j * v1_imag
                    v2 = v2_real + 1j * v2_imag
                    v3 = v3_real + 1j * v3_imag

                    noeh_dipole_full[ik, ib1, ib2, 0] = v1
                    noeh_dipole_full[ik, ib1, ib2, 1] = v2
                    noeh_dipole_full[ik, ib1, ib2, 2] = v3
        noeh_dipole_full_temp = np.zeros([main_class.nk, nb, nb, 3], dtype=np.complex)
        noeh_dipole_full_temp[:,:,:,0], noeh_dipole_full_temp[:,:,:,1], noeh_dipole_full_temp[:,:,:,2] = math_function.b123_to_xyz(
            main_class.a, noeh_dipole_full[:,:,:,0], noeh_dipole_full[:,:,:,1], noeh_dipole_full[:,:,:,2]
        )
        noeh_dipole_full = noeh_dipole_full_temp * 1.00

        noeh_dipole_full = noeh_dipole_full[:, main_class.nv_in_file - main_class.nv_for_r:main_class.nv_in_file + main_class.nc_for_r,
                           main_class.nv_in_file - main_class.nv_for_r:main_class.nv_in_file + main_class.nc_for_r, :]
        noeh_dipole_partial = noeh_dipole_full[:, main_class.nv_for_r - main_class.nv:main_class.nv_for_r + main_class.nc, main_class.nv_for_r - main_class.nv:main_class.nv_for_r + main_class.nc, :]
        print('finish reading dipoles from vmtxel')
        return noeh_dipole_full, noeh_dipole_partial

    def vmtxel_sort(self, vm,vm_sorted):
        '''
            get rid of the disorder in vmtxel*.dat
            '''
        f = open(vm, 'r')
        header = f.readline()

        dipole_element_list = {}  # [0:'(a1,b1)',1:'(a2,b2)',2:'(a3,b3)'....]
        i = 0
        while True:
            line = f.readline()
            #        print(line)
            if line == '':
                f.close()
                break
            else:
                temp = line.split(") (")
                if len(temp) != 1:
                    for content in temp:
                        dipole_element_list[i] = content
                        i += 1
                else:
                    dipole_element_list[i] = temp[0]
                    i += 1
        print('finish sorting vmtxel')
        f_new = open(vm_sorted, 'w')
        f_new.write(header)
        count = 0
        for i in range(len(dipole_element_list)):
            if '(' != dipole_element_list[i].strip()[0]:
                f_new.write('(' + dipole_element_list[i].strip() + '\n')
                count += 1
                continue
            elif ')' != dipole_element_list[i].strip()[-1]:
                f_new.write(dipole_element_list[i].strip() + ')\n')
                count += 1
                continue
            else:
                count += 1
                f_new.write(dipole_element_list[i].strip() + '\n')
        print('nk*nv*nc:', count)
        f_new.close()

