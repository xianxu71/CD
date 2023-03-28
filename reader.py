import numpy as np
from mpi import MPI, comm, size, rank
import h5py as h5


def vmtxel_sort(vm):
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
        if i % 10000 == 0:
            print("Progress:", i)
    print('finish reading from vmtxel')
    f_new = open(vm, 'w')
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


def read_noeh_dipole(nk, nv, nc , input_folder, nv_for_r, nc_for_r):
    '''
    read read <nk|p|mk> from vmtxel_nl*dat

    dim: [nk, nb, nb, 3]

    :param nk: number of k points
    :param nb: nv+nc
    :param input_folder: folders with all inputs
    :return: <nk|p|mk>
    '''
    filename_1 = input_folder + "vmtxel_nl_b1.dat"
    filename_2 = input_folder + "vmtxel_nl_b2.dat"
    filename_3 = input_folder + "vmtxel_nl_b3.dat"
    if rank == 0:
        vmtxel_sort(filename_1)
        vmtxel_sort(filename_2)
        vmtxel_sort(filename_3)
    comm.Barrier()
    file_1 = open(filename_1)
    file_2 = open(filename_2)
    file_3 = open(filename_3)
    header = file_1.readline()
    header = file_2.readline()
    header = file_3.readline()
    nb = nv_for_r+nc_for_r
    nb2 = nv + nc
    noeh_dipole_full = np.zeros([nk, nb, nb, 3], dtype=np.complex)
    noeh_dipole_partial = np.zeros([nk, nb2, nb2, 3], dtype=np.complex)
    for ik in range(0, nk):
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
    noeh_dipole_partial = noeh_dipole_full[:,nv_for_r-nv:nv_for_r+nc,nv_for_r-nv:nv_for_r+nc,:]
    return noeh_dipole_full, noeh_dipole_partial


def dft_energy_reader(nk, nc, nv, hovb, input_folder, nv_for_r, nc_for_r):
    '''
    read dft level energy of each band

    dim: [nk, nb, 3]

    :param nk: number of kpoints
    :param nc: number of conduction bands
    :param nv: number of valance bands
    :param hovb: index of the highest occupied band
    :param input_folder: folder with all input files
    :return: dft level energy of each band
    '''
    input_file = input_folder+'eigenvectors.h5'
    f = h5.File(input_file, 'r')
    energy_full = f['mf_header/kpoints/el'][()]
    energy = energy_full[0, :, hovb - nv: hovb + nc]
    f.close()
    return energy_full[0, :, hovb - nv_for_r: hovb + nc_for_r], energy

def avck_reader(nxct,input_folder):
    input_file = input_folder+'eigenvectors.h5'
    f = h5.File(input_file, 'r')
    avck = f['exciton_data/eigenvectors'][()]
    f.close()
    avck = np.transpose(avck, (0, 1, 2, 4, 3, 5, 6)) #eigenvectors in the h5 file is [..., c , v ...], we convert it to [..., v , c ...]
    avck = avck[0,0:nxct,:,:,:,0,0]+1j*avck[0,0:nxct,:,:,:,0,1]
    avck = np.transpose(avck,(1,2,3,0))
    return avck

def excited_energy_reader(nxct,input_folder):
    input_file = input_folder + 'eigenvectors.h5'
    f = h5.File(input_file, 'r')
    energy = f['exciton_data/eigenvalues'][0:nxct]
    f.close()
    return energy

def volume_reader(input_folder):
    input_file = input_folder + 'eigenvectors.h5'
    f = h5.File(input_file, 'r')
    volume = f['mf_header/crystal/celvol'][()]
    f.close()
    return volume