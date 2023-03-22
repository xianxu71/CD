import numpy as np
import h5py as h5

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