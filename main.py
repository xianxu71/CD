import numpy as np
import h5py
from reader import read_noeh_dipole, dft_energy_reader, avck_reader, excited_energy_reader, volume_reader
from mpi import MPI, comm, size, rank
from magetic import calculate_L, calculate_ElectricDipole, calculate_MM_ME
import matplotlib.pyplot as plt
from optical import calculate_epsR_epsL_eh, calculate_absorption_eh, calculate_absorption_noeh, calculate_m, calculate_epsR_epsL_noeh
from input import *

noeh_dipole_full, noeh_dipole = read_noeh_dipole(nk, nv, nc, input_folder, nv_for_r, nc_for_r)  # dim [nk, nb, nb, 3], read <nk|p|mk>

energy_dft_full, energy_dft = dft_energy_reader(nk, nc, nv, hovb, input_folder, nv_for_r, nc_for_r)  # dim [nk,nb], read dft level energy of each band

L_kvc = calculate_L(noeh_dipole_full, energy_dft_full, nk, nv, nc, nv_for_r, nc_for_r)  # dim [nk,nv,nc,3], orbital angular momentum

E_kvc = calculate_ElectricDipole(noeh_dipole, nk, nv, nc, energy_dft)  # dim [nk,nv,nc,3], electric dipole

avck = avck_reader(nxct, input_folder)

excited_energy = excited_energy_reader(nxct, input_folder)

MM, ME = calculate_MM_ME(nc, nv, nk, nxct, avck, E_kvc, L_kvc)


W = np.linspace(0.1, 6, 4000)
eta = 0.05


volume = volume_reader(input_folder)

#calculate_epsR_epsL_eh(nk,MM, ME, excited_energy, nxct, W, eta, volume)
#calculate_absorption_eh(nk, MM, ME, excited_energy, nxct, W, eta, volume)
calculate_absorption_noeh (noeh_dipole, nk, nv, nc, energy_dft, W, eta, volume)
#calculate_m(nk,MM, ME, excited_energy, nxct, W, eta, volume)
#calculate_epsR_epsL_noeh(E_kvc, L_kvc,nk, nv, nc, energy_dft, W, eta, volume)
print('test')
