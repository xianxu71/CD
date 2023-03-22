import numpy as np
import h5py
from eigenvalues_b_noeh_reader import read_noeh_dipole, dft_energy_reader
from mpi import MPI, comm, size, rank
from magetic import calculate_L, calculate_ElectricDipole, calculate_MM_ME
from exciton import avck_reader, excited_energy_reader
import matplotlib.pyplot as plt
from optical import calculate_epsR_epsL
from input import *

noeh_dipole = read_noeh_dipole(nk, nc + nv, input_folder)  # dim [nk, nb, nb, 3], read <nk|p|mk>

energy_dft = dft_energy_reader(nk, nc, nv, hovb, input_folder)  # dim [nk,nb], read dft level energy of each band

L_kvc = calculate_L(noeh_dipole, energy_dft, nk, nv, nc)  # dim [nk,nv,nc,3], orbital angular momentum

E_kvc = calculate_ElectricDipole(noeh_dipole, nk, nv, nc, energy_dft)  # dim [nk,nv,nc,3], electric dipole

avck = avck_reader(nxct, input_folder)

excited_energy = excited_energy_reader(nxct, input_folder)

MM, ME = calculate_MM_ME(nc, nv, nk, nxct, avck, E_kvc, L_kvc)


W = np.linspace(1.5, 2.5, 1000)
sigma = 0.02
alpha = 1. / (2 * sigma ** 2)

epsR_epsL, Y1, Y2 = calculate_epsR_epsL(MM, ME, excited_energy, nxct, W, alpha)

plt.figure()
plt.plot(W, epsR_epsL)
plt.plot(W, Y1)
plt.plot(W, Y2)
plt.show()

data = np.array([W, Y1, Y2, epsR_epsL])
np.savetxt('CD.dat', data.T)

print('test')
