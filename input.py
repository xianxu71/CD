import numpy as np
# nk = 16 * 16 * 2   #number of kpoints
# nc = 8    #number of conduction bands in eigenvectors.h5
# nv = 8    #number of valence bands in eigenvectors.h5
#
# nc_for_r = 8  #number of conduction bands for <\psi|r|\psi>
# nv_for_r = 8  #number of valence bands for <\psi|r|\psi>
#
# nc_in_file = 8  #number of conduction bands in vmtxel_nl_b*.dat
# nv_in_file = 8  #number of valence bands in vmtxel_nl_b*.dat
#
# hovb = 560      # index of the highest occupied band
# nxct = 4000     # number of exciton states
#
# input_folder = './SMBA2PbI4_442_16162_8_8/'  #address of input folder
#
# W = np.linspace(1.5, 2.7, 4000) #energy range
# eta = 0.02 #broadening coefficient
#
# use_eqp = True #use eqp correction or not
#
# write_temp = False
# read_temp = True



nk = 16 * 16 * 2
nc = 12
nv = 12

nc_for_r = 60
nv_for_r = 40

nc_in_file = 60
nv_in_file = 40

hovb = 352 # index of the highest occupied band
nxct = 800

input_folder = './S-NPB_scissor/'

use_eqp = False

W = np.linspace(2.8, 3.3, 4000) #energy range
eta = 0.044 #broadening coefficient

use_eqp = True #use eqp correction or not

write_temp = True
read_temp = False

energy_shift = 0.307
eps1_correction = 2.65
degeneracy_remover = 0.004

a = np.array([[ 8.425800753,   0.000000000,  -0.131321631],
              [ 0.000000000,   7.517499108,   0.000000000],
              [ -1.532446824,   0.000000000,  18.676624778]])


