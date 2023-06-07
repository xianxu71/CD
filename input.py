import numpy as np
nk = 16 * 16 * 2   #number of kpoints
nc = 8    #number of conduction bands in eigenvectors.h5
nv = 8    #number of valence bands in eigenvectors.h5

nc_for_r = 8  #number of conduction bands for <\psi|r|\psi>
nv_for_r = 8  #number of valence bands for <\psi|r|\psi>

nc_in_file = 8  #number of conduction bands in vmtxel_nl_b*.dat
nv_in_file = 8  #number of valence bands in vmtxel_nl_b*.dat

hovb = 560      # index of the highest occupied band
nxct = 4000     # number of exciton states

input_folder = './SMBA2PbI4_442_16162_8_8/'  #address of input folder

W = np.linspace(1.5, 2.7, 4000) #energy range
eta = 0.02 #broadening coefficient

use_eqp = True #use eqp correction or not

write_temp = False
read_temp = True

