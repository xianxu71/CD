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
# read_temp = False
#
# energy_shift = 0
# eps1_correction = 0
# degeneracy_remover = 0.002
# a = np.array([[ 1,   0,  0],
#               [ 0,   1,   0],
#               [ 0,   0,  1]])



# nk = 16 * 16 * 2
# nc = 12
# nv = 12
#
# nc_for_r = 60
# nv_for_r = 40
#
# nc_in_file = 60
# nv_in_file = 40
#
# hovb = 352 # index of the highest occupied band
# nxct = 800
#
# input_folder = './S-NPB/'
#
#
# #W = np.linspace(3.0, 3.4, 4000) #energy range
# W = np.linspace(3.5, 4.5, 4000)
# eta = 0.044 #eta = 0.044 #broadening coefficient
#
# use_eqp = True #use eqp correction or not
#
# write_temp = True
# read_temp = False
#
# energy_shift = 0.29
# eps1_correction = 2.65
# degeneracy_remover = 0.000000001
#
# # a = np.array([[ 8.425800753,   0.000000000,  -0.131321631],
# #               [ 0.000000000,   7.517499108,   0.000000000],
# #               [ -1.532446824,   0.000000000,  18.676624778]])
#
# a = np.array([[ 1,   0,  0],
#               [ 0,   1,   0],
#               [ 0,   0,  1]])


# nk = 20 * 20 * 2
# nc = 6
# nv = 6
#
# nc_for_r = 60
# nv_for_r = 40
#
# nc_in_file = 60
# nv_in_file = 40
#
# hovb = 352 # index of the highest occupied band
# nxct = 800
#
# input_folder = './S-NPB_20202_6_6/'
#
#
# #W = np.linspace(3.0, 3.4, 4000) #energy range
# W = np.linspace(0.5, 4, 4000)
# eta = 0.05 #eta = 0.044 #broadening coefficient
#
# use_eqp = False #use eqp correction or not
#
# write_temp = True
# read_temp = False
#
# energy_shift = 0
# eps1_correction = 0
# degeneracy_remover = 0.000000001

# a = np.array([[ 8.425800753,   0.000000000,  -0.131321631],
#               [ 0.000000000,   7.517499108,   0.000000000],
#               [ -1.532446824,   0.000000000,  18.676624778]])

# a = np.array([[ 1,   0,  0],
#               [ 0,   1,   0],
#               [ 0,   0,  1]])

nk = 40 * 40 * 1
nc = 6
nv = 4

nc_for_r = 40
nv_for_r = 4

nc_in_file = 40
nv_in_file = 4

hovb = 4 # index of the highest occupied band
nxct = 800

input_folder = './bn_40401/'


#W = np.linspace(3.0, 3.4, 4000) #energy range
W = np.linspace(0, 5, 4000)
eta = 0.05 #eta = 0.044 #broadening coefficient

use_eqp = False #use eqp correction or not

write_temp = True
read_temp = False

energy_shift = 0
eps1_correction = 0
degeneracy_remover = 0.0000000001
spinor = 1

# a = np.array([[0.5,       -0.866025404,       0],
#               [0.5,       0.866025404,       0],
#               [0,       0,       4]])

# a = np.array([[4.41813,       0,       0],
#                [-2.209065,       3.8262131617,       0],
#                [0,       0,       4]])

a = np.array([[1,       0,       0],
               [0,       1,       0],
               [0,       0,       1]])