import numpy as np
from input import *
import main_class
import optical
import kits
import debug
import optical_length

if __name__ == '__main__':

    main_class = main_class.main_class(nk, nc, nv, nc_for_r, nv_for_r, nc_in_file, nv_in_file,
                                       hovb, nxct, input_folder,W, eta , use_eqp, write_temp,
                                       read_temp, energy_shift, eps1_correction, degeneracy_remover, a, spinor,use_length,
                                       nc_for_length, nv_for_length)


    optical.calculate_epsR_epsL_noeh(main_class)
    #optical.calculate_epsR_epsL_eh(main_class)
    #optical.calculate_absorption_noeh(main_class)
    #optical.calculate_absorption_eh(main_class)
    #optical.calculate_m_eh(main_class)
    #optical.calculate_m_noeh(main_class)

    #kits.vc_contribution(main_class)
    #kits.dipole_k_xy(main_class)
    #kits.dipole_k_L(main_class)
    #kits.dipole_k_R(main_class)
    #kits.dipole_k_L2(main_class)
    #debug.calculate_diag_L(main_class)


    #optical_length.calculate_absorption_noeh(main_class)
    #optical_length.calculate_cd_noeh(main_class)
    #optical_length.calculate_absorption_eh(main_class)



