import numpy as np
from input import *
import main_class
import optical
import kits
import debug

if __name__ == '__main__':

    main_class = main_class.main_class(nk, nc, nv, nc_for_r, nv_for_r, nc_in_file, nv_in_file,
                                       hovb, nxct, input_folder,W, eta , use_eqp, write_temp,
                                       read_temp, energy_shift, eps1_correction, degeneracy_remover, a)


    #optical.calculate_epsR_epsL_noeh(main_class)
    #optical.calculate_epsR_epsL_eh(main_class)
    #optical.calculate_absorption_noeh(main_class)
    #optical.calculate_absorption_eh(main_class)
    #optical.calculate_m(main_class)

    #kits.vc_contribution(main_class)
    debug.calculate_diag_L(main_class)

