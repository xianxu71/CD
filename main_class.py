import numpy as np
import reader
import electromagnetic
import h5py as h5

class main_class:
    '''
    This is the main class where most parameters and data store
    '''
    def __init__(self, nk, nc, nv, nc_for_r, nv_for_r, nc_in_file, nv_in_file,
                 hovb, nxct, input_folder, W, eta , use_eqp, write_temp, read_temp,
                 energy_shift, eps1_correction, degeneracy_remover, a, spinor):
        """
        intialize main_class from input.py and all the input files
        """
        self.nk = nk  #number of kpoints
        self.nc = nc #number of conduction bands in eigenvectors.h5
        self.nv = nv  #number of valence bands in eigenvectors.h5
        self.nc_for_r = nc_for_r #number of conduction bands for <\psi|r|\psi>
        self.nv_for_r = nv_for_r #number of valence bands for <\psi|r|\psi>
        self.nc_in_file = nc_in_file #number of conduction bands in vmtxel_nl_b*.dat
        self.nv_in_file = nv_in_file #number of valence bands in vmtxel_nl_b*.dat
        self.hovb = hovb # index of the highest occupied band
        self.nxct = nxct # number of exciton states
        self.input_folder = input_folder #address of input folder
        self.W = W #energy range
        self.eta = eta #broadening coefficient
        self.use_eqp = use_eqp #use eqp correction or not
        self.write_temp = write_temp
        self.read_temp = read_temp
        self.energy_shift = energy_shift
        self.eps1_correction = eps1_correction
        self.degeneracy_remover = degeneracy_remover
        self.a = a
        self.spinor = spinor

        self.reader = reader.reader(self) #read all the data from input file
        self.electromagnetic = electromagnetic.electromagnetic(self) #calculate all the electromagnetic matrices

        if self.write_temp:
            h5_file_w = h5.File(self.input_folder+'temp.h5','w')
            h5_file_w.create_dataset('L_kvc',data = self.L_kvc)
            h5_file_w.create_dataset('E_kvc', data=self.E_kvc)
            h5_file_w.create_dataset('MM', data=self.MM)
            h5_file_w.create_dataset('ME', data=self.ME)
            h5_file_w.create_dataset('noeh_dipole_full', data=self.noeh_dipole_full)
            h5_file_w.create_dataset('noeh_dipole', data=self.noeh_dipole)
            print('write temp matrices to temp.h5')
            h5_file_w.close()