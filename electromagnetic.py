import numpy as np
import h5py as h5

class electromagnetic:
    def __init__(self, main_class):
        main_class.E_kvc = self.calculate_E(main_class)  # calculated electric dipole
        main_class.L_kvc = self.calculate_L4(main_class)  # calculated orbital angular momentum
        main_class.MM, main_class.ME, main_class.MM_k, main_class.ME_k  = self.calculate_MM_ME(
            main_class)  # calculate orbital angular momentum and electric dipole with exciton representation



    def calculate_E(self,main_class):
        fact = 1  # test
        E = main_class.noeh_dipole[:, 0:main_class.nv, main_class.nv:main_class.nc + main_class.nv, :] * fact
        print('finish calculating E_kvc')
        return E

    def calculate_MM_ME(self,main_class):
        idx = list(range(main_class.nv - 1, -1, -1))

        inds = np.ix_(range(main_class.nk), idx, range(main_class.nc), range(3))
        E_temp = main_class.E_kvc[inds]
        L_temp = main_class.L_kvc[inds]

        MM = np.einsum('kvcs,kvcd->sd', main_class.avck, L_temp)
        ME = np.einsum('kvcs,kvcd->sd', main_class.avck, E_temp)
        MM_k = np.einsum('kvcs,kvcd->skd', main_class.avck, L_temp)
        ME_k = np.einsum('kvcs,kvcd->skd', main_class.avck, E_temp)

        print('finish calculating MM ME')
        return MM, ME, MM_k, ME_k

    # def calculate_L(self, main_class):
    #     '''
    #     calculate orbital angular momentum
    #     '''
    #     '''
    #         # dim [nk,nv,nc,3], orbital angular momentum
    #         calculate orbital angular momentum
    #
    #         noeh_dipole: dim: [nk, nb, nb, 3]
    #
    #     '''
    #
    #     # m0 = 9.10938356e-31 # mass of electrons
    #     # eVtoJ = 1.602177e-19 # electron volt = ？Joule
    #     # p_ry_to_SI = 1.9928534e-24 # convert momentum operator to SI unit # ??
    #     # fact = (p_ry_to_SI ** 2) / m0 / eVtoJ # pre factor
    #     fact = 2*1j
    #     datax = main_class.noeh_dipole_full_not_over_dE [:, :, :, 0]
    #     datay = main_class.noeh_dipole_full_not_over_dE[:, :, :, 1]
    #     dataz = main_class.noeh_dipole_full_not_over_dE[:, :, :, 2]
    #
    #
    #
    #     L = np.zeros([main_class.nk, main_class.nv_for_r, main_class.nc_for_r, 3], dtype=np.complex)
    #
    #     Ekv = np.einsum('kv,m->kvm', main_class.energy_dft_full[:, 0:main_class.nv_for_r], np.ones(main_class.nv_for_r + main_class.nc_for_r))
    #     Ekm = np.einsum('km,v->kvm', main_class.energy_dft_full, np.ones(main_class.nv_for_r))
    #
    #     energy_diff = (Ekv - Ekm)  # e_diff(k,v,m) = [E(k,v) - E(k,m)]^-1, unit: ryd
    #     with np.errstate(divide='ignore'):
    #         energy_diff_inverse = 1 / energy_diff
    #         energy_diff_inverse[abs(energy_diff) < main_class.degeneracy_remover] = 0
    #
    #     totx = np.einsum('kvm,kvm,kmc-> kvc', datay[:, 0:main_class.nv_for_r, :], energy_diff_inverse,
    #                      dataz[:, :, main_class.nv_for_r:main_class.nv_for_r + main_class.nc_for_r]) - \
    #            np.einsum('kvm,kvm,kmc-> kvc', dataz[:, 0:main_class.nv_for_r, :], energy_diff_inverse,
    #                      datay[:, :, main_class.nv_for_r:main_class.nv_for_r + main_class.nc_for_r])
    #     toty = np.einsum('kvm,kvm,kmc-> kvc', dataz[:, 0:main_class.nv_for_r, :], energy_diff_inverse,
    #                      datax[:, :, main_class.nv_for_r:main_class.nv_for_r + main_class.nc_for_r]) - \
    #            np.einsum('kvm,kvm,kmc-> kvc', datax[:, 0:main_class.nv_for_r, :], energy_diff_inverse,
    #                      dataz[:, :, main_class.nv_for_r:main_class.nv_for_r + main_class.nc_for_r])
    #     totz = np.einsum('kvm,kvm,kmc-> kvc', datax[:, 0:main_class.nv_for_r, :], energy_diff_inverse,
    #                      datay[:, :, main_class.nv_for_r:main_class.nv_for_r + main_class.nc_for_r]) - \
    #            np.einsum('kvm,kvm,kmc-> kvc', datay[:, 0:main_class.nv_for_r, :], energy_diff_inverse,
    #                      datax[:, :, main_class.nv_for_r:main_class.nv_for_r + main_class.nc_for_r])
    #     for ik in range(main_class.nk):
    #         for iv in range(main_class.nv_for_r):
    #             for ic in range(main_class.nc_for_r):
    #                 energy_diff_for_cancel_diple = main_class.energy_dft_full[ik,iv] - main_class.energy_dft_full[ik,ic+main_class.nv_for_r]
    #                 energy_diff_for_cancel_diple_inv = 1/energy_diff_for_cancel_diple
    #                 totx[ik,iv,ic] = totx[ik,iv,ic]*energy_diff_for_cancel_diple_inv
    #                 toty[ik, iv, ic] = toty[ik, iv, ic] * energy_diff_for_cancel_diple_inv
    #                 totz[ik, iv, ic] = totz[ik, iv, ic] * energy_diff_for_cancel_diple_inv
    #
    #
    #     L[:, :, :, 0] = totx * fact
    #     L[:, :, :, 1] = toty * fact
    #     L[:, :, :, 2] = totz * fact
    #     print('finish calculating L_kvc')
    #
    #     return L[:, main_class.nv_for_r - main_class.nv:main_class.nv_for_r, 0:main_class.nc, :]

    def calculate_L2(self, main_class):
        '''
        calculate orbital angular momentum
        '''
        '''
            # dim [nk,nv,nc,3], orbital angular momentum
            calculate orbital angular momentum

            noeh_dipole: dim: [nk, nb, nb, 3]

        '''

        # m0 = 9.10938356e-31 # mass of electrons
        # eVtoJ = 1.602177e-19 # electron volt = ？Joule
        # p_ry_to_SI = 1.9928534e-24 # convert momentum operator to SI unit # ??
        # fact = (p_ry_to_SI ** 2) / m0 / eVtoJ # pre factor
        fact = 2*1j
        datax = main_class.noeh_dipole_full[:, :, :, 0]
        datay = main_class.noeh_dipole_full[:, :, :, 1]
        dataz = main_class.noeh_dipole_full[:, :, :, 2]



        L = np.zeros([main_class.nk, main_class.nv_for_r, main_class.nc_for_r, 3], dtype=np.complex)

        Ekc = np.einsum('kc,m->kmc', main_class.energy_dft_full[:, main_class.nv_for_r:main_class.nv_for_r + main_class.nc_for_r], np.ones(main_class.nv_for_r + main_class.nc_for_r))
        Ekm = np.einsum('km,c->kmc', main_class.energy_dft_full, np.ones(main_class.nc_for_r))


        energy_diff = ((Ekm-Ekc))  # e_diff(k,v,m) = [E(k,v) - E(k,m)]^-1, unit: ryd


        totx = np.einsum('kvm,kmc,kmc-> kvc', datay[:, 0:main_class.nv_for_r, :], energy_diff,
                         dataz[:, :, main_class.nv_for_r:main_class.nv_for_r + main_class.nc_for_r]) - \
               np.einsum('kvm,kmc,kmc-> kvc', dataz[:, 0:main_class.nv_for_r, :], energy_diff,
                         datay[:, :, main_class.nv_for_r:main_class.nv_for_r + main_class.nc_for_r])
        toty = np.einsum('kvm,kmc,kmc-> kvc', dataz[:, 0:main_class.nv_for_r, :], energy_diff,
                         datax[:, :, main_class.nv_for_r:main_class.nv_for_r + main_class.nc_for_r]) - \
               np.einsum('kvm,kmc,kmc-> kvc', datax[:, 0:main_class.nv_for_r, :], energy_diff,
                         dataz[:, :, main_class.nv_for_r:main_class.nv_for_r + main_class.nc_for_r])
        totz = np.einsum('kvm,kmc,kmc-> kvc', datax[:, 0:main_class.nv_for_r, :], energy_diff,
                         datay[:, :, main_class.nv_for_r:main_class.nv_for_r + main_class.nc_for_r]) - \
               np.einsum('kvm,kmc,kmc-> kvc', datay[:, 0:main_class.nv_for_r, :], energy_diff,
                         datax[:, :, main_class.nv_for_r:main_class.nv_for_r + main_class.nc_for_r])
        for ik in range(main_class.nk):
            for iv in range(main_class.nv_for_r):
                for ic in range(main_class.nc_for_r):
                    energy_diff_for_cancel_diple = main_class.energy_dft_full[ik,iv] - main_class.energy_dft_full[ik,ic+main_class.nv_for_r]
                    energy_diff_for_cancel_diple_inv = 1/energy_diff_for_cancel_diple
                    energy_diff_for_cancel_diple_inv = (energy_diff_for_cancel_diple_inv)
                    totx[ik,iv,ic] = totx[ik,iv,ic]*energy_diff_for_cancel_diple_inv
                    toty[ik, iv, ic] = toty[ik, iv, ic] * energy_diff_for_cancel_diple_inv
                    totz[ik, iv, ic] = totz[ik, iv, ic] * energy_diff_for_cancel_diple_inv


        L[:, :, :, 0] = totx * fact
        L[:, :, :, 1] = toty * fact
        L[:, :, :, 2] = totz * fact
        print('finish calculating L_kvc')

        return L[:, main_class.nv_for_r - main_class.nv:main_class.nv_for_r, 0:main_class.nc, :]

    def calculate_L3(self, main_class):
        '''
        calculate orbital angular momentum
        '''
        '''
            # dim [nk,nv,nc,3], orbital angular momentum
            calculate orbital angular momentum

            noeh_dipole: dim: [nk, nb, nb, 3]

        '''

        # m0 = 9.10938356e-31 # mass of electrons
        # eVtoJ = 1.602177e-19 # electron volt = ？Joule
        # p_ry_to_SI = 1.9928534e-24 # convert momentum operator to SI unit # ??
        # fact = (p_ry_to_SI ** 2) / m0 / eVtoJ # pre factor
        fact = 2*1j
        datax = main_class.noeh_dipole_full[:, :, :, 0]
        datay = main_class.noeh_dipole_full[:, :, :, 1]
        dataz = main_class.noeh_dipole_full[:, :, :, 2]



        #L = np.zeros([main_class.nk, main_class.nv_for_r, main_class.nc_for_r, 3], dtype=np.complex)
        L = np.zeros([main_class.nk, main_class.nv_for_r+main_class.nc_for_r, main_class.nv_for_r+main_class.nc_for_r, 3], dtype=np.complex)
        nk = main_class.nk
        nv = main_class.nv_for_r
        nc = main_class.nc_for_r
        energy_dft_full = main_class.energy_dft_full

        Lx = []
        Ly = []
        Lz = []
        for k in range(nk):
            Lx.append([])
            Ly.append([])
            Lz.append([])
            for v in range(nv+nc):#(nv):
                Lx[k].append([])
                Ly[k].append([])
                Lz[k].append([])
                for c in range(nv+nc):#(nv, nv + nc):
                    totx = 0
                    toty = 0
                    totz = 0
                    for m in range(nv + nc):  # this is the sum over all the bands
                        if np.abs(energy_dft_full[k][m] - energy_dft_full[k][v])<0.0005:
                            degen = 1
                        else:
                            degen = 1

                        curx = datay[k][v][m] * dataz[k][m][c] - dataz[k][v][m] * datay[k][m][c]
                        # curx = datay[k][m][v] * dataz[k][c][m] - dataz[k][m][v] * datay[k][c][m]
                        curx = curx * (energy_dft_full[k][m] - energy_dft_full[k][c])
                        totx += curx * fact*degen

                        cury = dataz[k][v][m] * datax[k][m][c] - datax[k][v][m] * dataz[k][m][c]
                        cury = cury * (energy_dft_full[k][m] - energy_dft_full[k][c])
                        toty += cury * fact*degen

                        curz = datax[k][v][m] * datay[k][m][c] - datay[k][v][m] * datax[k][m][c]
                        curz = curz * (energy_dft_full[k][m] - energy_dft_full[k][c])
                        totz += curz * fact*degen
                    Lx[k][v].append(totx)
                    Ly[k][v].append(toty)
                    Lz[k][v].append(totz)
        L[:, :, :, 0] = np.array(Lx)
        L[:, :, :, 1] = np.array(Ly)
        L[:, :, :, 2] = np.array(Lz)
        for ik in range(main_class.nk):
            for iv in range(nv+nc):
                for ic in range(nv+nc):
                    energy_diff_for_cancel_diple = main_class.energy_dft_full[ik, iv] - main_class.energy_dft_full[
                        ik, ic]
                    energy_diff_for_cancel_diple_inv = 1 / (energy_diff_for_cancel_diple+0.00001)
                    energy_diff_for_cancel_diple_inv = (energy_diff_for_cancel_diple_inv)
                    L[ik, iv, ic,0] = L[ik, iv, ic,0] * energy_diff_for_cancel_diple_inv
                    L[ik, iv, ic,1] = L[ik, iv, ic,1] * energy_diff_for_cancel_diple_inv
                    L[ik, iv, ic,2] = L[ik, iv, ic,2] * energy_diff_for_cancel_diple_inv

        newL = np.zeros_like(L)
        for ik in range(main_class.nk):
            for iv in range(nv+nc):
                for ic in range(nv+nc):
                    newL[ik, iv, ic, 0] = L[ik, iv, ic, 0]+ np.conj(L[ik, ic, iv, 0])
                    newL[ik, iv, ic, 1] = L[ik, iv, ic, 1]+ np.conj(L[ik, ic, iv, 1])
                    newL[ik, iv, ic, 2] = L[ik, iv, ic, 2]+ np.conj(L[ik, ic, iv, 2])

        L_diag = np.zeros_like(L)
        for ik in range(main_class.nk):
            for id in range(3):
                L_diag[ik,:,:,id] = np.diag(np.diag(L[ik,:,:,id]))


        L_her = L_diag.sum()
        print(L_her)


        L_output = newL[:, main_class.nv_for_r - main_class.nv:main_class.nv_for_r,
                   main_class.nv_for_r:main_class.nv_for_r+main_class.nc, :]

        return L_output

    def calculate_L4(self, main_class):
        '''
        calculate orbital angular momentum
        '''
        '''
            # dim [nk,nv,nc,3], orbital angular momentum
            calculate orbital angular momentum

            noeh_dipole: dim: [nk, nb, nb, 3]

        '''

        # m0 = 9.10938356e-31 # mass of electrons
        # eVtoJ = 1.602177e-19 # electron volt = ？Joule
        # p_ry_to_SI = 1.9928534e-24 # convert momentum operator to SI unit # ??
        # fact = (p_ry_to_SI ** 2) / m0 / eVtoJ # pre factor
        fact = 2 * 1j
        datax = main_class.noeh_dipole_full[:, :, :, 0]
        datay = main_class.noeh_dipole_full[:, :, :, 1]
        dataz = main_class.noeh_dipole_full[:, :, :, 2]

        L = np.zeros([main_class.nk, main_class.nv_for_r+main_class.nc_for_r, main_class.nv_for_r+main_class.nc_for_r, 3], dtype=np.complex)

        Ekc = np.einsum('kc,m->kmc', main_class.energy_dft_full, np.ones(main_class.nv_for_r + main_class.nc_for_r))
        Ekm = np.einsum('km,c->kmc', main_class.energy_dft_full, np.ones(main_class.nv_for_r + main_class.nc_for_r))

        energy_diff = ((Ekm - Ekc))  # e_diff(k,v,m) = [E(k,v) - E(k,m)]^-1, unit: ryd

        Ekv2 = np.einsum('kv,m->kvm', main_class.energy_dft_full,
                        np.ones(main_class.nv_for_r + main_class.nc_for_r))
        Ekm2 = np.einsum('km,v->kvm', main_class.energy_dft_full, np.ones(main_class.nv_for_r + main_class.nc_for_r))

        nondegen = np.abs(Ekv2-Ekm2)>main_class.degeneracy_remover

        totx = np.einsum('kvm, kvm,kmc,kmc-> kvc',nondegen, datay, energy_diff,dataz) - \
               np.einsum('kvm, kvm,kmc,kmc-> kvc',nondegen, dataz, energy_diff,datay)
        toty = np.einsum('kvm, kvm,kmc,kmc-> kvc',nondegen, dataz, energy_diff,datax) - \
               np.einsum('kvm, kvm,kmc,kmc-> kvc',nondegen, datax, energy_diff,dataz)
        totz = np.einsum('kvm, kvm,kmc,kmc-> kvc',nondegen, datax, energy_diff,datay) - \
               np.einsum('kvm, kvm,kmc,kmc-> kvc',nondegen, datay, energy_diff,datax)
        for ik in range(main_class.nk):
            for iv in range(main_class.nv_for_r+main_class.nc_for_r):
                for ic in range(main_class.nv_for_r+main_class.nc_for_r):
                    energy_diff_for_cancel_diple = main_class.energy_dft_full[ik, iv] - main_class.energy_dft_full[
                        ik, ic]
                    energy_diff_for_cancel_diple_inv = 1 / (energy_diff_for_cancel_diple+0.000000001)
                    totx[ik, iv, ic] = totx[ik, iv, ic] * energy_diff_for_cancel_diple_inv
                    toty[ik, iv, ic] = toty[ik, iv, ic] * energy_diff_for_cancel_diple_inv
                    totz[ik, iv, ic] = totz[ik, iv, ic] * energy_diff_for_cancel_diple_inv

        L[:, :, :, 0] = totx * fact
        L[:, :, :, 1] = toty * fact
        L[:, :, :, 2] = totz * fact
        newL = np.zeros_like(L)

        for ik in range(main_class.nk):
            for iv in range(main_class.nv_for_r+main_class.nc_for_r):
                for ic in range(main_class.nv_for_r+main_class.nc_for_r):
                    newL[ik, iv, ic, 0] = L[ik, iv, ic, 0]+ np.conj(L[ik, ic, iv, 0])
                    newL[ik, iv, ic, 1] = L[ik, iv, ic, 1]+ np.conj(L[ik, ic, iv, 1])
                    newL[ik, iv, ic, 2] = L[ik, iv, ic, 2]+ np.conj(L[ik, ic, iv, 2])

        L_output = newL[:, main_class.nv_for_r - main_class.nv:main_class.nv_for_r,
                   main_class.nv_for_r:main_class.nv_for_r + main_class.nc, :]
        print('finish calculating L_kvc')

        return L_output
