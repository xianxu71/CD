import numpy as np

def calculate_L(noeh_dipole_full, energy_dft_full, nk, nv, nc, nv_for_r, nc_for_r, use_eqp, eqp_corr):
    '''
    # dim [nk,nv,nc,3], orbital angular momentum
    calculate orbital angular momentum

    noeh_dipole: dim: [nk, nb, nb, 3]

    '''

    # m0 = 9.10938356e-31 # mass of electrons
    # eVtoJ = 1.602177e-19 # electron volt = ？Joule
    # p_ry_to_SI = 1.9928534e-24 # convert momentum operator to SI unit # ??
    # fact = (p_ry_to_SI ** 2) / m0 / eVtoJ # pre factor
    fact = 1
    datax = noeh_dipole_full[:, :, :, 0]
    datay = noeh_dipole_full[:,:,:,1]
    dataz = noeh_dipole_full[:, :, :, 2]

    L = np.zeros([nk, nv_for_r, nc_for_r, 3], dtype=np.complex)


    Ekv = np.einsum('kv,m->kvm', energy_dft_full[:, 0:nv_for_r], np.ones(nv_for_r + nc_for_r))
    Ekm = np.einsum('km,v->kvm', energy_dft_full, np.ones(nv_for_r))

    energy_diff = (Ekv - Ekm)  # e_diff(k,v,m) = [E(k,v) - E(k,m)]^-1
    with np.errstate(divide='ignore'):
        energy_diff_inverse = 1 / energy_diff
        energy_diff_inverse[abs(energy_diff) < 0.002] = 0

    totx = np.einsum('kvm,kvm,kmc-> kvc' , datay[:,0:nv_for_r,:],energy_diff_inverse,dataz[:,:,nv_for_r:nv_for_r+nc_for_r]) - \
           np.einsum('kvm,kvm,kmc-> kvc' , dataz[:,0:nv_for_r,:],energy_diff_inverse,datay[:,:,nv_for_r:nv_for_r+nc_for_r])
    toty = np.einsum('kvm,kvm,kmc-> kvc' , dataz[:,0:nv_for_r,:],energy_diff_inverse,datax[:,:,nv_for_r:nv_for_r+nc_for_r]) - \
           np.einsum('kvm,kvm,kmc-> kvc' , datax[:,0:nv_for_r,:],energy_diff_inverse,dataz[:,:,nv_for_r:nv_for_r+nc_for_r])
    totz = np.einsum('kvm,kvm,kmc-> kvc' , datax[:,0:nv_for_r,:],energy_diff_inverse,datay[:,:,nv_for_r:nv_for_r+nc_for_r]) - \
           np.einsum('kvm,kvm,kmc-> kvc' , datay[:,0:nv_for_r,:],energy_diff_inverse,datax[:,:,nv_for_r:nv_for_r+nc_for_r])
    L[:, :, :, 0] = totx*fact
    L[:, :, :, 1] = toty*fact
    L[:, :, :, 2] = totz*fact

    return L[:,nv_for_r-nv:nv_for_r,0:nc,:]

def calculate_ElectricDipole(noeh_dipole, nk, nv, nc, energy_dft):
    '''
    # dim [nk,nv,nc,3], electric dipole
    '''
    # m0 = 9.10938356e-31
    # eVtoJ = 1.602177e-19
    # p_ry_to_SI = 1.9928534e-24
    # fact = (p_ry_to_SI ** 2) / m0 / eVtoJ   # why there is a **2 ?
    fact = 1 #test
    E = noeh_dipole[:,0:nv,nv:nc+nv,:]*fact
    return E

def calculate_MM_ME(nc,nv,nk,nxct,avck,E,L_kvc):

    idx = list(range(nv - 1, -1, -1))

    inds = np.ix_(range(nk), idx, range(nv), range(3))
    E_temp = E[inds]
    L_temp = L_kvc[inds]

    MM = np.einsum('kvcs,kvcd->sd', avck, L_temp)
    ME = np.einsum('kvcs,kvcd->sd', avck, E_temp)
    return MM, ME

def calculate_L2(noeh_dipole_full, energy_dft_full, nk, nv, nc, nv_for_r, nc_for_r,use_eqp, eqp_corr):
    '''
    # dim [nk,nv,nc,3], orbital angular momentum
    calculate orbital angular momentum

    noeh_dipole: dim: [nk, nb, nb, 3]

    '''

    # m0 = 9.10938356e-31 # mass of electrons
    # eVtoJ = 1.602177e-19 # electron volt = ？Joule
    # p_ry_to_SI = 1.9928534e-24 # convert momentum operator to SI unit # ??
    # fact = (p_ry_to_SI ** 2) / m0 / eVtoJ # pre factor
    fact = 1
    RYD = 13.6057039763
    datax = noeh_dipole_full[:, :, :, 0]
    datay = noeh_dipole_full[:,:,:,1]
    dataz = noeh_dipole_full[:, :, :, 2]
    #dataz = dataz/dataz

    L = np.zeros([nk, nv_for_r, nc_for_r, 3], dtype=np.complex)
    #energy_dft_full = energy_dft_full + eqp_corr/RYD

    Lx = []
    Ly = []
    Lz = []
    for k in range(nk):
        Lx.append([])
        Ly.append([])
        Lz.append([])
        for v in range(nv):
            Lx[k].append([])
            Ly[k].append([])
            Lz[k].append([])
            for c in range(nv, nv+nc):
                totx = 0
                toty = 0
                totz = 0
                for m in range(nv+nc):  # this is the sum over all the bands
                    if m == v:  # these if statements are to avoid dividing by 0 and having m = v
                        continue
                    elif abs(energy_dft_full[k][v] - energy_dft_full[k][m]) <0.002:
                        continue

                    curx = datay[k][v][m] * dataz[k][m][c] - dataz[k][v][m] * datay[k][m][c]
                    #curx = datay[k][m][v] * dataz[k][c][m] - dataz[k][m][v] * datay[k][c][m]
                    curx = curx / (energy_dft_full[k][v] - energy_dft_full[k][m])
                    totx += curx * fact


                    cury = dataz[k][v][m] * datax[k][m][c] - datax[k][v][m] * dataz[k][m][c]
                    cury = cury / (energy_dft_full[k][v] - energy_dft_full[k][m])
                    toty += cury * fact

                    curz = datax[k][v][m] * datay[k][m][c] - datay[k][v][m] * datax[k][m][c]
                    curz = curz / (energy_dft_full[k][v] - energy_dft_full[k][m])
                    totz += curz * fact
                # if tot.real != 0.0:
                #     print(k, v, tot)
                Lx[k][v].append(totx)
                Ly[k][v].append(toty)
                Lz[k][v].append(totz)
    L[:,:,:,0] = Lx
    L[:, :, :, 1] = Ly
    L[:, :, :, 2] = Lz






    #print('test')

    #energy_dft_full = energy_dft_full #+ eqp_corr/RYD

    # Ekv = np.einsum('kv,m->kvm', energy_dft_full[:, 0:nv_for_r], np.ones(nv_for_r + nc_for_r))
    # Ekm = np.einsum('km,v->kvm', energy_dft_full, np.ones(nv_for_r))
    #
    # energy_diff = (Ekv - Ekm)  # e_diff(k,v,m) = [E(k,v) - E(k,m)]^-1
    # with np.errstate(divide='ignore'):
    #     energy_diff_inverse = 1 / energy_diff
    #     energy_diff_inverse[energy_diff == 0] = 0
    #
    # totx = np.einsum('kvm,kvm,kmc-> kvc' , datay[:,0:nv_for_r,:],energy_diff_inverse,dataz[:,:,nv_for_r:nv_for_r+nc_for_r]) - \
    #        np.einsum('kvm,kvm,kmc-> kvc' , dataz[:,0:nv_for_r,:],energy_diff_inverse,datay[:,:,nv_for_r:nv_for_r+nc_for_r])
    # toty = np.einsum('kvm,kvm,kmc-> kvc' , dataz[:,0:nv_for_r,:],energy_diff_inverse,datax[:,:,nv_for_r:nv_for_r+nc_for_r]) - \
    #        np.einsum('kvm,kvm,kmc-> kvc' , datax[:,0:nv_for_r,:],energy_diff_inverse,dataz[:,:,nv_for_r:nv_for_r+nc_for_r])
    # totz = np.einsum('kvm,kvm,kmc-> kvc' , datax[:,0:nv_for_r,:],energy_diff_inverse,datay[:,:,nv_for_r:nv_for_r+nc_for_r]) - \
    #        np.einsum('kvm,kvm,kmc-> kvc' , datay[:,0:nv_for_r,:],energy_diff_inverse,datax[:,:,nv_for_r:nv_for_r+nc_for_r])
    # L[:, :, :, 0] = totx*fact
    # L[:, :, :, 1] = toty*fact
    # L[:, :, :, 2] = totz*fact

    return L[:,nv_for_r-nv:nv_for_r,0:nc,:]