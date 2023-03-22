import numpy as np

def calculate_L(noeh_dipole, energy_dft, nk, nv, nc ):
    '''
    # dim [nk,nv,nc,3], orbital angular momentum
    calculate orbital angular momentum

    noeh_dipole: dim: [nk, nb, nb, 3]

    '''

    m0 = 9.10938356e-31 # mass of electrons
    eVtoJ = 1.602177e-19 # electron volt = ？Joule
    p_ry_to_SI = 1.9928534e-24 # convert momentum operator to SI unit # ??
    fact = (p_ry_to_SI ** 2) / m0 / eVtoJ # pre factor
    datax = noeh_dipole[:, :, :, 0]
    datay = noeh_dipole[:,:,:,1]
    dataz = noeh_dipole[:, :, :, 2]

    L = np.zeros([nk, nv, nc, 3], dtype=np.complex)

    Ekv = np.einsum('kv,m->kvm', energy_dft[:, 0:nv], np.ones(nv + nc))
    Ekm = np.einsum('km,v->kvm', energy_dft, np.ones(nv))

    energy_diff = (Ekv - Ekm)  # e_diff(k,v,m) = [E(k,v) - E(k,m)]^-1
    with np.errstate(divide='ignore'):
        energy_diff_inverse = 1 / energy_diff
        energy_diff_inverse[energy_diff == 0] = 0

    totx = np.einsum('kvm,kvm,kmc-> kvc' , datay[:,0:nv,:], energy_diff_inverse,dataz[:,:,nv:nv+nc]) - \
           np.einsum('kvm,kvm,kmc-> kvc' , dataz[:,0:nv,:],energy_diff_inverse,datay[:,:,nv:nv+nc])
    toty = np.einsum('kvm,kvm,kmc-> kvc' , dataz[:,0:nv,:],energy_diff_inverse,datax[:,:,nv:nv+nc]) - \
           np.einsum('kvm,kvm,kmc-> kvc' , datax[:,0:nv,:],energy_diff_inverse,dataz[:,:,nv:nv+nc])
    totz = np.einsum('kvm,kvm,kmc-> kvc' , datax[:,0:nv,:],energy_diff_inverse,datay[:,:,nv:nv+nc]) - \
           np.einsum('kvm,kvm,kmc-> kvc' , datay[:,0:nv,:],energy_diff_inverse,datax[:,:,nv:nv+nc])
    L[:, :, :, 0] = totx*fact
    L[:, :, :, 1] = toty*fact
    L[:, :, :, 2] = totz*fact

    return L

def calculate_ElectricDipole(noeh_dipole, nk, nv, nc, energy_dft):
    '''
    # dim [nk,nv,nc,3], electric dipole
    '''
    m0 = 9.10938356e-31
    eVtoJ = 1.602177e-19
    p_ry_to_SI = 1.9928534e-24
    fact = (p_ry_to_SI ** 2) / m0 / eVtoJ   # why there is a **2 ?
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