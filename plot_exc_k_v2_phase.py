import h5py
import numpy as np

evec_filename = './S-NPB_40401_8_8/eigenvectors.h5'
nS_start = 3
nS_end = 3
nc = 1
nv = 1

f = h5py.File(evec_filename,'r')
kpt = f['exciton_header/kpoints/kpts']
num_k = kpt.shape[0]
bvec = f['mf_header/crystal/bvec'][()]
# k = open('kgrid.out')
# k_info = k.readlines()
# k.close()
# del k_info[0:2]


# def AcvS_reading(S,k,c,v): # all arguments follow fortran convention, ex: S=1 represents 1st exciton state
#     imagi=1j
#     AcvS_Re = f['exciton_data/eigenvectors'][0,S-1,k-1,c-1,v-1,0,0]
#     AcvS_Im = f['exciton_data/eigenvectors'][0,S-1,k-1,c-1,v-1,0,1]
#     return AcvS_Re + AcvS_Im * imagi

def AcvS_plot_data(S_start,S_end,c_n,v_n,k_n): # prepare |AcvS(K)|^2 dataFrame: [k1 k2 k3 |AcvS(k)|^2]
    Acv0 = f['exciton_data/eigenvectors'][0,0,:,0:c_n,0:v_n,0,0]+1j*f['exciton_data/eigenvectors'][0,0,:,0:c_n,0:v_n,0,1]
    AcvS_temp_matrix = f['exciton_data/eigenvectors'][0,S_start-1:S_end,:,0:c_n,0:v_n,0,0]+1j*f['exciton_data/eigenvectors'][0,S_start-1:S_end,:,0:c_n,0:v_n,0,1]
    exc = open('phase_exc_wfn_%s-%s.dat'%(S_start,S_end),'w')
    
    for ix in range(S_end-S_start+1):
        for ic in range(c_n):
            for iv in range(v_n):
                for ik in range(k_n):
                    AcvS_temp_matrix[ix,ik,ic,iv] = AcvS_temp_matrix[ix,ik,ic,iv] * np.conj(Acv0[ik,ic,iv])/np.abs(Acv0[ik,ic,iv])
    
    evecs_real = np.real(np.sum(np.sum(AcvS_temp_matrix, axis=3), axis=2))
    v0_real = np.sum(evecs_real[:, :], axis=0)
    norm_real = np.amax(np.abs(v0_real))
    
    evecs_imag = np.imag(np.sum(np.sum(AcvS_temp_matrix, axis=3), axis=2))
    v0_imag = np.sum(evecs_imag[:, :], axis=0)
    norm_imag = np.amax(np.abs(v0_imag))
    
    
    for i in range(num_k):
        exc.write(str(kpt[i][0])+' '+str(kpt[i][1])+' '+str(kpt[i][2])+' '+str(v0_real[i])+' '+str(v0_imag[i])+'\n')
        print("kpt:",i,'/',num_k,'  A(k):', v0_imag[i])
    exc.close()
    
    
    
    
                
        
    
    # for i in range(num_k):
    #     AcvS_2 = np.abs(AcvS_temp_matrix[:,i,:,:,0]+1j*AcvS_temp_matrix[:,i,:,:,1]).sum(axis=(0,1,2))
    #     exc.write(str(kpt[i][0])+' '+str(kpt[i][1])+' '+str(kpt[i][2])+' '+str(AcvS_2)+'\n')
    #     print("kpt:",i,'/',num_k,'  A(k):',AcvS_2)
    # exc.close()

if __name__ == '__main__':
    AcvS_plot_data(nS_start,nS_end,nc,nv,num_k)
    f.close()

