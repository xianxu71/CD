import numpy as np

def delta_lorentzian(x, xc, eta):

    return (eta / np.pi) / ( (x-xc) ** 2 + eta ** 2)

def delta_gauss(x,xc,eta):
    return np.exp(-(x-xc)**2/(2*(eta)**2))/(np.sqrt(2*np.pi)*eta)

def calculate_reflectivity(eps1,eps2):
    a = np.sqrt(0.5*(eps1+np.sqrt(eps1**2+eps2**2)))
    b = np.sqrt(0.5*(-eps1+np.sqrt(eps1**2+eps2**2)))
    R = ((a-1)**2+b**2)/((a+1)**2+b**2)
    return R

def b123_to_xyz(a,v1,v2,v3):
    '''
    convert v in b1 b2 b3 direction to v in x y z direction
    '''
    V = np.inner(np.cross(a[0], a[1]), a[2])

    b1 = 2 * np.pi / V * (np.cross(a[1], a[2]))
    b2 = 2 * np.pi / V * (np.cross(a[2], a[0]))
    b3 = 2 * np.pi / V * (np.cross(a[0], a[1]))

    b1 = b1 / np.sqrt(np.inner(b1, b1))
    b2 = b2 / np.sqrt(np.inner(b2, b2))
    b3 = b3 / np.sqrt(np.inner(b3, b3))

    T = np.array([b1,b2,b3])
    invT = np.linalg.inv(T)

    vx = invT[0][0]*v1+invT[0][1]*v2+invT[0][2]*v3
    vy = invT[1][0]*v1+invT[1][1]*v2+invT[1][2]*v3
    vz = invT[2][0]*v1+invT[2][1]*v2+invT[2][2]*v3


    return vx, vy, vz
