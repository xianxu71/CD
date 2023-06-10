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

    vx = b1[0] * v1 + b2[0] * v2 + b3[0] * v3
    vy = b1[1] * v1 + b2[1] * v2 + b3[1] * v3
    vz = b1[2] * v1 + b2[2] * v2 + b3[2] * v3

    return vx, vy, vz
