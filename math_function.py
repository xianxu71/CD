import numpy as np

def delta_lorentzian(x, xc, eta):

    return (eta / np.pi) / ( (x-xc) ** 2 + eta ** 2)

def delta_gauss(x,xc,eta):
    return np.exp(-(x-xc)**2/(2*(eta)**2))/(np.sqrt(2*np.pi)*eta)