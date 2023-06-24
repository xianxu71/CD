import numpy as np
import matplotlib.pyplot as plt

x2 = 6.48+1.47j
x3 = 2.03+0.47j

y2 = 0.44-2.03j
y3 = -1.24+6.37j

L2 = (x2+y2*1j)/np.sqrt(2)
L3 = (x3+y3*1j)/np.sqrt(2)

R2 = (x2-y2*1j)/np.sqrt(2)
R3 = (x3-y3*1j)/np.sqrt(2)

plt.figure()

# plt.quiver(0,0,np.real(x2),np.imag(x2),color='k',units='xy',label='x',scale = 1)
# plt.quiver(0,0,np.real(y2),np.imag(y2),color='r',units='xy',label='y',scale = 1)
# plt.quiver(0,0,np.real(L2),np.imag(L2),color='g',units='xy',label='L',scale = 1)
# plt.quiver(0,0,np.real(R2),np.imag(R2),color='b',units='xy',label='R',scale = 1)

plt.quiver(0,0,np.real(x3),np.imag(x3),color='k',units='xy',label='x',scale = 1)
plt.quiver(0,0,np.real(y3),np.imag(y3),color='r',units='xy',label='y',scale = 1)
plt.quiver(0,0,np.real(L3),np.imag(L3),color='g',units='xy',label='L',scale = 1)
plt.quiver(0,0,np.real(R3),np.imag(R3),color='b',units='xy',label='R',scale = 1)

plt.axes().set_aspect('equal')
plt.xlim(-7,7)
plt.ylim(-7,7)
plt.legend()


plt.show()
