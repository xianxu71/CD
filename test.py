import numpy as np

a = np.array([[ 8.425800753,   0.000000000,  -0.131321631],
              [ 0.000000000,   7.517499108,   0.000000000],
              [ -1.532446824,   0.000000000,  18.676624778]])

V = np.inner(np.cross(a[0],a[1]),a[2])

b1 = 2*np.pi/V * (np.cross(a[1],a[2]))
b2 = 2*np.pi/V * (np.cross(a[2],a[0]))
b3 = 2*np.pi/V * (np.cross(a[0],a[1]))

b1 = b1/np.sqrt(np.inner(b1,b1))
b2 = b2/np.sqrt(np.inner(b2,b2))
b3 = b3/np.sqrt(np.inner(b3,b3))

v1 = np.array([1,2,3,4])
v2 = np.array([5,6,7,8])
v3 = np.array([9,10,11,12])

vx = b1[0]*v1+b2[0]*v2+b3[0]*v3
vy = b1[1]*v1+b2[1]*v2+b3[1]*v3
vz = b1[2]*v1+b2[2]*v2+b3[2]*v3

print(b1)
print(b2)
print(b3)

print(vx)
print(vy)
print(vz)