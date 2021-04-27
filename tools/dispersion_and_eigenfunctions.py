import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp

import pyvista as pv

# grids
v = np.linspace(-8, 8, num=200)
vx = np.tensordot(v, np.ones_like(v), axes=0)
vy = np.tensordot(np.ones_like(v), v, axes=0)
# complex phase velocity
z = vx + 1j * vy
# wave-numbers
k = np.linspace(1.0e-6, 1, num=100)
# thermal velocity
vt = 2.0 ** 0.5
om_p = 1.0

# 3-grids
scale = 1.0
k3 = np.tensordot(scale * k, np.ones_like(vx), axes=0)
u3 = np.tensordot(np.ones_like(k), vx, axes=0)
v3 = np.tensordot(np.ones_like(k), vy, axes=0)
grid = pv.StructuredGrid(k3, u3, v3)

# plasma dispersion functions / parameters
# gaussian speeds
vg0 = 0.0
vg1 = 6.0
vg2 = -6.0
# hilbert transforms
sqrt_pi = np.pi ** 0.5
def pd_func(zeta):
    return 1j * sqrt_pi * np.exp(-zeta ** 2.0) * (1.0 + sp.erf(1j * zeta))

# Z = (np.pi ** 0.5) * sp.wofz(z) / 1j
# fh = 1.0 + z * Z

vb0 = (z - vg0) / vt
vb1 = (z - vg1) / vt
vb2 = (z - vg2) / vt

### Don't trust fadeeva function
Z0 = pd_func(vb0)
Z1 = pd_func(vb1)
Z2 = pd_func(vb2)
# fh = ( (1.0 + vb0 * Z0) ) # + (1.0 + vb1 * Z1) + (1.0 + vb2 * Z2) ) / 3.0
fh = 0.5 * ( (1.0 + vb1 * Z1) + (1.0 + vb2 * Z2) )

# dispersion function
kx = 0.1
D = 1.0 + np.tensordot(np.divide(1.0, k ** 2.0), fh, axes=0)
Dk = 1.0 + fh / (kx ** 2.0)

# distribution
fp0 = - (v - vg0) / (0.5 * vt ** 2.0) * np.exp(-(v - vg0)**2.0 / vt**2.0) / (vt * sqrt_pi)
fp1 = - (v - vg1) / (0.5 * vt ** 2.0) * np.exp(-(v - vg1)**2.0 / vt**2.0) / (vt * sqrt_pi)
fp2 = - (v - vg2) / (0.5 * vt ** 2.0) * np.exp(-(v - vg2)**2.0 / vt**2.0) / (vt * sqrt_pi)
# fp = (fp0 + fp1 + fp2) / 3.0
fp = 0.5 * (fp1 + fp2)

L = 2.0 * np.pi / kx
x = np.linspace(0, L, num=100)

v_part = np.divide(fp, (v - (3.7305 - 1.866j))) # (5.2889 - 0.0003139j)))# (2.0 + 0.7741j)))# 1.499j)) # + fp / (v + 1.5j)

psi = -1j * np.tensordot(np.exp(1j * kx * x), v_part, axes=0) # * 0.01 * np.exp(1.768 * kx * 25)
x2 = np.tensordot(x, np.ones_like(v), axes=0)
v2 = np.tensordot(np.ones_like(x), v, axes=0)

f1 = np.exp(-(v-vg1) ** 2.0 / vt ** 2.0) / (vt * sqrt_pi)
f2 = np.exp(-(v-vg2) ** 2.0 / vt ** 2.0) / (vt * sqrt_pi)
f = 0.5 * np.tensordot(np.ones_like(x), (f1 + f2), axes=0)
fpx = 0.5 * np.tensordot(np.ones_like(x), fp, axes=0)

rpsi = np.real(psi) # + f
cb = np.linspace(np.amin(rpsi), np.amax(rpsi), num=100)
cbf = np.linspace(np.amin(f), np.amax(f), num=100)
cbp = np.linspace(np.amin(fp), np.amax(fp), num=100)

plt.figure()
plt.contourf(x2, v2, rpsi, cb)
plt.xlabel('Position x')
plt.ylabel('Velocity v')
plt.colorbar()
plt.tight_layout()

idx_z = np.where(np.absolute(v - 2.0) <= 1.0e-3) # np.amin(np.absolute(v)))
idx_k = np.where(np.absolute(k - 0.2) <= 1.0e-2)
# print(idx_z)
# print(idx_k)
# print(k[idx_k])
# quit()
# print(v3[0, 66, 0])

plt.figure()
# plt.contour(k3[:, 66, :], v3[:, 66, :], np.real(D[:, 66, :]), 0, colors='r')
# plt.contour(k3[:, 66, :], v3[:, 66, :], np.imag(D[:, 66, :]), 0, colors='b')
plt.contour(u3[60, :, :], v3[60, :, :], np.real(Dk[:, :]), 0, colors='r')
plt.contour(u3[60, :, :], v3[60, :, :], np.imag(Dk[:, :]), 0, colors='b')
# print(k3[60, 0, 0])

plt.show()

grid["vol"] = np.real(D).flatten(order='F')
contours0 = grid.contour([0])
# Imag zero contour
grid["vol"] = np.imag(D).flatten(order='F')
contours1 = grid.contour([0])

p = pv.Plotter()
p.add_mesh(grid.outline(), color='k')
p.add_mesh(contours0, color='r', label='Real zero')
p.add_mesh(contours1, color='g', label='Imaginary zero')
p.add_legend()
p.show_grid(xlabel=r'Wavenumber (norm. to Debye length) x ' + str(scale), 
            ylabel='Real phase velocity (norm. to thermal vel.)', 
            zlabel='Imaginary phase velocity (norm. to thermal vel.)')
p.show()
