import numpy as np
import cupy as cp
import basis as base
import grid as grid
import reference as ref
import elliptic as ell
import timestep as ts
import fluxes as flux
import data_management

import matplotlib.pyplot as plt


print('Beginning program...')
# Orders
order = 9  # 9
time_order = 3  # Currently hard-coded for third-order
res_x, res_v = 128, 128  # 256, 256  # 64, 64  # 128, 128
folder = '..\\data\\'
filename = 'test'

# Flags
plot_IC = True

# Initialize reference normalization parameters
print('\nInitializing reference values...')
triplet = np.array([1.0e21, 1.0e3, 1.0])
refs = ref.Reference(triplet=triplet, mass_fraction=1836.15267343)

# Build basis
print('\nInitializing the basis...')
orders = np.array([order, order])
basis = base.Basis2D(orders)

# Initialize grids
print('\nInitializing grids...')
k_est = 0.1
L = 2.0 * np.pi / k_est
print('Domain length is %.3f' % L)
lows = np.array([0, -14*refs.vt_e])
highs = np.array([L, 14*refs.vt_e])
resolutions = np.array([res_x, res_v])
resolutions_ghosts = np.array([res_x+2, res_v+2])
# Make grid
grids = grid.Grid2D(basis=basis, lows=lows, highs=highs, resolutions=resolutions)
# Store grid info for run file
geo_info = np.array([[lows[0], highs[0], resolutions[0], orders[0]],
                     [lows[1], highs[1], resolutions[1], orders[1]]])

# Time parameters
final_time = 50.0  # 30.0  # 3.2
write_time = 0.5

# Build distribution
print('\nInitializing distribution function...')
f0 = grid.Distribution(vt=refs.vt_e, resolutions=resolutions, orders=orders, perturbation=True, omega=3.4047j)
f0.initialize_quad_weights(grids)
f0.initialize(grids)
f0.arr[0, :, :, :] = f0.arr[-2, :, :, :]
f0.arr[-1, :, :, :] = f0.arr[1, :, :, :]
# Optional: delta-f
df0 = f0.grid_flatten() - f0.flatten_no_pert()
cbd0 = np.linspace(np.amin(df0), np.amax(df0), num=100)

# Build elliptic operator
print('\nInitializing elliptic operator...')
fields = ell.Elliptic(poisson_coefficient=refs.charge_density_multiplier)
fields.build_central_flux_operator(grid=grids.x, basis=basis.b1)
fields.invert()

# Build initial field
n0 = f0.moment_zero()
f0f = f0.grid_flatten()
cb0 = np.linspace(np.amin(f0f), np.amax(f0f), num=100)
x, v = np.meshgrid(grids.x.arr[1:-1, :].flatten(), grids.v.arr[1:-1, :].flatten(), indexing='ij')

charge = cp.mean(n0) - n0
electric_field = fields.poisson(charge_density=charge, grid=grids.x, basis=basis.b1)

if plot_IC:
    plt.figure()
    plt.contourf(x, v, df0, cbd0)
    plt.colorbar()
    plt.xlabel('Position x')
    plt.ylabel('Velocity v')
    plt.title('Initial perturbation')
    plt.tight_layout()

    plt.figure()
    plt.contourf(x, v, f0f, cb0)
    plt.colorbar()
    plt.xlabel('Position x')
    plt.ylabel('Velocity v')
    plt.title('Initial condition')
    plt.tight_layout()

    print('Initial field max is ' + str(abs(refs.electron_acceleration_multiplier *
                                            cp.amax(electric_field[1:-1, :]))))

    # True field
    true = -refs.charge_density_multiplier * cp.amax(charge) / grids.x.k1 * cp.cos(grids.x.k1 * grids.x.arr_cp)
    true_p = refs.charge_density_multiplier * cp.amax(charge) / (grids.x.k1 ** 2.0) * cp.sin(grids.x.k1 * grids.x.arr_cp)

    plt.figure()
    plt.plot(grids.x.arr[1:-1, :].flatten(), charge.get().flatten(), 'o--')
    plt.title('Initial charge density')
    plt.grid(True)

    # plt.figure()
    # plt.plot(grids.x.arr[1:-1, :].flatten(), fields.potential.get().flatten(), 'o--')
    # plt.plot(grids.x.arr[1:-1, :].flatten(), true_p[1:-1, :].get().flatten(), 'o--')
    # plt.title('Potential')
    # plt.grid(True)

    plt.figure()
    plt.plot(grids.x.arr.flatten(), electric_field.flatten().get(), 'o--', label='apprx')
    plt.plot(grids.x.arr.flatten(), true.flatten().get(), 'o--', label='true')
    plt.grid(True)
    plt.legend(loc='best')
    plt.title('Field')

    plt.show()


# Flux set-up
print('\nSetting up fluxes...')
fluxes = flux.DGFlux(resolutions=resolutions_ghosts,
                     orders=orders,
                     coefficient=refs.electron_acceleration_multiplier)

# Set up time-stepper
print('\nSetting up time-stepper')
stepper = ts.Stepper(time_order=time_order, space_order=orders[0], write_time=write_time, final_time=final_time)
time_info = np.array([final_time, write_time, stepper.courant, time_order])

# Save initial condition
print('\nSetting up save file...')
save_file = data_management.RunData(folder=folder, filename=filename, shape=f0.arr.shape,
                                    geometry=geo_info, time=time_info, refs=refs)
save_file.create_file(distribution=f0.arr.get(), elliptic=fields, density=n0)

# Time-step loop
print('\nBeginning main loop...')
stepper.main_loop(distribution=f0, basis=basis, elliptic=fields,
                  grids=grids, dg_flux=fluxes, refs=refs, save_file=save_file)

# All done, check out for post-process
print('\nProceeding to post: Plotting now...')
f0_f = f0.grid_flatten()
df_f = f0.grid_flatten() - f0.flatten_no_pert()
cb = np.linspace(np.amin(f0_f), np.amax(f0_f), num=100)
cbd = np.linspace(np.amin(df_f), np.amax(df_f), num=100)

xt = np.tensordot(np.ones_like(stepper.time_array), grids.x.arr[1:-1, :].flatten(),  axes=0)
tt = np.tensordot(stepper.time_array, np.ones_like(grids.x.arr[1:-1, :].flatten()), axes=0)
den_xt = stepper.density.reshape(xt.shape[0], xt.shape[1])
cb_den = np.linspace(np.amin(den_xt), np.amax(den_xt), num=100)

plt.figure()
plt.contourf(x, v, f0_f, cb)
plt.xlabel(r'Position $x$')
plt.ylabel(r'Velocity $v$')
plt.colorbar()
plt.title('Final state')
plt.tight_layout()

plt.figure()
plt.contourf(x, v, df_f, cbd)
plt.xlabel(r'Position $x$')
plt.ylabel(r'Velocity $v$')
plt.colorbar()
plt.title('Difference from equilibrium')
plt.tight_layout()

plt.figure()
plt.semilogy(stepper.time_array, stepper.field_energy, 'o--')
plt.xlabel(r'Simulation time $t$')
plt.ylabel(r'Field energy')
# plt.axis([0, stepper.time_array[-1], 0, np.amax(stepper.field_energy)])
plt.grid(True)
plt.tight_layout()

plt.figure()
plt.contourf(xt, tt, den_xt, cb_den)
plt.xlabel('Space')
plt.ylabel('Time')
plt.colorbar()
plt.tight_layout()

plt.show()

# Bin
# print(basis.b1.up)
# print(basis.b1.xi)
#
# with open('up8.npy', 'rb') as f:
#     up8 = np.load(f)
#
# with open('xi8.npy', 'rb') as f:
#     xi8 = np.load(f)

# up_diff = up8 - basis.b1.up.get()
# xi_diff = xi8 - basis.b1.xi.get()

# print(up_diff)
# print(xi_diff)
# quit()

# Reset basis
# basis.b1.up = cp.asarray(up8)
# basis.b1.xi = cp.asarray(xi8)

# with open('ic.npy', 'rb') as f:
#     other_ic = np.load(f)
#
# f0f = f0.grid_flatten()
#
# diff = f0f - other_ic
#
# plt.figure()
# plt.imshow(diff.T)
# plt.colorbar()
# plt.title('IC diff')
# plt.show()
# # print('Max f0f is ')
# # print(np.amax(f0f))
# plt.figure()
# # plt.plot(grids.v.arr[1:-1, :].flatten(), f0f[1, :])
# plt.plot(f0f[1, :])
# plt.grid(True)
# plt.show()

# Reset jacobians
# grids.x.J = 2.8571428571428745
# grids.v.J = 14.285120394559149
