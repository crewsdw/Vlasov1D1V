import data_management as dm
import reference
import grid as g
import basis as b

import numpy as np
import matplotlib.pyplot as plt


def grid_flatten(arr, res, order):
    return arr.reshape(res[0] * order[0], res[1] * order[1])


# Filename
folder = '..\\data\\'
filename = 'test'

# Read data files
save_file = dm.ReadData(folder, filename)
time, distribution, potential, density, field_energy = save_file.read_data()

# Run info
orders, resolutions, lows, highs, time_info, ref_values = save_file.read_info()

# Now refs
refs = reference.Reference(triplet=ref_values, mass_fraction=1836.15267343)
basis = b.Basis2D(orders)
grids = g.Grid2D(basis=basis, lows=lows, highs=highs, resolutions=resolutions)

# Build equilibrium distribution
f0 = g.Distribution(vt=refs.vt_e, resolutions=resolutions, orders=orders, perturbation=True, omega=3.4047j)
f0.initialize_quad_weights(grids)
f0.initialize(grids)

# Visualization
plt.figure()
plt.semilogy(time, field_energy, 'o--')
plt.grid(True)
plt.xlabel('Time t')
plt.ylabel('Field energy')
plt.tight_layout()

plt.figure()
plt.plot(grids.x.arr[1:-1, :].flatten(), density[-1, :, :].flatten(), 'o--')
plt.grid(True)
plt.xlabel(r'Position $x$')
plt.ylabel(r'Density $n(x)$')
plt.tight_layout()

plt.show()
