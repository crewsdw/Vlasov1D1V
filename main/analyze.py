import data_management as dm
import reference
import grid as g
import basis as b

import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Try an auto-encoder
# import tensorflow as tf
# from tensorflow.keras import layers, losses
# from tensorflow.keras.models import Model


# space-time viz
import pyvista as pv


def grid_flatten(arr, res, order):
    return arr.reshape(res[0] * order[0], res[1] * order[1])


# Filename
folder = '..\\data\\'
filename = 'test'

# flags
save_animation = False
spacetime_3d = True

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

flat = distribution[:, 1:-1, :, 1:-1, :].reshape(time.shape[0],
                                                 grids.x.res * grids.x.order,
                                                 grids.v.res * grids.v.order)
flat2 = np.flip(flat, axis=0)
flat3 = np.concatenate((flat, flat2), axis=0)
# data_matrix = flat.reshape(-1, flat.shape[1] * flat.shape[2]).shape

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

# 3D viz
if spacetime_3d:
    shrink = 1.0
    grow = 1.0
    t3, x3, v3 = np.meshgrid(shrink * time, grow * grids.x.arr[1:-1, :].flatten(),
                             grow * grids.v.arr[1:-1, :].flatten(), indexing='ij')
    grid = pv.StructuredGrid(t3, x3, v3)
    # Add volume info
    grid['f'] = flat.transpose().flatten()
    low, high = np.amin(flat), np.amax(flat)
    # Make contours
    contour_array = np.linspace(low, high, num=8)
    contours = grid.contour(contour_array)
    # plot
    p = pv.Plotter()
    actor = p.add_mesh(contours, cmap='plasma')  # , opacity='sigmoid')
    p.show_grid()
    p.show(auto_close=False)
    view_up = [1, 0, 0]
    path = p.generate_orbital_path(n_points=72, shift=contours.length, viewup=viewup)
    p.open_movie('orbit.mp4')
    p.orbit_on_path(path, write_frames=True, viewup=view_up)
    p.close()

if save_animation:
    fig, ax = plt.subplots(figsize=(12,12))
    xf = grids.x.arr[1:-1, :].flatten()
    vf = grids.v.arr[1:-1, :].flatten()
    XF, VF = np.meshgrid(xf, vf, indexing='ij')
    ax.axis('equal')
    ax.axis('off')
    cb = np.linspace(np.amin(flat), np.amax(flat))
    cf = ax.contourf(XF, VF, flat[0, :, :], cb)

    # Make a contour plot movie
    def animate(i):
        idx = i
        global cf
        
        # Update plot
        for coll in cf.collections:
            coll.remove()

        cf = ax.contourf(XF, VF, flat3[idx, :, :], cb)
        return cf

    anim = animation.FuncAnimation(fig, animate, frames=2*flat.shape[0]-1, interval=30, blit=False, repeat=False)
    anim.save('streams.gif', writer='imagemagick', fps=20)


# # Try an auto-encoder ... plain vanilla auto-encoder
# class Autoencoder(Model):
#     def __init__(self, latent_dim):
#         super(Autoencoder, self).__init__()
#         self.latent_dim = latent_dim
#         self.encoder = tf.keras.Sequential([layers.Flatten(),
#                                             layers.Dense(latent_dim, activation='relu'), ])
#         self.decoder = tf.keras.Sequential([layers.Dense(flat.shape[1] * flat.shape[2], activation='sigmoid'),
#                                             layers.Reshape((flat.shape[1], flat.shape[2]))
#                                             ])
#
#     def call(self, x):
#         encoded = self.encoder(x)
#         decoded = self.decoder(encoded)
#         return decoded
#
#
# checkpoint_path = "..\\data\\cp.ckpt"
# checkpoint_dir = os.path.dirname(checkpoint_path)
# cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
#                                                  save_weights_only=True,
#                                                  verbose=1)
#
#
# autoencoder = Autoencoder(latent_dim=100)
# autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())
# autoencoder.fit(flat, flat,
#                 epochs=100, shuffle=True,
#                 validation_data=(flat, flat),
#                 callbacks=[cp_callback])
# autoencoder.summary()
#
# encoded_imgs = autoencoder.encoder(flat).numpy()
# decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()
#
# n = 3
# plt.figure(figsize=(20, 4))
# for i in range(n):
#     # display original
#     ax = plt.subplot(2, n, i + 1)
#     plt.imshow(flat[i * 50, :, :])
#     plt.title("original")
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
#
#     # display reconstruction
#     ax = plt.subplot(2, n, i + 1 + n)
#     plt.imshow(decoded_imgs[i * 50, :, :])
#     plt.title("reconstructed")
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
#
# plt.show()



