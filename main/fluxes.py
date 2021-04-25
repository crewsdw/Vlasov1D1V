import numpy as np
import cupy as cp

# For debug
import matplotlib.pyplot as plt


def basis_product(flux, basis_arr, axis, permutation):
    return cp.transpose(cp.tensordot(flux, basis_arr,
                                     axes=([axis], [1])),
                        axes=permutation)


class DGFlux:
    def __init__(self, resolutions, orders, coefficient):
        self.resolutions = resolutions
        self.orders = orders
        # Permutations
        self.permutations = [(0, 3, 1, 2), (0, 1, 2, 3)]
        # Boundary slices
        self.boundary_slices = [[(slice(resolutions[0]), 0,
                                  slice(resolutions[1]), slice(orders[1])),
                                 (slice(resolutions[0]), -1,
                                  slice(resolutions[1]), slice(orders[1]))],
                                [(slice(resolutions[0]), slice(orders[0]),
                                  slice(resolutions[1]), 0),
                                 (slice(resolutions[0]), slice(orders[0]),
                                  slice(resolutions[1]), -1)]]
        # Speed slices
        self.speed_slices = [[(None, slice(resolutions[1]), slice(orders[1])),
                              (None, slice(resolutions[1]), slice(orders[1]))],
                             [(slice(resolutions[0]), slice(orders[0]), None),
                              (slice(resolutions[0]), slice(orders[0]), None)]]
        # Grid and sub-element axis
        # self.grid_axis = np.array([2, 0])
        # self.sub_element_axis = np.array([3, 1])
        self.grid_axis = np.array([0, 2])
        self.sub_element_axis = np.array([1, 3])
        # acceleration coefficient
        self.multiplier = coefficient
        # Numerical flux
        self.num_flux_sizes = [(resolutions[0], 2, resolutions[1], orders[1]),
                               (resolutions[0], orders[0], resolutions[1], 2)]
        # self.num_flux_sizes = [(resolutions[0], orders[0], resolutions[1], 2),
        #                        (resolutions[0], 2, resolutions[1], orders[1])]  # now reversed

    def semi_discrete_rhs(self, function, field, basis, grids, debug=False):
        """
        Calculate the right-hand side of semi-discrete equation
        """
        return cp.add((grids.x.J * self.x_flux(function=function, basis=basis.b1,
                                               grid_v=grids.v, debug=debug)),  # +
                      grids.v.J * (self.v_flux(function=function, basis=basis.b2,
                                               field=field, debug=debug)))  #

    def x_flux(self, function, basis, grid_v, debug):
        dim = 0
        flux = cp.multiply(function, grid_v.arr_cp[None, None, :, :])
        if debug:
            print('x-flux...')
            internal = basis_product(flux=flux, basis_arr=basis.up,
                                     axis=self.sub_element_axis[dim],
                                     permutation=self.permutations[dim])
            numerical = self.spatial_flux(flux=flux, speed=grid_v,
                                          basis=basis, dim=dim)
            ff = (internal - numerical)
            internal_f = internal[1:-1, :, 1:-1, :].reshape((self.resolutions[0] - 2) * self.orders[0],
                                                            (self.resolutions[1] - 2) * self.orders[1]).get()
            numerical_f = numerical[1:-1, :, 1:-1, :].reshape((self.resolutions[0] - 2) * self.orders[0],
                                                              (self.resolutions[1] - 2) * self.orders[1]).get()
            ff_ng = ff[1:-1, :, 1:-1, :].reshape((self.resolutions[0] - 2) * self.orders[0],
                                                 (self.resolutions[1] - 2) * self.orders[1]).get()
            flux_f = flux[1:-1, :, 1:-1, :].reshape((self.resolutions[0] - 2) * self.orders[0],
                                                    (self.resolutions[1] - 2) * self.orders[1]).get()
            function_f = function.reshape((self.resolutions[0]) * self.orders[0],
                                          (self.resolutions[1]) * self.orders[1]).get()
            full_flux_f = flux.reshape((self.resolutions[0]) * self.orders[0],
                                       (self.resolutions[1]) * self.orders[1]).get()
            l1, h1 = 590, 620
            l2, h2 = 390, 420

            plt.figure()
            plt.imshow(full_flux_f[(l1 + 8):(h1 + 8), (l2 + 8):(h2 + 8)].T, cmap='gray')
            plt.title('Full flux')
            plt.colorbar()

            plt.figure()
            plt.imshow(internal_f[l1:h1, l2:h2].T, cmap='gray')
            plt.colorbar()
            plt.title('Internal')

            plt.figure()
            plt.imshow(numerical_f[l1:h1, l2:h2].T, cmap='gray')
            plt.colorbar()
            plt.title('Numerical')

            plt.figure()
            plt.imshow(function_f[(l1 + 8):(h1 + 8), (l2 + 8):(h2 + 8)].T, cmap='gray')
            plt.colorbar()
            plt.title('Flat func')
            # print(function_f[758:, :23])
            plt.figure()
            plt.imshow(ff_ng[l1:h1, l2:h2].T, cmap='gray')
            plt.colorbar()
            plt.title('Flux gradient')
            plt.figure()
            plt.imshow(flux_f[l1:h1, l2:h2].T, cmap='gray')
            plt.colorbar()
            plt.title('Flux function')
            plt.show()
        # Compute internal and numerical fluxes
        return (basis_product(flux=flux, basis_arr=basis.up,
                              axis=self.sub_element_axis[dim],
                              permutation=self.permutations[dim])
                - self.spatial_flux(flux=flux, speed=grid_v,
                                    basis=basis, dim=dim))

    def v_flux(self, function, basis, field, debug):
        dim = 1
        # o = cp.ones_like(function)
        flux = self.multiplier * cp.multiply(function, field[:, :, None, None])
        # Debug
        if debug:
            print('v-flux')
            speed = self.multiplier * field
            # Alternative:
            one_negatives = cp.where(condition=speed < 0, x=1, y=0)
            one_positives = cp.where(condition=speed >= 0, x=1, y=0)
            # print(one_negatives[self.speed_slices[dim][0]].shape)
            # print(flux[self.boundary_slices[dim][0]].shape)
            # quit()
            # Upwind flux, left face
            num_flux = cp.zeros_like(flux)
            num_flux[self.boundary_slices[dim][0]] = -1.0 * (cp.multiply(flux[self.boundary_slices[dim][0]],
                                                                         one_negatives[self.speed_slices[dim][0]]))
            #             (cp.multiply(cp.roll(flux[self.boundary_slices[dim][1]],
            #                                                 shift=1,
            #                                                    axis=self.grid_axis[dim]),
            #                                                              one_positives[self.speed_slices[dim][0]]))  # +
            # cp.multiply(flux[self.boundary_slices[dim][0]],
            #             one_negatives[self.speed_slices[dim][0]]))
            # Upwind fluxes, right face
            num_flux[self.boundary_slices[dim][1]] = (cp.multiply(flux[self.boundary_slices[dim][1]],
                                                                  one_positives[self.speed_slices[dim][1]]))  # +
            # cp.multiply(cp.roll(flux[self.boundary_slices[dim][0]], shift=-1,
            #                     axis=self.grid_axis[dim]),
            #             one_negatives[self.speed_slices[dim][1]]))

            internal = basis_product(flux=flux, basis_arr=basis.up,
                                     axis=self.sub_element_axis[dim],
                                     permutation=self.permutations[dim])
            numerical = self.velocity_flux(flux=flux, speed=self.multiplier * field,
                                           basis=basis, dim=dim)
            ff = (internal - numerical)
            internal_f = internal[1:-1, :, 1:-1, :].reshape((self.resolutions[0] - 2) * self.orders[0],
                                                            (self.resolutions[1] - 2) * self.orders[1]).get()
            numerical_f = numerical[1:-1, :, 1:-1, :].reshape((self.resolutions[0] - 2) * self.orders[0],
                                                              (self.resolutions[1] - 2) * self.orders[1]).get()
            ff_ng = ff[1:-1, :, 1:-1, :].reshape((self.resolutions[0] - 2) * self.orders[0],
                                                 (self.resolutions[1] - 2) * self.orders[1]).get()
            flux_f = flux[1:-1, :, 1:-1, :].reshape((self.resolutions[0] - 2) * self.orders[0],
                                                    (self.resolutions[1] - 2) * self.orders[1]).get()
            function_f = function.reshape((self.resolutions[0]) * self.orders[0],
                                          (self.resolutions[1]) * self.orders[1]).get()
            full_flux_f = flux.reshape((self.resolutions[0]) * self.orders[0],
                                       (self.resolutions[1]) * self.orders[1]).get()
            num_flux_f = num_flux[1:-1, :, 1:-1, :].reshape((self.resolutions[0] - 2) * self.orders[0],
                                                            (self.resolutions[1] - 2) * self.orders[1]).get()
            pos_neg = cp.copy(flux_f)
            pos_neg[pos_neg > 0] = 1.0
            pos_neg[pos_neg < 0] = -1.0
            l1, h1 = 590, 620
            l2, h2 = 390, 420

            plt.figure()
            plt.imshow(full_flux_f[(l1 + 8):(h1 + 8), (l2 + 8):(h2 + 8)].T, cmap='gray')
            plt.title('Full flux')
            plt.colorbar()

            plt.figure()
            plt.imshow(internal_f[l1:h1, l2:h2].T, cmap='gray')
            plt.colorbar()
            plt.title('Internal')

            plt.figure()
            plt.imshow(numerical_f[l1:h1, l2:h2].T, cmap='gray')
            plt.colorbar()
            plt.title('Numerical')

            plt.figure()
            plt.imshow(num_flux_f[l1:h1, l2:h2].T, cmap='gray')
            plt.colorbar()
            plt.title('Boundary fluxes')

            plt.figure()
            plt.imshow(function_f[(l1 + 8):(h1 + 8), (l2 + 8):(h2 + 8)].T, cmap='gray')
            plt.colorbar()
            plt.title('Flat func')
            # print(function_f[758:, :23])
            plt.figure()
            plt.imshow(ff_ng[l1:h1, l2:h2].T, cmap='gray')
            plt.colorbar()
            plt.title('Flux gradient')
            plt.figure()
            plt.imshow(flux_f[l1:h1, l2:h2].T, cmap='gray')
            plt.colorbar()
            plt.title('Flux function')
            plt.figure()
            plt.imshow(pos_neg[l1:h1, l2:h2].T, cmap='gray')
            plt.colorbar()
            plt.title('Positives/negatives')
            plt.show()

        # Compute internal and numerical fluxes
        return (basis_product(flux=flux, basis_arr=basis.up,
                              axis=self.sub_element_axis[dim],
                              permutation=self.permutations[dim])
                - self.velocity_flux(flux=flux, speed=self.multiplier * field,
                                     basis=basis, dim=dim))

    # noinspection PyTypeChecker
    def spatial_flux(self, flux, speed, basis, dim):
        # Allocate
        num_flux = cp.zeros(self.num_flux_sizes[dim])
        # Debug
        # print(speed.one_positives[self.speed_slices[dim][0]].shape)
        # print(flux[self.boundary_slices[dim][0]].shape)
        # quit()
        # Upwind fluxes, left face
        num_flux[self.boundary_slices[dim][0]] = -1.0 * (cp.multiply(cp.roll(flux[self.boundary_slices[dim][1]],
                                                                             shift=1,
                                                                             axis=self.grid_axis[dim]),
                                                                     speed.one_positives[self.speed_slices[dim][0]]) +
                                                         cp.multiply(flux[self.boundary_slices[dim][0]],
                                                                     speed.one_negatives[self.speed_slices[dim][0]]))
        # Upwind fluxes, right face
        num_flux[self.boundary_slices[dim][1]] = (cp.multiply(flux[self.boundary_slices[dim][1]],
                                                              speed.one_positives[self.speed_slices[dim][1]]) +
                                                  cp.multiply(cp.roll(flux[self.boundary_slices[dim][0]], shift=-1,
                                                                      axis=self.grid_axis[dim]),
                                                              speed.one_negatives[self.speed_slices[dim][1]]))

        return basis_product(flux=num_flux, basis_arr=basis.xi,
                             axis=self.sub_element_axis[dim],
                             permutation=self.permutations[dim])

    # noinspection PyTypeChecker
    def velocity_flux(self, flux, speed, basis, dim):
        # Allocate
        num_flux = cp.zeros(self.num_flux_sizes[dim])
        # Alternative:
        one_negatives = cp.where(condition=speed < 0, x=1, y=0)
        one_positives = cp.where(condition=speed >= 0, x=1, y=0)
        # Debug
        # print(flux[self.boundary_slices[dim][0]].shape)
        # print(one_negatives[self.speed_slices[dim][0]].shape)
        # quit()
        # Upwind flux, left face
        num_flux[self.boundary_slices[dim][0]] = -1.0 * (cp.multiply(cp.roll(flux[self.boundary_slices[dim][1]],
                                                                             shift=1,
                                                                             axis=self.grid_axis[dim]),
                                                                     one_positives[self.speed_slices[dim][0]]) +
                                                         cp.multiply(flux[self.boundary_slices[dim][0]],
                                                                     one_negatives[self.speed_slices[dim][0]]))
        # Upwind fluxes, right face
        num_flux[self.boundary_slices[dim][1]] = (cp.multiply(flux[self.boundary_slices[dim][1]],
                                                              one_positives[self.speed_slices[dim][1]]) +
                                                  cp.multiply(cp.roll(flux[self.boundary_slices[dim][0]], shift=-1,
                                                                      axis=self.grid_axis[dim]),
                                                              one_negatives[self.speed_slices[dim][1]]))

        # Upwind flux, left face
        # num_flux[self.boundary_slices[dim][0]] = -1.0 * (cp.where(condition=speed[self.speed_slices[dim][0]] >= 0,
        #                                                           # Where the *speed* on left face is positive,
        #                                                           x=cp.roll(flux[self.boundary_slices[dim][1]], shift=1,
        #                                                                     axis=self.grid_axis[dim]),
        #                                                           # Then (x) use the left neighbor's right face,
        #                                                           # else (y) zero
        #                                                           y=0) +
        #                                                  cp.where(condition=speed[self.speed_slices[dim][0]] < 0,
        #                                                           # Where the *speed* on left face is negative,
        #                                                           x=flux[self.boundary_slices[dim][0]],
        #                                                           # Then keep the local values, else zero
        #                                                           y=0))
        #
        # # Upwind flux, right face
        # num_flux[self.boundary_slices[dim][1]] = (cp.where(condition=speed[self.speed_slices[dim][1]] >= 0,
        #                                                    # Where the *speed* on the right face is positive,
        #                                                    x=flux[self.boundary_slices[dim][1]],
        #                                                    # Then keep the local values, else zero
        #                                                    y=0) +
        #                                           cp.where(condition=speed[self.speed_slices[dim][1]] < 0,
        #                                                    # Where the *speed* on the right face is negative,
        #                                                    x=cp.roll(flux[self.boundary_slices[dim][0]], shift=-1,
        #                                                              axis=self.grid_axis[dim]),
        #                                                    # Then use the right neighbor's left face, else zero
        #                                                    y=0))

        return basis_product(flux=num_flux, basis_arr=basis.xi,
                             axis=self.sub_element_axis[dim],
                             permutation=self.permutations[dim])

    # def numerical_flux(self, flux, basis, dim):
    #     # Allocate
    #     num_flux = cp.zeros(self.num_flux_sizes[dim])
    #     # Speed
    #     # speed =
    #
    #     # Upwind flux
    #     # Left face
    #     num_flux[self.boundary_slices[dim][0]] = -1.0 * (cp.where(condition=flux[self.boundary_slices[dim][0]] >= 0,
    #                                                               # Where the flux on left face (0) is positive
#                                                               x=cp.roll(flux[self.boundary_slices[dim][1]], shift=1,
#                                                                         axis=self.grid_axis[dim])
#                                                               ,
#                                                               # Then use the left neighbor (-1) right face (1)
#                                                               y=0.0) +  # else zero
#                                                      cp.where(condition=flux[self.boundary_slices[dim][0]] < 0,
#                                                               # Where the flux on left face (0) is negative
#                                                               x=flux[self.boundary_slices[dim][0]],
#                                                               # Then keep local values, else zero
#                                                               y=0.0))
#     # Right face
#     num_flux[self.boundary_slices[dim][1]] = (cp.where(condition=flux[self.boundary_slices[dim][1]] >= 0,
#                                                        # Where the flux on right face (1) is positive
#                                                        x=flux[self.boundary_slices[dim][1]],
#                                                        # Then use the local value, else zero
#                                                        y=0.0) +
#                                               cp.where(condition=flux[self.boundary_slices[dim][1]] < 0,
#                                                        # Where the flux on right face (1) is negative
#                                                        x=cp.roll(flux[self.boundary_slices[dim][0]], shift=-1,
#                                                                  axis=self.grid_axis[dim])
#                                                        ,
#                                                        # Then use the right neighbor (-1) left face (0)
#                                                        y=0.0))
#     # Central
#     # num_flux[self.boundary_slices[dim][0]] = -0.5 * (flux[self.boundary_slices[dim][0]] +
#     #                                                  cp.roll(flux, shift=1,
#                                                       axis=self.grid_axis[dim])[self.boundary_slices[dim][1]])
#     # num_flux[self.boundary_slices[dim][1]] = 0.5 * (flux[self.boundary_slices[dim][1]] +
#     #                                                 cp.roll(flux, shift=-1,
#                                                      axis=self.grid_axis[dim])[self.boundary_slices[dim][0]])
#     # if dim == 1:
#     #     # a = cp.roll(flux, shift=-1,
#     #     #         axis=self.grid_axis[dim])[self.boundary_slices[dim][0]]
#     #     # print(a[50, :, -1])
#     #     # print(a[50, :, -2])
#     #     # print(a[50, :, 0])
#     #     print(num_flux[50, :, -2, :])
#     #     print(num_flux[50, :, 1, :])
#     #     print(flux[50, :, -1, 0])
#     #     print(flux[50, :, -2, -1])
#     #     print(flux[50, :, 0, -1])
#     #     print(flux[50, :, 1, 0])
#     #     quit()
#     #     flux_full = cp.zeros_like(flux)
#     #     # flux_full_1 = cp.zeros_like(flux)
#     #     flux_full[self.boundary_slices[dim][0]] = num_flux[self.boundary_slices[dim][0]]
#     #     flux_full[self.boundary_slices[dim][1]] = num_flux[self.boundary_slices[dim][1]]
#     #     ff0 = flux_full.reshape(self.resolutions[0] * self.orders[0], self.resolutions[1] * self.orders[1])
#     #     # ff1 = flux_full_1.reshape(self.resolutions[0] * self.orders[0], self.resolutions[1] * self.orders[1])
#     #     print('\nFor dim = ' + str(dim) + ' the num. fluxes are')
#     #     plt.figure()
#     #     plt.imshow(ff0.get().T)
#     #     plt.colorbar()
#     #     plt.title('Numerical fluxes')
#     #     plt.show()
#
#     # Return product
#     return basis_product(flux=num_flux, basis_arr=basis.xi,
#                          axis=self.sub_element_axis[dim],
#                          permutation=self.permutations[dim])

# class DGFlux:
# def __init__(self, resolutions, orders, coefficient):
#     self.resolutions = resolutions
#     self.orders = orders
#     # Permutations
#     self.permutations = [(0, 3, 1, 2), (0, 1, 2, 3)]
#     # Boundary slices
#     self.boundary_slices = [[(slice(resolutions[0]), 0,
#                               slice(resolutions[1]), slice(orders[1])),
#                              (slice(resolutions[0]), -1,
#                               slice(resolutions[1]), slice(orders[1]))],
#                             [(slice(resolutions[0]), slice(orders[0]),
#                               slice(resolutions[1]), 0),
#                              (slice(resolutions[0]), slice(orders[0]),
#                               slice(resolutions[1]), -1)]]
#     # Grid and sub-element axis
#     self.grid_axis = np.array([0, 2])
#     self.sub_element_axis = np.array([1, 3])
#     # acceleration coefficient
#     self.multiplier = coefficient
#     # Numerical flux
#     self.num_flux_sizes = [(resolutions[0], 2, resolutions[1], orders[1]),
#                            (resolutions[0], orders[0], resolutions[1], 2)]
#
# def semi_discrete_rhs(self, function, elliptic, basis, grids):
#     """
#     Calculate the right-hand side of semi-discrete equation
#     """
#     # Debug
#     # Check it out
#     xf = (self.x_flux(function=function, basis=basis.b1, grid_v=grids.v) * grids.x.J)
#     xfg = xf[grids.no_ghost_slice].reshape((self.resolutions[0] - 2) * self.orders[0],
#                                            (self.resolutions[1] - 2) * self.orders[1])
#     xff = xf.reshape(self.resolutions[0] * self.orders[0], self.resolutions[1] * self.orders[1])
#     vf = (self.v_flux(function=function, basis=basis.b2, elliptic=elliptic) * grids.v.J)
#     vff = vf.reshape(self.resolutions[0] * self.orders[0], self.resolutions[1] * self.orders[1])
#     print('\nFull flux outputs')
#     plt.figure()
#     plt.imshow(xff.get().T)
#     plt.colorbar()
#     plt.title('Full x-flux')
#     plt.figure()
#     plt.imshow(xfg.get().T)
#     plt.colorbar()
#     plt.title('No ghost full x-flux')
#     plt.figure()
#     plt.imshow(vff.get().T)
#     plt.colorbar()
#     plt.title('Full v-flux')
#     plt.show()
#     return ((self.x_flux(function=function, basis=basis.b1, grid_v=grids.v) * grids.x.J) +
#             (self.v_flux(function=function, basis=basis.b2, elliptic=elliptic) * grids.v.J))
#
# def x_flux(self, function, basis, grid_v):
#     dim = 0
#     flux = cp.multiply(function, grid_v.arr_cp[None, None, :, :])
#     # Debug
#     # Look at it...
#     # print('\nFunction:')
#     # print(function[100:, :, 40, :])
#     # print('\nVelocity grid:')
#     # print(grid_v.arr_cp[40, :])
#     # print('\nFlux product:')
#     # print(flux[100:, :, 40, :])
#     # quit()
#     ff0 = flux.reshape(self.resolutions[0] * self.orders[0], self.resolutions[1] * self.orders[1])
#     # ff1 = flux_full_1.reshape(self.resolutions[0] * self.orders[0], self.resolutions[1] * self.orders[1])
#     print('\nFor dim = ' + str(dim) + ' the full fluxes are')
#     plt.figure()
#     plt.imshow(ff0.get().T)
#     plt.colorbar()
#     plt.title('x-directed flux function')
#     internal = basis_product(flux=flux, basis_arr=basis.up,
#                           axis=self.sub_element_axis[dim],
#                           permutation=self.permutations[dim])
#     numerical = self.numerical_flux(flux=flux, basis=basis, dim=dim)
#     internal_f = internal.reshape(self.resolutions[0] * self.orders[0], self.resolutions[1] * self.orders[1])
#     numerical_f = numerical.reshape(self.resolutions[0] * self.orders[0], self.resolutions[1] * self.orders[1])
#     plt.figure()
#     plt.imshow(internal_f.get().T)
#     plt.colorbar()
#     plt.title('x-directed internal flux')
#     plt.figure()
#     plt.imshow(numerical_f.get().T)
#     plt.colorbar()
#     plt.title('x-directed numerical flux')
#     net = internal_f - numerical_f
#     plt.figure()
#     plt.imshow(net.get().T)
#     plt.title('Net x-flux')
#     plt.colorbar()
#     print('\nInt flux print:')
#     print(internal_f[800:, (407-50)])
#     print(internal_f[800:, (408-50)])
#     print(internal_f[800:, (407+50)])
#     print(internal_f[800:, (408+50)])
#     print('\nNum flux print:')
#     print(numerical_f[800:, (407-50)])
#     print(numerical_f[800:, (408-50)])
#     print(numerical_f[800:, (407+50)])
#     print(numerical_f[800:, (408+50)])
#     print('\nNet print:')
#     print(net[800:, (407 - 50)])
#     print(net[800:, (408 - 50)])
#     print(net[800:, (407 + 50)])
#     print(net[800:, (408 + 50)])
#     print('\nFlux print:')
#     print(ff0[800:, (407 - 50)])
#     print(ff0[800:, (408 - 50)])
#     # print(ff0[:16, (407-50)])
#     print(ff0[800:, (407 + 50)])
#     print(ff0[800:, (408 + 50)])
#     print('Max net is ' + str(cp.amax(net).get()))
#     plt.show()
#     # Compute internal and numerical fluxes
#     return (basis_product(flux=flux, basis_arr=basis.up,
#                           axis=self.sub_element_axis[dim],
#                           permutation=self.permutations[dim])
#             - self.numerical_flux(flux=flux, basis=basis, dim=dim))
#
# def v_flux(self, function, basis, elliptic):
#     dim = 1
#     flux = self.multiplier * cp.multiply(function, elliptic.electric_field[:, :, None, None])
#     # Debug
#     ff0 = flux.reshape(self.resolutions[0] * self.orders[0], self.resolutions[1] * self.orders[1])
#     # ff1 = flux_full_1.reshape(self.resolutions[0] * self.orders[0], self.resolutions[1] * self.orders[1])
#     print('\nFor dim = ' + str(dim) + ' the full fluxes are')
#     plt.figure()
#     plt.imshow(ff0.get().T)
#     plt.colorbar()
#     plt.title('v-directed flux function')
#     internal = basis_product(flux=flux, basis_arr=basis.up,
#                              axis=self.sub_element_axis[dim],
#                              permutation=self.permutations[dim])
#     numerical = self.numerical_flux(flux=flux, basis=basis, dim=dim)
#     internal_f = internal.reshape(self.resolutions[0] * self.orders[0], self.resolutions[1] * self.orders[1])
#     numerical_f = numerical.reshape(self.resolutions[0] * self.orders[0], self.resolutions[1] * self.orders[1])
#     plt.figure()
#     plt.imshow(internal_f.get().T)
#     plt.colorbar()
#     plt.title('v-directed internal flux')
#     plt.figure()
#     plt.imshow(numerical_f.get().T)
#     plt.title('v-directed numerical flux')
#     plt.show()
#     # Compute internal and numerical fluxes
#     return (basis_product(flux=flux, basis_arr=basis.up,
#                           axis=self.sub_element_axis[dim],
#                           permutation=self.permutations[dim])
#             - self.numerical_flux(flux=flux, basis=basis, dim=dim))
#
# def numerical_flux(self, flux, basis, dim):
#     # Allocate
#     num_flux = cp.zeros(self.num_flux_sizes[dim])
#     # Debug
#     a = cp.where(condition=flux[self.boundary_slices[dim][0]] >= 0,
#                  x=1, y=0) + cp.where(condition=flux[self.boundary_slices[dim][0]] < 0, x=-1, y=0)
#     az = cp.zeros_like(flux)
#     az[self.boundary_slices[dim][0]] = a
#     azf = az.reshape(self.resolutions[0] * self.orders[0], self.resolutions[1] * self.orders[1])
#     print('\nLeft faces for dim = ' + str(dim))
#     plt.figure()
#     plt.imshow(azf.get().T)
#     plt.show()
#
#     # Left face
#     num_flux[self.boundary_slices[dim][0]] = -1.0 * (cp.where(condition=flux[self.boundary_slices[dim][0]] >= 0,
#                                                               # Where the flux on left face (0) is positive
#                                                               x=cp.roll(flux[self.boundary_slices[dim][1]], shift=1,
#                                                                         axis=self.grid_axis[dim])
#                                                               ,
#                                                               # Then use the left neighbor (-1) right face (1)
#                                                               y=0.0) +  # else zero
#                                                      cp.where(condition=flux[self.boundary_slices[dim][0]] < 0,
#                                                               # Where the flux on left face (0) is negative
#                                                               x=flux[self.boundary_slices[dim][0]],
#                                                               # Then keep local values, else zero
#                                                               y=0.0))
#     # Right face
#     num_flux[self.boundary_slices[dim][1]] = (
#         # cp.where(condition=flux[self.boundary_slices[dim][1]] >= 0,
#         #                                                # Where the flux on right face (1) is positive
#         #                                                x=flux[self.boundary_slices[dim][1]],
#         #                                                # Then use the local value, else zero
#         #                                                y=0.0)) # +
#                                               cp.where(condition=flux[self.boundary_slices[dim][1]] < 0,
#                                                        # Where the flux on right face (1) is negative
#                                                        x=cp.roll(flux[self.boundary_slices[dim][0]], shift=-1,
#                                                                  axis=self.grid_axis[dim])
#                                                        ,
#                                                        # Then use the right neighbor (-1) left face (0)
#                                                        y=0.0))
#     print(flux.shape)
#     print(num_flux.shape)
#     # print('\nFluxes:')
#     # print(flux[:2, :, 35, 0])
#     # print(flux[-2:, :, 35, 0])
#     # quit()
#     print('\nFlux:')
#     print(flux[100:, 0, 35, :])
#     print(flux[100:, -1, 35, :])
#     print(flux[100:, 0, 65, :])
#     print(flux[100:, -1, 65, :])
#     print('\nNum flux left face:')
#     print(num_flux[100:, 0, 35, :])
#     print(num_flux[100:, 0, 65, :])
#     print('\nNum flux right face:')
#     print(num_flux[100:, 1, 35, :])
#     print(num_flux[100:, 1, 65, :])
#     quit()
#     # Debug
#     flux_full = cp.zeros_like(flux)
#     # flux_full_1 = cp.zeros_like(flux)
#     flux_full[self.boundary_slices[dim][0]] = num_flux[self.boundary_slices[dim][0]]
#     flux_full[self.boundary_slices[dim][1]] = num_flux[self.boundary_slices[dim][1]]
#     ff0 = flux_full.reshape(self.resolutions[0] * self.orders[0], self.resolutions[1] * self.orders[1])
#     # ff1 = flux_full_1.reshape(self.resolutions[0] * self.orders[0], self.resolutions[1] * self.orders[1])
#     print('\nFor dim = ' + str(dim) + ' the num. fluxes are')
#     plt.figure()
#     plt.imshow(ff0.get().T)
#     plt.colorbar()
#     plt.title('Numerical fluxes')
#     plt.show()
#
#     # Return product
#     return basis_product(flux=num_flux, basis_arr=basis.xi,
#                          axis=self.sub_element_axis[dim],
#                          permutation=self.permutations[dim])

# Bin
# print(cp.roll(flux[self.boundary_slices[dim][1]], shift=1,
#               axis=self.grid_axis[dim]).shape)
# quit()
# a = cp.where(condition=speed[self.speed_slices[dim][1]] >= 0,
#              # Where the *speed* on the right face is positive,
#              x=flux[self.boundary_slices[dim][1]],
#              # Then keep the local values, else zero
#              y=0.0)
# b = cp.where(condition=speed[self.speed_slices[dim][1]] < 0,
#              # Where the *speed* on the right face is negative,
#              x=cp.roll(flux[self.boundary_slices[dim][0]], shift=-1,
#                        axis=self.grid_axis[dim]),
#              # Then use the right neighbor's left face, else zero
#              y=0.0)
# if dim == 1:
#     print(speed[self.speed_slices[dim][1]].shape)
#     print(flux[self.boundary_slices[dim][1]].shape)
#     quit()

# flux = jacobian * self.multiplier * cp.multiply(function, field[None, None, :, :])
# flux_fake = self.multiplier * cp.multiply(cp.transpose(function, axes=(2, 3, 0, 1)), field[:, :, None, None])
# print(flux.shape)
# print(self.sub_element_axis[dim])
# print(self.permutations[dim])
# quit()
# size = self.resolutions[1]
# ff = flux[:, :, :, :].reshape(self.resolutions[0] * self.orders[0], size * self.orders[1]).get()
# plt.figure()
# plt.imshow(ff.T, cmap='gray')
# plt.colorbar()
# plt.show()
# plt.figure()
# plt.plot(field.flatten().get(), 'o--')

# def basis_product(flux, basis_arr, axis, permutation):
#     print(flux.shape)
#     print(basis_arr.shape)
#     return cp.transpose(cp.tensordot(flux, basis_arr,
#                                      axes=([1, 3], [1, 3])),
#                         axes=(0, 2, 1, 3))
