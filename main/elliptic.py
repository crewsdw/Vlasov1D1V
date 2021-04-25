import numpy as np
import cupy as cp

import matplotlib.pyplot as plt


class Elliptic:
    def __init__(self, poisson_coefficient):
        # Operators
        self.central_flux_operator = None
        self.gradient_operator = None
        self.inv_op = None

        # Fields
        self.potential = None
        # self.electric_field = None
        self.magnetic_field = None

        # Charge density coefficient in poisson equation
        self.poisson_coefficient = poisson_coefficient

    def build_central_flux_operator(self, grid, basis):
        # Build using indicating array
        indicator = np.zeros((grid.res, grid.order))
        # face differences for numerical flux
        face_diff0 = np.zeros((grid.res, 2))
        face_diff1 = np.zeros((grid.res, 2))
        num_flux = np.zeros_like(face_diff0)
        grad_num_flux = np.zeros_like(face_diff1)

        central_flux_operator = np.zeros((grid.res, grid.order, grid.res, grid.order))
        self.gradient_operator = np.zeros_like(central_flux_operator)

        for i in range(grid.res):
            for j in range(grid.order):
                # Choose node
                indicator[i, j] = 1.0

                # Compute strong form boundary flux (central)
                face_diff0[:, 0] = indicator[:, 0] - np.roll(indicator[:, -1], 1)
                face_diff0[:, 1] = indicator[:, -1] - np.roll(indicator[:, 0], -1)
                # face_diff0[:, 0] = (0.5 * (indicator[:, 0] + np.roll(indicator[:, -1], 1)) -
                #                    indicator[:, 0])
                # face_diff0[:, 1] = -1.0 * (0.5 * (indicator[:, -1] + np.roll(indicator[:, 0], -1)) -
                #                           indicator[:, -1])
                num_flux[:, 0] = 0.5 * face_diff0[:, 0]
                num_flux[:, 1] = -0.5 * face_diff0[:, 1]

                # Compute gradient of this node
                grad = (np.tensordot(basis.der, indicator, axes=([1], [1])) +
                        np.tensordot(basis.np_xi, num_flux, axes=([1], [1]))).T

                # Compute gradient's numerical flux (central)
                face_diff1[:, 0] = grad[:, 0] - np.roll(grad[:, -1], 1)
                face_diff1[:, 1] = grad[:, -1] - np.roll(grad[:, 0], -1)
                grad_num_flux[:, 0] = 0.5 * face_diff1[:, 0]
                grad_num_flux[:, 1] = -0.5 * face_diff1[:, 1]

                # Compute operator from gradient matrix
                operator = (np.tensordot(basis.stf, grad, axes=([1], [1])) +
                            np.tensordot(basis.face_mass, grad_num_flux + face_diff0, axes=([1], [1]))).T

                # place this operator in the global matrix
                central_flux_operator[i, j, :, :] = operator
                self.gradient_operator[i, j, :, :] = grad

                # reset nodal indicator
                indicator[i, j] = 0

        # Reshape to matrix and set gauge condition by fixing quadrature integral = 0 as extra equation in system
        op0 = np.hstack([central_flux_operator.reshape(grid.res * grid.order, grid.res * grid.order),
                         grid.quad_weights.get().reshape(grid.res * grid.order, 1)])
        self.central_flux_operator = np.vstack([op0, np.append(grid.quad_weights.get().flatten(), 0)])
        # Clear machine errors
        self.central_flux_operator[np.abs(self.central_flux_operator) < 1.0e-15] = 0

        # Send gradient operator to device
        self.gradient_operator = cp.asarray(self.gradient_operator)

    def invert(self):
        self.inv_op = cp.asarray(np.linalg.inv(self.central_flux_operator))
        # self.inv_op = np.linalg.inv(self.central_flux_operator)

    def poisson(self, charge_density, grid, basis):
        """
        Poisson solve in 1D using stabilized central flux
        """
        # Preprocess (last entry is average value)
        rhs = cp.zeros((grid.res * grid.order + 1))
        rhs[:-1] = cp.tensordot(self.poisson_coefficient * charge_density, basis.d_mass, axes=([1], [1])).flatten()
        # print(self.poisson_coefficient)
        # Compute solution and remove last entry
        sol = cp.matmul(self.inv_op, rhs)[:-1] / (grid.J ** 2.0)
        self.potential = sol.reshape(grid.res, grid.order)

        # Clean solution (anti-alias)
        coefficients = grid.fourier_basis(self.potential)
        self.potential = grid.sum_fourier(coefficients)

        # Compute gradient
        # self.electric_field = cp.zeros_like(grid.arr_cp)
        electric_field = cp.zeros_like(grid.arr_cp)
        # print(self.gradient_operator.shape)
        # print(self.potential.shape)
        # quit()
        # self.electric_field[1:-1, :] = (grid.J *
        #                                 cp.tensordot(self.gradient_operator, self.potential, axes=([0, 1], [0, 1])))
        electric_field[1:-1, :] = -(grid.J *
                                   cp.tensordot(self.gradient_operator, self.potential, axes=([0, 1], [0, 1])))

        # Clean solution (anti-alias)
        # coefficients = grid.fourier_basis(self.electric_field[1:-1, :])
        # self.electric_field[1:-1, :] = grid.sum_fourier(coefficients)
        #
        # # Set ghost cells
        # self.electric_field[0, :] = self.electric_field[-2, :]
        # self.electric_field[-1, :] = self.electric_field[1, :]

        # Anti-alias
        coefficients = grid.fourier_basis(electric_field[1:-1, :])
        electric_field[1:-1, :] = grid.sum_fourier(coefficients)

        # Set ghost cells
        electric_field[0, :] = electric_field[-2, :]
        electric_field[-1, :] = electric_field[1, :]

        # Return field
        return electric_field

    def set_magnetic_field(self, magnetic_field):
        self.magnetic_field = magnetic_field

    def electric_energy(self, field, grid):
        return cp.tensordot(field[1:-1, :] ** 2.0, grid.quad_weights, axes=([0, 1], [0, 1]))
        # return cp.tensordot(self.electric_field[1:-1, :] ** 2.0, grid.quad_weights, axes=([0, 1], [0, 1]))

# bin
        # Compare operators
        # with open('poisson_operator.npy', 'rb') as file_var:
        #     other_op = np.load(file_var)
        # with open('gradient_operator.npy', 'rb') as file_var:
        #     other_gr = np.load(file_var)
        #
        # diff_op = self.central_flux_operator - other_op
        # diff_gr = self.gradient_operator.reshape(800, 800).T - other_gr
        #
        # plt.figure()
        # plt.imshow(self.central_flux_operator)
        # plt.title('This operator')
        # plt.colorbar()
        #
        # plt.figure()
        # plt.imshow(other_op)
        # plt.title('The other operator')
        # plt.colorbar()
        #
        # plt.figure()
        # plt.imshow(self.gradient_operator.reshape(800, 800))
        # plt.title('This gradient operator')
        # plt.colorbar()
        #
        # plt.figure()
        # plt.imshow(other_gr)
        # plt.title('Other gradient operator')
        # plt.colorbar()
        #
        # plt.figure()
        # plt.imshow(diff_gr)
        # plt.title('Difference of gradients')
        # plt.colorbar()
        #
        # plt.figure()
        # plt.imshow(diff_op)
        # plt.title('Difference of operators')
        # plt.colorbar()
        #
        # plt.show()
