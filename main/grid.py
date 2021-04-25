import numpy as np
import cupy as cp


import matplotlib.pyplot as plt


class Grid1D:
    def __init__(self, low, high, res, basis, spectrum=False):
        self.low = low
        self.high = high
        self.res = int(res)  # somehow gets non-int...
        self.res_ghosts = int(res + 2)  # resolution including ghosts
        self.order = basis.order

        # domain and element widths
        self.length = self.high - self.low
        self.dx = self.length / self.res

        # element Jacobian
        self.J = 2.0 / self.dx

        # The grid does not have a basis but does have quad weights
        self.quad_weights = cp.tensordot(cp.ones(self.res), cp.asarray(basis.weights), axes=0)
        # print(self.quad_weights.shape)
        # arrays
        self.arr = self.create_grid(basis.nodes)
        self.arr_cp = cp.asarray(self.arr)
        self.midpoints = np.array([(self.arr[i, -1] + self.arr[i, 0]) / 2.0 for i in range(1, self.res_ghosts - 1)])
        self.arr_max = np.amax(abs(self.arr))

        # Find where it switches sign (sort of complicated got off stack exchange)
        # idx_low =  # cp.where((cp.diff(cp.sign(self.arr_cp)) != 0)*1 == 1)[0]
        self.one_negatives = cp.where(condition=self.arr_cp < 0, x=1, y=0)
        self.one_positives = cp.where(condition=self.arr_cp >= 0, x=1, y=0)

        # spectral coefficients
        if spectrum:
            self.nyquist_number = self.length // self.dx  # mode number of nyquist frequency
            self.k1 = 2.0 * np.pi / self.length  # fundamental mode
            self.wave_numbers = self.k1 * np.arange(1 - self.nyquist_number, self.nyquist_number)
            self.grid_phases = cp.asarray(np.exp(1j * np.tensordot(self.wave_numbers, self.arr[1:-1, :], axes=0)))

            # Spectral matrices
            self.spectral_transform = basis.fourier_transform_array(self.midpoints, self.J, self.wave_numbers)

    def create_grid(self, nodes):
        """
        Initialize array of global coordinates (including ghost elements).
        """
        # shift to include ghost cells
        min_gs = self.low - self.dx
        max_gs = self.high  # + self.dx
        # nodes (iso-parametric)
        nodes = (np.array(nodes) + 1) / 2

        # element left boundaries (including ghost elements)
        xl = np.linspace(min_gs, max_gs, num=self.res_ghosts)

        # construct coordinates
        self.arr = np.zeros((self.res_ghosts, self.order))
        for i in range(self.res_ghosts):
            self.arr[i, :] = xl[i] + self.dx * nodes

        return self.arr

    def grid2cp(self):
        self.arr = cp.asarray(self.arr)

    def grid2np(self):
        self.arr = self.arr.get()

    def fourier_basis(self, function):
        """
        On GPU, compute Fourier coefficients on the LGL grid of the given grid function
        """
        return cp.tensordot(function, self.spectral_transform, axes=([0, 1], [0, 1])) * self.dx / self.length

    def sum_fourier(self, coefficients):
        """
        On GPU, re-sum Fourier coefficients up to pre-set cutoff
        """
        return cp.real(cp.tensordot(coefficients, self.grid_phases, axes=([0], [0])))


class Grid2D:
    def __init__(self, basis, lows, highs, resolutions):
        # Grids
        self.x = Grid1D(low=lows[0], high=highs[0], res=resolutions[0], basis=basis.b1, spectrum=True)
        self.v = Grid1D(low=lows[1], high=highs[1], res=resolutions[1], basis=basis.b2)
        # No ghost slice
        # self.no_ghost_slice = (slice(1, self.v.res_ghosts - 1), slice(self.v.order),
        #                        slice(1, self.x.res_ghosts - 1), slice(self.x.order))
        # # Ghost slice
        # self.ghost_slice = ([0, -1], slice(self.v.order),
        #                     [0, -1], slice(self.x.order))
        # # No ghost slice
        self.no_ghost_slice = (slice(1, self.x.res_ghosts - 1), slice(self.x.order),
                               slice(1, self.v.res_ghosts - 1), slice(self.v.order))
        # Ghost slice
        self.ghost_slice = ([0, -1], slice(self.x.order),
                            [0, -1], slice(self.v.order))


class Distribution:
    def __init__(self, vt, resolutions, orders):
        # parameters
        self.vt = vt
        # self.ring_j = ring_j
        # array init
        self.arr = None
        self.arr_no_pert = None

        # resolutions
        self.x_res, self.v_res = resolutions[0], resolutions[1]

        # orders
        self.x_ord, self.v_ord = orders[0], orders[1]

        # velocity-space quad weights
        self.quad_weights = None

    def perturbation(self, grids, v0, v1, v2):
        # Initialize perturbation
        z = 4.557 - 3.050j
        # 3.72991 - 1.86362j  # 5.3044557 - 0.00029450j  # 2.0 + 0.77409j  # 1.4986j
        # z_alt = -2.0 + 0.77409j
        # (0.2k) # 0.295j (0.4k) # 0.354j  # 1.304j / 3.0  # 1.768j  # 1.768j  # 0.25j # 0.5j  # 1.768j
        # complex phase velocity
        # ix = cp.ones_like(grids.x.arr_cp)
        # iv = cp.ones_like(grids.v.arr_cp)
        # shifted arrays
        v2_0 = grids.v.arr_cp - v0
        # v2_1 = grids.v.arr_cp - v1
        # v2_2 = grids.v.arr_cp - v2
        factor = 1.0 / (np.sqrt(2.0 * np.pi * self.vt ** 2.0))
        df0 = - factor * v2_0 / self.vt ** 2.0 * cp.exp(- 0.5 * v2_0 ** 2.0 / self.vt ** 2.0)
        # df1 = - factor * v2_1 / self.vt ** 2.0 * cp.exp(- 0.5 * v2_1 ** 2.0 / self.vt ** 2.0)
        # df2 = - factor * v2_2 / self.vt ** 2.0 * cp.exp(- 0.5 * v2_2 ** 2.0 / self.vt ** 2.0)
        df = df0  # + df1 + df2) / 3.0
        psi_0 = cp.divide(df, (grids.v.arr_cp - z))  # * cp.exp(1j * grids.x.k1 * grids.v.arr_cp)
        # psi_a = cp.divide(df, (grids.v.arr_cp - z_alt))
        # psi_1 = cp.divide(df, (grids.v.arr_cp + z))  # * cp.exp(-1j * grids.x.k1 * grids.v.arr_cp)
        # psi_1 = cp.multiply(psi, cp.exp(1j * grids.x.k1 * grids.v.arr_cp))
        # x2 = cp.tensordot(grids.x.arr_cp.flatten(), cp.ones_like(grids.v.arr_cp).flatten(), axes=0)
        # v2 = cp.tensordot(cp.ones_like(grids.x.arr_cp).flatten(), grids.v.arr_cp.flatten(), axes=0)
        # print(x2.shape)
        # dfx = cp.tensordot(cp.ones_like(grids.x.arr_cp).flatten(), 0.5 * (df1 + df2).flatten(), axes=0)
        # cbd = np.linspace(cp.amin(dfx).get(), cp.amax(dfx).get(), num=100)
        #
        # plt.figure()
        # plt.contourf(x2.get(), v2.get(), dfx.reshape(x2.shape[0], v2.shape[1]).get(), cbd)
        # plt.show()

        eig0 = cp.tensordot(cp.exp(1j * grids.x.k1 * grids.x.arr_cp), psi_0, axes=0) / 2.0j
        # eiga = cp.tensordot(cp.exp(1j * grids.x.k1 * grids.x.arr_cp), psi_a, axes=0) / 2.0j
        # eig1 = cp.tensordot(cp.exp(-1j * grids.x.k1 * grids.x.arr_cp), psi_1, axes=0) / 2.0j
        return cp.real(eig0) / 10.0  # + cp.imag(eiga)  #  - eig1) / 10.0

    def initialize(self, grids):
        # As CuPy arrays
        # Indicators
        ix = cp.ones_like(grids.x.arr_cp)
        iv = cp.ones_like(grids.v.arr_cp)
        # Density factor (incl. perturbations)
        den = ix + 0.01 * cp.sin(grids.x.k1 * grids.x.arr_cp)
        factor = 1.0 / (np.sqrt(2.0 * np.pi * self.vt * self.vt)) * cp.tensordot(ix, iv, axes=0)
        factor_p = 1.0 / (np.sqrt(2.0 * np.pi * self.vt * self.vt)) * cp.tensordot(den, iv, axes=0)

        # gaussian
        v0 = 0.0
        v1 = 4.0
        v2 = -4.0
        vsq0 = cp.tensordot(ix, cp.power(grids.v.arr_cp - v0, 2.0), axes=0)
        # vsq1 = cp.tensordot(ix, cp.power(grids.v.arr_cp - v1, 2.0), axes=0)
        # vsq2 = cp.tensordot(ix, cp.power(grids.v.arr_cp - v2, 2.0), axes=0)
        gauss0 = cp.exp(-0.5 * vsq0 / self.vt ** 2.0)
        # gauss1 = cp.exp(-0.5 * vsq1 / self.vt ** 2.0)
        # gauss2 = cp.exp(-0.5 * vsq2 / self.vt ** 2.0)
        gauss = gauss0  # + gauss1 + gauss2) / 3.0

        # Build distribution
        self.arr_no_pert = cp.multiply(factor, gauss)
        # self.arr = cp.multiply(factor_p, gauss)
        self.arr = self.arr_no_pert + self.perturbation(grids, v0, v1, v2)

    # def initialize_no_pert(self, grids):
    #     # As CuPy arrays
    #     # Indicators
    #     ix = cp.ones_like(grids.x.arr_cp)
    #     iv = cp.ones_like(grids.v.arr_cp)
    #     # Density factor
    #     den = ix  # + 0.01 * cp.sin(grids.x.k1 * grids.x.arr_cp)
    #     # den = ix
    #     factor = 1.0 / (self.vt * np.sqrt(2.0 * np.pi)) * cp.tensordot(den, iv, axes=0)
    #     # factor = 1.0 / (self.vt * np.sqrt(2.0 * np.pi)) * cp.tensordot(iv, den, axes=0)
    #
    #     # gaussian
    #     vsq1 = cp.tensordot(ix, cp.power(grids.v.arr_cp + 2.5, 2.0), axes=0)
    #     vsq2 = cp.tensordot(ix, cp.power(grids.v.arr_cp - 2.5, 2.0), axes=0)
    #     gauss1 = 0.5 * cp.exp(-0.5 * vsq1 / self.vt ** 2.0)
    #     gauss2 = 0.5 * cp.exp(-0.5 * vsq2 / self.vt ** 2.0)
    #     gauss = gauss1 + gauss2

        # Build distribution
        # self.arr = cp.multiply(factor, gauss)
        # self.arr_no_pert = cp.multiply(factor, cp.exp(-0.5 * vsq / self.vt ** 2.0))

    def initialize_quad_weights(self, grids):
        """
        Initialize the velocity-space quadrature weights
        """
        self.quad_weights = grids.v.quad_weights / grids.v.J

    def moment_zero(self):
        """
        Compute zeroth moment on gpu
        """
        # Permute pdf array to natural tensor product order
        # self.tensor_product_order_gpu()
        # Compute quadrature as tensor contraction on index pairs, avoiding ghost cells ([1:-1] etc.)
        moment = cp.tensordot(self.arr[1:-1, :, 1:-1, :],
                              self.quad_weights, axes=([2, 3], [0, 1]))  # [0, 1], [0, 1]
        # Permute pdf array back to grid order
        # self.grid_order_gpu()

        # Return zeroth moment
        return moment  # / 1.0e12

    def grid_flatten(self):
        return self.arr[1:-1, :, 1:-1, :].reshape(self.x_res * self.x_ord, self.v_res * self.v_ord).get()

    # def grid_flatten(self):
    #     return self.arr[1:-1, :, 1:-1, :].reshape(self.x_res * self.x_ord, self.v_res * self.v_ord).get()

    def flatten_no_pert(self):
        return self.arr_no_pert[1:-1, :, 1:-1, :].reshape(self.x_res * self.x_ord, self.v_res * self.v_ord).get()
