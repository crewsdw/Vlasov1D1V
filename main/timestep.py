import numpy as np
import cupy as cp
import time as timer
import copy
import matplotlib.pyplot as plt

# SSP-RK coefficients
ssp_rk_switch = {
    1: [1],
    2: [1 / 2, 1 / 2],
    3: [1 / 3, 1 / 2, 1 / 6],
    4: [3 / 8, 1 / 3, 1 / 4, 1 / 24],
    5: [11 / 30, 3 / 8, 1 / 6, 1 / 12, 1 / 120],
    6: [53 / 144, 11 / 30, 3 / 16, 1 / 18, 1 / 48, 1 / 720],
    7: [103 / 280, 53 / 144, 11 / 60, 3 / 48, 1 / 72, 1 / 240, 1 / 5040],
    8: [2119 / 5760, 103 / 280, 53 / 288, 11 / 180, 1 / 64, 1 / 360, 1 / 1440, 1 / 40320]
}

# Courant numbers for RK-DG stability from Cockburn and Shu 2001, [time_order][space_order-1]
courant_numbers = {
    1: [1.0],
    2: [1.0, 0.333],
    3: [1.256, 0.409, 0.209, 0.130, 0.089, 0.066, 0.051, 0.040, 0.033],
    4: [1.392, 0.464, 0.235, 0.145, 0.100, 0.073, 0.056, 0.045, 0.037],
    5: [1.608, 0.534, 0.271, 0.167, 0.115, 0.085, 0.065, 0.052, 0.042],
    6: [1.776, 0.592, 0.300, 0.185, 0.127, 0.093, 0.072, 0.057, 0.047],
    7: [1.977, 0.659, 0.333, 0.206, 0.142, 0.104, 0.080, 0.064, 0.052],
    8: [2.156, 0.718, 0.364, 0.225, 0.154, 0.114, 0.087, 0.070, 0.057]
}

nonlinear_ssp_rk_switch = {
    2: [[1 / 2, 1 / 2, 1 / 2]],
    3: [[3 / 4, 1 / 4, 1 / 4],
        [1 / 3, 2 / 3, 2 / 3]]
}


class Stepper:
    def __init__(self, time_order, space_order, write_time, final_time, linear=False):
        # Time-stepper order and SSP-RK coefficients
        self.time_order = time_order
        self.space_order = space_order
        if linear:
            self.coefficients = self.get_coefficients()
        else:
            self.coefficients = self.get_nonlinear_coefficients()

        # Courant number
        self.courant = self.get_courant_number()
        # print('Courant number is ' + str(self.courant))
        # quit()

        # Simulation time init
        self.time = 0.0  # cp.asarray(0.0)
        self.dt = None
        self.steps_counter = 0
        self.write_counter = 1  # IC already written
        self.density = None
        self.time_array = None
        self.field_energy = None

        # Time between write-outs
        self.write_time = write_time
        # Final time to step to
        self.final_time = final_time

    def get_coefficients(self):
        return np.array([ssp_rk_switch.get(self.time_order, "nothing")][0])

    def get_nonlinear_coefficients(self):
        return np.array(nonlinear_ssp_rk_switch.get(self.time_order, "nothing"))

    def get_courant_number(self):
        return courant_numbers.get(self.time_order)[self.space_order - 1]

    def main_loop(self, distribution, basis, elliptic, grids, dg_flux, refs, save_file):
        print('Courant number is ' + str(self.courant))
        # Loop while time is less than final time
        t0 = timer.time()
        # Initial field energy
        density = cp.tensordot(distribution.arr[1:-1, :, 1:-1, :],
                               grids.v.quad_weights, axes=([2, 3], [0, 1])) / grids.v.J
        charge_initial = cp.mean(density) - density
        # electric_field_initial = elliptic.poisson(charge_density=charge_initial, grid=grids.x, basis=basis.b1)
        electric_field_initial = elliptic.poisson2(charge_density=charge_initial, grid=grids.x)
        #
        self.density = np.array([density.get()])
        self.field_energy = np.array([elliptic.electric_energy(grid=grids.x, field=electric_field_initial).get()])
        self.time_array = np.array([self.time])  # .get()])
        # initial dt set
        max_speeds = [grids.v.arr_max, abs(refs.electron_acceleration_multiplier) *
                      cp.amax(cp.absolute(electric_field_initial[1:-1, :])).get()]
        self.dt = self.adapt_time_step(max_speeds=max_speeds, dx=grids.x.dx, dv=grids.v.dx)
        # Copies of IC
        initial_condition_copy = copy.deepcopy(distribution.arr)
        # initial_condition = distribution.flatten_no_pert()  # distribution.grid_flatten()
        while self.time < self.final_time:
            # Swap ghosts
            distribution.arr[0, :, :, :] = distribution.arr[-2, :, :, :]
            distribution.arr[-1, :, :, :] = distribution.arr[1, :, :, :]
            distribution.arr[:, :, 0, :] = initial_condition_copy[:, :, 0, :]
            distribution.arr[:, :, -1, :] = initial_condition_copy[:, :, -1, :]
            # Do RK update
            distribution_next, electric_field, density = self.nonlinear_ssp_rk(func=distribution, basis=basis,
                                                                               elliptic=elliptic,
                                                                               grids=grids, dg_flux=dg_flux, refs=refs)
            # distribution_next, electric_field, density = self.ssp_rk_update(func=distribution, basis=basis,
            #                                                                 elliptic=elliptic,
            #                                                                 grids=grids, dg_flux=dg_flux, refs=refs)
            # Reset array
            distribution.arr[grids.no_ghost_slice] = distribution_next[grids.no_ghost_slice]
            # Get time and energy
            self.steps_counter += 1
            self.time += self.dt
            self.time_array = np.append(self.time_array, self.time)  # .get())
            energy = elliptic.electric_energy(field=electric_field, grid=grids.x).get()
            density = (cp.tensordot(distribution.arr[1:-1, :, 1:-1, :],
                               grids.v.quad_weights, axes=([2, 3], [0, 1])) / grids.v.J).get()
            self.field_energy = np.append(self.field_energy, energy)
            self.density = np.append(self.density, density)
            # Write-out at specified times
            if self.time > self.write_counter * self.write_time:
                print('\nI made it through step ' + str(self.steps_counter))
                self.write_counter += 1
                print('Saving data...')
                save_file.save_data(distribution=distribution.arr.get(),
                                    elliptic=elliptic,
                                    density=density,
                                    time=self.time,
                                    field_energy=energy)
                print('Done.')
                print('The simulation time is {:0.3e}'.format(self.time))
                print('The time-step is {:0.3e}'.format(self.dt))
                print('Time since start is %.3f' % ((timer.time() - t0) / 60.0) + ' minutes')

        # Loop finished
        print('\nDone...!')

    def nonlinear_ssp_rk(self, func, basis, elliptic, grids, dg_flux, refs):
        # Set up stages
        arr_stage = cp.zeros((self.time_order, func.arr.shape[0], func.arr.shape[1],
                              func.arr.shape[2], func.arr.shape[3]))
        # First stage, moment
        density = cp.tensordot(func.arr[1:-1, :, 1:-1, :],
                               grids.v.quad_weights, axes=([2, 3], [0, 1])) / grids.v.J
        charge = cp.mean(density) - density
        # Electric field
        # field = elliptic.poisson(charge_density=charge, grid=grids.x, basis=basis.b1)
        field = elliptic.poisson2(charge_density=charge, grid=grids.x)
        # Forward euler advance
        df_dt = dg_flux.semi_discrete_rhs(function=func.arr,
                                          field=field, basis=basis, grids=grids)
        # Set first stage
        g_idx = tuple([0] + [idx for idx in grids.no_ghost_slice])
        arr_stage[g_idx] = func.arr[grids.no_ghost_slice] + self.dt * df_dt[grids.no_ghost_slice]
        # Compute further stages
        for i in range(1, self.time_order):
            # Sync ghost-cells
            arr_stage[i - 1, 0, :, :, :] = arr_stage[i - 1, -2, :, :, :]
            arr_stage[i - 1, -1, :, :, :] = arr_stage[i - 1, 1, :, :, :]
            arr_stage[i - 1, :, :, 0, :] = func.arr[:, :, 0, :]
            arr_stage[i - 1, :, :, -1, :] = func.arr[:, :, -1, :]
            # Next stage
            density = cp.tensordot(arr_stage[i - 1, 1:-1, :, 1:-1, :],
                                   grids.v.quad_weights, axes=([2, 3], [0, 1])) / grids.v.J
            charge = cp.mean(density) - density
            # Electric field
            # field = elliptic.poisson(charge_density=charge, grid=grids.x, basis=basis.b1)
            field = elliptic.poisson2(charge_density=charge, grid=grids.x)
            # RK stage advance
            df_dt = dg_flux.semi_discrete_rhs(function=arr_stage[i - 1, :, :, :, :],
                                              field=field, basis=basis, grids=grids)
            g_idx_i = tuple([i] + [idx for idx in grids.no_ghost_slice])
            g_idx_i1 = tuple([i - 1] + [idx for idx in grids.no_ghost_slice])
            arr_stage[g_idx_i] = (self.coefficients[i - 1, 0] * func.arr[grids.no_ghost_slice] +
                                  self.coefficients[i - 1, 1] * arr_stage[g_idx_i1] +
                                  self.coefficients[i - 1, 2] * self.dt * df_dt[grids.no_ghost_slice])
        # Update dt
        max_speeds = [grids.v.arr_max, abs(refs.electron_acceleration_multiplier) *
                      cp.amax(cp.absolute(field[1:-1, :])).get()]
        self.dt = self.adapt_time_step(max_speeds=max_speeds, dx=grids.x.dx, dv=grids.v.dx)
        # Return result
        return arr_stage[-1, :, :, :, :], field, density

    def ssp_rk_update(self, func, basis, elliptic, grids, dg_flux, refs):
        # Set up stage-advance
        arr_step = copy.deepcopy(func.arr)
        arr_next = cp.zeros_like(func.arr)
        arr_next[grids.no_ghost_slice] = self.coefficients[0] * func.arr[grids.no_ghost_slice]
        # Time-step loop
        for i in range(1, self.time_order):
            # Sync ghost-cells (periodic + fixed velocity at IC)
            arr_step[0, :, :, :] = arr_step[-2, :, :, :]
            arr_step[-1, :, :, :] = arr_step[1, :, :, :]
            arr_step[:, :, 0, :] = func.arr[:, :, 0, :]
            arr_step[:, :, -1, :] = func.arr[:, :, -1, :]
            # Moment
            density = cp.tensordot(arr_step[1:-1, :, 1:-1, :],
                                   grids.v.quad_weights, axes=([2, 3], [0, 1])) / grids.v.J
            charge = cp.mean(density) - density
            # Electric field
            field = elliptic.poisson(charge_density=charge, grid=grids.x, basis=basis.b1)
            # Forward euler advance
            df_dt = dg_flux.semi_discrete_rhs(function=arr_step,  # cp.ascontiguousarray(, dtype=cp.float64)
                                              field=field, basis=basis, grids=grids)
            arr_step[grids.no_ghost_slice] += self.dt * df_dt[grids.no_ghost_slice]
            # Accumulate update
            arr_next[grids.no_ghost_slice] += self.coefficients[i] * arr_step[grids.no_ghost_slice]

        # Last stage: sync ghost-cells
        arr_step[0, :, :, :] = arr_step[-2, :, :, :]
        arr_step[-1, :, :, :] = arr_step[1, :, :, :]
        arr_step[:, :, 0, :] = func.arr[:, :, 0, :]
        arr_step[:, :, -1, :] = func.arr[:, :, -1, :]
        # Moment
        density = cp.tensordot(arr_step[1:-1, :, 1:-1, :],
                               grids.v.quad_weights, axes=([2, 3], [0, 1])) / grids.v.J
        charge = cp.mean(density) - density
        # Electric field
        field = elliptic.poisson(charge_density=charge, grid=grids.x, basis=basis.b1)
        # Explicit advance
        df_dt = dg_flux.semi_discrete_rhs(function=arr_step,  # cp.ascontiguousarray(, dtype=cp.float64)
                                          field=field, basis=basis, grids=grids)
        arr_next[grids.no_ghost_slice] += self.coefficients[-1] * self.dt * df_dt[grids.no_ghost_slice]
        # Update dt
        max_speeds = [grids.v.arr_max, abs(refs.electron_acceleration_multiplier) *
                      cp.amax(cp.absolute(field[1:-1, :])).get()]
        self.dt = self.adapt_time_step(max_speeds=max_speeds, dx=grids.x.dx, dv=grids.v.dx)
        return arr_next, field, density

    def adapt_time_step(self, max_speeds, dx, dv):
        # self.courant = 0.045  # 0.02
        # self.courant = 0.5
        # self.courant = 0.2
        return self.courant / (max_speeds[0] / dx + max_speeds[1] / dv)  # / 2.1

# Bin
# Main loop:
# max_speeds = [grids.v.arr_max, abs(refs.electron_acceleration_multiplier) *
#               cp.amax(cp.absolute(electric_field[1:-1, :]))]
# self.dt = self.adapt_time_step(max_speeds=max_speeds, dx=grids.x.dx, dv=grids.v.dx)
# print('\nStep is ' + str(self.steps_counter) + ' and sim time is {:0.3e}'.format(self.time.get()))
# print('Examining ghost layer...')
# # print(distribution.arr[45, :, 0, -1])
# # print(distribution.arr[45, :, 1, 0])
# df_dt = dg_flux.semi_discrete_rhs(function=cp.ascontiguousarray(distribution.arr, dtype=cp.float64),
#                                   field=electric_field, basis=basis, grids=grids)[grids.no_ghost_slice]
# df_dt = df_dt[:, :, -5:, :]
# size = 5
# df_dt_f = df_dt.reshape(grids.x.res * grids.x.order, size * grids.v.order).get()
# print(df_dt[50, :, 0, :])
# print(df_dt[50, :, 1, :])
# print(df_dt.shape)
# # print(df_dt_f[:, 5])
# plt.figure()
# plt.imshow(df_dt_f.T, cmap='gray')
# plt.colorbar()
# plt.show()
# if self.steps_counter == 1990:
#     plt.figure()
#     plt.plot(grids.x.arr.flatten(), electric_field_initial.flatten().get(), 'o--', label='initial step field')
#     plt.plot(grids.x.arr.flatten(), electric_field.flatten().get(), 'o--', label='last step field')
#     plt.legend(loc='best')
#     plt.grid(True)
#     plt.show()
# print(max_speeds)
# self.dt = self.adapt_time_step(max_speeds=max_speeds, dx=grids.x.dx, dv=grids.v.dx)
# print('\nStep is ' + str(self.steps_counter))
# print('Time-step dt is ' + str(self.dt))

# RK update:
# Find maximum speeds and adapt time-step
# Moment
# density = cp.tensordot(distribution.arr[1:-1, :, 1:-1, :],
#                        grids.v.quad_weights, axes=([2, 3], [0, 1])) / grids.v.J  # [0, 1], [0, 1]
# Electric field
# electric_field = elliptic.poisson(charge_density=density - cp.mean(density), grid=grids.x, basis=basis.b1)
# with open('fields.npy', 'rb') as f:
#     fields_o = cp.asarray(np.load(f))
# fields_o[:, 0, :] = fields_o[:, -2, :]
# fields_o[:, -1, :] = fields_o[:, 1, :]

# fields[-1, :, :] = field
# plt.figure()
# plt.plot(grids.x.arr[1:-1, :].flatten(), density_initial.flatten().get(), 'o--', label='initial step density')
# plt.plot(grids.x.arr[1:-1, :].flatten(), density.flatten().get(), 'o--', label='last step density')
# plt.legend(loc='best')
# plt.grid(True)
#
# plt.figure()
# plt.plot(grids.x.arr.flatten(), field_initial.flatten().get(), 'o--', label='initial step field')
# plt.plot(grids.x.arr.flatten(), field.flatten().get(), 'o--', label='last step field')
# plt.legend(loc='best')
# plt.grid(True)
# plt.show()
# Forward euler advance
# df_dt = dg_flux.semi_discrete_rhs(function=cp.ascontiguousarray(arr_step, dtype=cp.float64),
#                                   field=field, basis=basis, grids=grids)
# Load stages
# with open('first_stage_nohadapt.npy', 'rb') as f:
#     stages_other = np.load(f)
# print(stages.shape)
# print(stages_other.shape)
# print('here dt is ' + str(self.dt))
# Flatten stages other
# stages = stages[:, 1:-1, :, 1:-1, :]
# stages_other = stages_other[:, 1:-1, 1:-1, :]
# so_f = np.zeros((stages_other.shape[0], grids.x.res * grids.x.order, grids.v.res * grids.v.order))
# s_f = stages.reshape(stages.shape[0], grids.x.res * grids.x.order, grids.v.res * grids.v.order)
# for i in range(grids.x.res):
#     for j in range(grids.v.res):
#         for k in range(grids.x.order):
#             for ll in range(grids.v.order):
#                 so_f[:, grids.x.order * i + k, grids.v.order * j + ll] = stages_other[:, i, j,
#                                                                          ll*grids.x.order + k]
# plt.figure()
# plt.imshow((so_f[0, :, :] - s_f[0, :, :].get()).T)
# plt.title('Difference of zeros')
# plt.colorbar()
# plt.figure()
# plt.imshow(s_f[0, :, :].get().T)
# plt.colorbar()
# plt.figure()
# plt.imshow(df_dt[1:-1, :, 1:-1, :].reshape(grids.x.res * grids.x.order, grids.v.res * grids.v.order).get().T)
# plt.colorbar()
# plt.title('df_dt')
# for i in range(1, self.time_order):
#     print(np.amax(so_f[i, :, :] - s_f[i, :, :].get()))
#     # plt.figure()
#     # plt.imshow((so_f[i, :, :] - so_f[i-1, :, :]).T)
#     # plt.title('Stage other diff ' + str(i))
#     # plt.colorbar()
#     # plt.figure()
#     # plt.imshow((s_f[i, :, :] - s_f[i-1, :, :]).get().T)
#     # plt.title('Stage here diff ' + str(i))
#     # plt.colorbar()
#     plt.figure()
#     plt.imshow((so_f[i, :, :] - s_f[i, :, :].get()).T)
#     plt.title('Both stage diff ' + str(i))
#     plt.colorbar()
# plt.show()

# df[-1, :, :] = df_dt[grids.no_ghost_slice].reshape((grids.x.res * grids.x.order,
#                                                     grids.v.res * grids.v.order))
# # Load other dfdt
# with open('dfdt.npy', 'rb') as f:
#     dfo = np.load(f)
# with open('fields.npy', 'rb') as f:
#     fields_o = np.load(f)
# for i in range(self.time_order):
#     plt.figure()
#     plt.imshow(dfo[i, :, :].T)
#     plt.colorbar()
#     plt.figure()
#     plt.plot(grids.x.arr.flatten(), fields_o[i, :, :].flatten(), 'o--', label='other field')
#     plt.plot(grids.x.arr.flatten(), fields[i, :, :].flatten().get(), 'o--', label='this field')
#     plt.grid(True)
#     plt.legend(loc='best')
#     plt.figure()
#     plt.plot(grids.x.arr[1:-1, :].flatten(), (fields[i, :, :].get() - fields_o[i, :, :])[1:-1, :].flatten(), 'o--')
#     plt.grid(True)
#     plt.figure()
#     plt.imshow((df[i, :, :].get() - dfo[i, :, :]).T)
#     plt.colorbar()
#     plt.title('dfdt diff ' + str(i))
# plt.show()

# density_initial = cp.tensordot(func.arr[1:-1, :, 1:-1, :],
#                                grids.v.quad_weights, axes=([2, 3], [0, 1])) / grids.v.J
# field_initial = elliptic.poisson(charge_density=density_initial - cp.mean(density_initial),
#                                  grid=grids.x, basis=basis.b1)
# stages = cp.zeros((self.time_order, grids.x.res_ghosts, grids.x.order,
#                    grids.v.res_ghosts, grids.v.order))
# stages[0, :, :, :, :] = copy.deepcopy(func.arr)

# df_dt = dg_flux.semi_discrete_rhs(function=cp.ascontiguousarray(stages[-1, :, :, :, :], dtype=cp.float64),
#                                   field=field, basis=basis, grids=grids)  # s_o[-1, :, :]
# Last step
# arr_next = cp.zeros_like(func.arr)
# arr_next[grids.no_ghost_slice] = cp.average(stages[:, 1:-1, :, 1:-1, :], axis=0, weights=self.coefficients)

# RK Loop:
# arr_step[:, :, 0, :] = ghosts[:, :, 0, :]
# arr_step[:, :, -1, :] = ghosts[:, :, -1, :]
# stages[i - 1, 0, :, :, :] = stages[i - 1, -2, :, :, :]
# stages[i - 1, -1, :, :, :] = stages[i - 1, 1, :, :, :]
# stages[i - 1, :, :, 0, :] = func.arr[:, :, 0, :]
# stages[i - 1, :, :, -1, :] = func.arr[:, :, -1, :]
# density = cp.tensordot(stages[i - 1, 1:-1, :, 1:-1, :],
#                        grids.v.quad_weights, axes=([2, 3], [0, 1])) / grids.v.J  # [0, 1], [0, 1]
# max_speeds = [grids.v.arr_max, abs(refs.electron_acceleration_multiplier) *
#               cp.amax(cp.absolute(field[1:-1, :]))]
# self.dt = self.adapt_time_step(max_speeds=max_speeds, dx=grids.x.dx, dv=grids.v.dx)
# df_dt = dg_flux.semi_discrete_rhs(
#     function=cp.ascontiguousarray(stages[i - 1, :, :, :, :], dtype=cp.float64),
#     field=field, basis=basis, grids=grids)  # s_o[i-1, :, :]
# df[i - 1, :, :] = df_dt[grids.no_ghost_slice].reshape((grids.x.res * grids.x.order,
#                                                        grids.v.res * grids.v.order))
# fields[i-1, :, :] = field
# max_speeds = [grids.v.arr_max, abs(refs.electron_acceleration_multiplier) *
#               cp.amax(cp.absolute(field[1:-1, :]))]
# self.dt = self.adapt_time_step(max_speeds=max_speeds, dx=grids.x.dx, dv=grids.v.dx)
# stages[i, 1:-1, :, 1:-1, :] = stages[i - 1, 1:-1, :, 1:-1, :] + self.dt * df_dt[grids.no_ghost_slice]
# stages[i, :, :, 0, :] = func.arr[:, :, 0, :]  # stages[i - 1, :, :, 0, :]
# stages[i, :, :, -1, :] = func.arr[:, :, -1, :]  # stages[i - 1, :, :, -1, :]
# Check out arr_step
# print('Step is ' + str(i))
# step_f = (arr_step - func.arr)[grids.no_ghost_slice].reshape(grids.x.res * grids.x.order,
#                                                              grids.v.res * grids.v.order).get()
# print(np.amax(step_f))
# plt.figure()
# plt.imshow(step_f.T)
# plt.colorbar()
# plt.show()

# if self.steps_counter == 10000:
#     print('\nAll done!')
#     print('Made it through step ' + str(self.steps_counter))
#     print('The simulation time is {:0.3e}'.format(self.time.get()))
#     print('The time-step is {:0.3e}'.format(self.dt.get()))
#     print('Time since start is ' + str((timer.time() - t0) / 60.0) + ' minutes')
#     # Examine
#     df_dt = dg_flux.semi_discrete_rhs(function=cp.ascontiguousarray(distribution.arr, dtype=cp.float64),
#                                       field=electric_field, basis=basis,
#                                       grids=grids, debug=False)[grids.no_ghost_slice]
#     df_dt_f = df_dt.reshape(grids.x.res * grids.x.order, grids.v.res * grids.v.order).get()
#     cb = np.linspace(np.amin(df_dt_f), np.amax(df_dt_f), num=100)
#     dn_f = distribution_next[grids.no_ghost_slice].reshape(grids.x.res * grids.x.order,
#                                                            grids.v.res * grids.v.order).get()
#     df = dn_f - initial_condition
#     cbn = np.linspace(np.amin(dn_f), np.amax(dn_f), num=100)
#     cbd = np.linspace(np.amin(df), np.amax(df), num=100)
#     print(cbd[0])
#     print(cbd[-1])
#     # Visualize
#     plt.figure()
#     plt.plot(time_array, field_energy, 'o--')
#     plt.grid(True)
#     plt.xlabel(r'Time $t$')
#     plt.ylabel('Field energy')
#
#     plt.figure()
#     plt.contourf(x, v, df_dt_f, cb)
#     plt.title('RHS for step i')
#     plt.figure()
#     plt.imshow(df_dt_f.T)
#     plt.colorbar()
#     plt.title('Pixel-view of df_dt_f')
#     plt.figure()
#     plt.contourf(x, v, dn_f, cbn)
#     plt.title('Function after step i')
#     plt.colorbar()
#     plt.figure()
#     plt.contourf(x, v, df, cbd)
#     plt.title('Difference from IC')
#     plt.colorbar()
#
#     f0f = distribution.grid_flatten()
#     cb = np.linspace(np.amin(f0f), np.amax(f0f), num=100)
#     x, v = np.meshgrid(grids.x.arr[1:-1, :].flatten(), grids.v.arr[1:-1, :].flatten(), indexing='ij')
#     plt.figure()
#     plt.contourf(x, v, f0f, cb)
#     plt.title('Flat dist.')
#     plt.show()