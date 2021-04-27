import numpy as np
import datetime

# Physical Constants
me = 9.10938188e-31  # electron mass [kg]
mp = 1.67262158e-27  # proton mass [kg]
e = 1.60217646e-19  # elementary charge [C]
eps0 = 8.854187e-12  # [F/m]
mu0 = 4.0e-7 * np.pi  # [H/m]
c = 1.0 / np.sqrt(eps0 * mu0)  # speed of light [m/s]


class Reference:
    def __init__(self, triplet, mass_fraction):
        self.n = triplet[0]  # density [particles/m^3]
        self.T = triplet[1]  # proton temperature [eV]
        self.TeTi = triplet[2]  # temperature ratio [Te/Ti]
        # Length scale (actual triplet[2]):
        self.Ld = np.sqrt(eps0 * self.T * self.TeTi / (self.n * e))  # electron Debye length [m]
        # electron mass fraction (fraction of true mass)
        self.mass_fraction = mass_fraction

        # inferred values
        self.p = e * self.n * self.T  # pressure [Pa]
        self.B = np.sqrt(self.p * mu0)  # magnetic field [T]
        self.vth = np.sqrt(e * self.T / mp)  # reference proton thermal velocity [m/s]
        self.v = self.B / np.sqrt(mu0 * mp * self.n)  # reference Alfven velocity [m/s]
        self.tau = self.Ld / self.v  # ion Debye length transit time

        # dimensionless parameters
        self.omp_p_tau = self.tau * np.sqrt(e * e * self.n / (eps0 * mp))  # proton frequency
        self.omp_e_tau = self.tau * np.sqrt(e * e * self.n / (eps0 * me * self.mass_fraction))  # electron freq.
        self.omc_p_tau = self.tau * e * self.B / mp  # proton magnetic frequency
        self.omc_e_tau = self.tau * e * self.B / (me * self.mass_fraction)  # electron magnetic frequency
        self.dp = c * self.tau / self.omp_p_tau  # skin depth (proton)

        # problem parameters
        self.ze = -1.0  # electron charge ratio
        self.zi = +1.0  # proton charge ratio
        self.ae = self.mass_fraction * me / mp  # electron mass ratio
        self.ai = +1.0  # proton mass ratio

        # thermal properties
        self.Ti = self.T  # ion temperature [eV]
        self.Te = self.TeTi * self.T  # electron temperature [eV]
        self.vt_i = np.sqrt(e * self.Ti / mp) / self.v  # normalized ion therm. vel.
        self.vt_e = np.sqrt(e * self.Te / (me * self.mass_fraction)) / self.v  # electron therm. vel.
        # self.vt_e = 1.0
        self.cs = np.sqrt(e * (self.Te + self.Ti) / mp) / self.v  # sound speed
        self.deb_norm = (self.vt_e * self.v) / (self.omp_e_tau / self.tau)  # debye length normalized
        self.length = self.deb_norm  # problem length scale

        # acceleration parameters
        self.electron_acceleration_multiplier = (self.ze / self.ae) * (self.length / self.dp)
        self.ion_acceleration_multiplier = (self.zi / self.ai) * (self.length / self.dp)
        self.charge_density_multiplier = -self.omp_p_tau ** 2.0 * (self.dp / self.length)

        # print(self.electron_acceleration_multiplier)
        # print(self.ion_acceleration_multiplier)
        # print(self.charge_density_multiplier)

    # Create run file
    def info_file(self, info_name, geometry_info, time_info):
        # Unpack geometry info: space
        x_min = geometry_info[0, 0]
        x_max = geometry_info[0, 1]
        resolution_x = geometry_info[0, 2]
        order_x = geometry_info[0, 3]
        # Unpack geometry info: velocity u
        u_min = geometry_info[1, 0]
        u_max = geometry_info[1, 1]
        resolution_u = geometry_info[1, 2]
        order_u = geometry_info[1, 3]
        # Unpack geometry info: velocity v
        # v_min = geometry_info[2, 0]
        # v_max = geometry_info[2, 1]
        # resolution_v = geometry_info[2, 2]
        # order_v = geometry_info[2, 3]

        # Unpack time info
        final_time = time_info[0]
        write_time = time_info[1]
        courant_number = time_info[2]
        order_t = time_info[3]

        # Create run parameters file
        write_file = open(info_name, 'w')
        write_file.write('######################################')
        write_file.write('\n### V-high-order Run File ############')
        write_file.write('\n######################################')
        write_file.write('\nRun performed ' + datetime.datetime.today().strftime('%Y-%m-%d'))
        write_file.write('\n\nBasis Triplet:')
        write_file.write('\nRef density:                           ' + "{:.2E}".format(self.n) + ' [m^-3]')
        write_file.write('\nRef temperature:                       ' + "{:.2E}".format(self.T) + ' [eV]')
        write_file.write('\nRef length (Debye):                    ' + "{:.2E}".format(self.length) + ' [m]')
        write_file.write('\n\nValues Inferred from Triplet:')
        write_file.write('\nRef velocity:                          ' + "{:.2E}".format(self.v) + ' [m/s]')
        write_file.write('\nRef time:                              ' + "{:.2E}".format(self.tau) + ' [s]')
        write_file.write('\nRef proton plasma frequency:           ' +
                         "{:.2E}".format(self.omp_p_tau / self.tau) + ' [Hz]')
        write_file.write('\nRef electron plasma frequency:         ' +
                         "{:.2E}".format(self.omp_e_tau / self.tau) + ' [Hz]')
        write_file.write('\nRef proton thermal velocity:           ' + "{:.2E}".format(self.v) + ' [m/s]')
        write_file.write('\nRef proton skin depth:                 ' + "{:.2E}".format(self.dp) + ' [m]')
        write_file.write('\n\nNormalized Run Values (Dimensionless):')
        write_file.write('\nRun density                            ' + "{:.2f}".format(self.n / self.n))
        write_file.write('\nRun omptau                             ' + "{:.2f}".format(self.omp_p_tau))
        write_file.write('\nRun proton thermal velocity            ' + "{:.2f}".format(self.vt_i))
        write_file.write('\nRun proton temperature                 ' + "{:.2f}".format(self.Ti / self.T))
        write_file.write('\nRun electron-ion temperature ratio     ' + "{:.2f}".format(self.TeTi))
        write_file.write('\nRun ometau                             ' + "{:.2f}".format(self.omp_e_tau))
        write_file.write('\nRun electron thermal velocity          ' + "{:.2f}".format(self.vt_e))
        write_file.write('\nRun electron temperature               ' + "{:.2f}".format(self.Te / self.T))
        write_file.write('\nRun Debye length                       ' + "{:.2f}".format(self.Ld))
        write_file.write('\nRun skin depth                         ' + "{:.2f}".format(self.dp / self.length))
        write_file.write('\nElectron acceleration coefficient      ' +
                         "{:.2E}".format(self.electron_acceleration_multiplier))
        write_file.write('\nProton acceleration coefficient        ' +
                         "{:.2E}".format(self.ion_acceleration_multiplier))
        write_file.write(
            '\nPoisson equation coefficient           ' + "{:.2E}".format(self.charge_density_multiplier))
        write_file.write('\n\nGrid and time-stepping parameters')
        write_file.write('\nX elements:                    ' + str(resolution_x))
        write_file.write('\nU elements:                    ' + str(resolution_u))
        # write_file.write('\nV elements:                    ' + str(resolution_v))
        write_file.write('\nX domain minimum:              ' + "{:.2E}".format(x_min))
        write_file.write('\nX domain maximum:              ' + "{:.2E}".format(x_max))
        write_file.write('\nElectron u-velocity minimum    ' + "{:.2E}".format(u_min))
        write_file.write('\nElectron u-velocity maximum    ' + "{:.2E}".format(u_max))
        # write_file.write('\nElectron v-velocity minimum    ' + "{:.2E}".format(v_min))
        # write_file.write('\nElectron v-velocity maximum    ' + "{:.2E}".format(v_max))
        write_file.write('\nTime to run:                   ' + "{:.2f}".format(final_time))
        write_file.write('\nTime per data write-out         ' + "{:.2f}".format(write_time))
        write_file.write('\nCFL number:                    ' + "{:.2f}".format(courant_number))
        write_file.write('\nSpatial element order:         ' + str(order_x))
        write_file.write('\nVelocity-u element order:      ' + str(order_u))
        # write_file.write('\nVelocity-v element order:      ' + str(order_v))
        write_file.write('\nTemporal stepping order:       ' + str(order_t))
        write_file.close()
