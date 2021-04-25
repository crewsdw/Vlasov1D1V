import numpy as np
# import datetime

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

        print(self.electron_acceleration_multiplier)
        print(self.ion_acceleration_multiplier)
        print(self.charge_density_multiplier)
