import matplotlib.pyplot as plt
import numpy as np
from select_data import select_data_txt


def model_off(file_name = 'test_data_corr_thrust.txt'):

    K = 0.9                     # Correction factor
    C = 2.07                    # Wind tunnel cross section
    V_wing_str = 0.001765*2     # Front strut
    V_aft      = 0.000196       # Aft strut
    V = 2*V_wing_str + V_aft

    # Correction factor for solid blockage
    e_sb = K*V/(C**(3/2))

    # Import the data points to be corrected
    data = np.genfromtxt(file_name)

    # Get headers
    header_names = open(file_name, 'r').readlines()[0].split()[1:]
    data[0, 0] -= 1

    for i in range(len(data[:, 0])):

        data_point = data[i, :]

        # Correction for angle of attack and sideslip separately
        off_aoa = np.mean(select_data_txt(['AoA'],
                              [data_point[header_names.index('AoA')]],
                              ['CD', 'Cy', 'CL', 'CMpitch', 'CMyaw'], file_name = 'model_off_aoa.txt'), axis = 0)

        off_aos = np.mean(select_data_txt(['AoS'],
                              [data_point[header_names.index('AoS')]],
                              ['CD', 'Cy', 'CL', 'CMpitch', 'CMyaw'], file_name = 'model_off_aos.txt'), axis = 0)

        # Add corrections together
        off = off_aoa + off_aos

        # Correct model off data
        off_corr = off/((1 + e_sb)**2)
        data[i, header_names.index('CD_uncorr')]    -= off_corr[0]
        data[i, header_names.index('CY')]           -= off_corr[1]
        data[i, header_names.index('CL_uncorr')]    -= off_corr[2]
        data[i, header_names.index('CMpitch')]      -= off_corr[3]
        data[i, header_names.index('CMyaw')]        -= off_corr[4]

    np.savetxt('test_data_thrust_model_off_corr.txt', data, header = '        '.join(header_names), fmt='%10.5f')


class Corrections:

    # ===== Model geometry ====
    # Part volumes
    V_wing_strut = 0.001765*2   # Front strut
    V_aft_strut  = 0.000196     # Aft strut
    V_fuselage   = 0.01606      # Fuselage
    V_wing       = 0.00302      # Wing (all of it)
    V_vtail      = 0.000355     # Vertical tail
    V_htail      = 0.000975     # Horizontal tail
    V_nacelle    = 0.000792*2   # Nacelles (both)

    S   = 0.2172   # [m^2] Reference area for the model

    # ==== Factors for solid blockage ====
    tau = 0.88
    K_3_fuse    = 0.9   # Fuselage
    K_3_nacelle = 0.94  # Nacelle
    K_1_htail   = 1.02  # Horizontal tail
    K_1_vtail   = 1.04  # Vertical tail
    K_th_strut  = 0.9/tau
    C           = 2.07  # [m^2] Tunnel cross sectional area

    # Propeller dimensions for slipstream blockage ------------
    D = 0.2032               # m
    Sp = np.pi / 4 * D ** 2  # m^2

    # lift int
    b_wing = 1.397
    wing_mac = 0.165

    def __init__(self, data_for_correction):
        self.data = data_for_correction

    def solid_blockage_element(self, K, part_volume):
        return K*self.tau*part_volume/(self.C**(3/2))

    def solid_blockage(self):

        e_sb_f  = self.solid_blockage_element(self.K_3_fuse, self.V_fuselage)
        e_sb_n  = self.solid_blockage_element(self.K_3_nacelle, self.V_nacelle)
        e_sb_ht = self.solid_blockage_element(self.K_1_htail, self.V_htail)
        e_sb_vt = self.solid_blockage_element(self.K_1_vtail, self.V_vtail)
        e_sb_w  = self.solid_blockage_element(self.K_th_strut, self.V_wing)     # Using Thom's approximation
        e_sb_st = self.solid_blockage_element(self.K_th_strut, self.V_aft_strut+self.V_wing_strut)

        return e_sb_f + e_sb_n + e_sb_ht + e_sb_vt + e_sb_w + e_sb_st

    def zero_lift_drag(self, data_point):

        # For each datapoint, keep everything constant apart from AoA, CL and CD
        data = select_data_txt(['AoS', 'Re', 'J_M1', 'dr'], [data_point[3], data_point[6], data_point[4], data_point[1]],
                               ['AoA', 'CL_uncorr', 'CD_uncorr', 'run'], file_name='test_data_corr_thrust.txt')

        cl   = data[:, 1]
        cd   = data[:, 2]

        # Fit a line through the CL^2 - CD graph to find cd0
        poly_clcd = np.polyfit(cl ** 2, cd, deg=1)
        cd_0 = poly_clcd[1]

        return cd_0

    def wake_blockage(self):

        # Number of data points
        N_pts = len(self.data[:, 0])

        # Array to store correction
        e_wb = np.zeros(N_pts)

        # Loop through all the points and find the zero lift drag for that specific case
        for i in range(N_pts):
            cd_0 = self.zero_lift_drag(self.data[i, :])

            e_wb[i] = self.S * cd_0 / (4 * self.C)

        return e_wb

    def slipstream_blockage(self):

        # Retrieve necessary data
        J1 = self.data[:, 4]  # J is used to retrieve CT from graph, so it is rounded, so only J1 is used
        rho = self.data[:, 12]
        V = self.data[:, 13]
        n1 = self.data[:, 14]
        n2 = self.data[:, 15]
        n = (n1 + n2) / 2  # Average n from both engines to get T

        # Thrust coefficient at each condition (from graph)
        CT_data = np.array([[0.30155, 0.23681, 0.14191, 0.044346],  # V = 30 m/s - J: [1.6,  1.75, 2,     2.25]
                            [0.25455, 0.15610, 0.10554, 0.051441]])  # V = 40 m/s - J: [1.75, 2,    2.125, 2.25]

        # Construct array of thrust coefficient with same shape as data arrays
        CTs = np.zeros_like(V)

        # Round function to round to needed precision for J
        # (Taken from https://stackoverflow.com/questions/2272149/round-to-5-or-other-number-in-python)
        def myround(x, prec=2, base=.05):
            return np.round(base * np.round(x / base), prec)

        # Get correct CT for each data point
        for i in range(len(V)):

            # Get row of CT array depending on V
            if round(V[i]) == 30:
                row = 0

                # Round J to values at which CT was taken (rounding precision depends on V, so leave inside loop
                J = myround(J1[i], prec=2, base=.05)

                # Get CT depending on J
                if J <= 1.6:
                    column = 0
                elif J == 1.75:
                    column = 1
                elif J == 2.:
                    column = 2
                elif J >= 2.25:
                    column = 3

            elif round(V[i]) == 40:
                row = 1

                # Round J to values at which CT was taken (rounding precision depends on V, so leave inside loop
                J = myround(J1[i], prec=3, base=.025)

                # Get CT depending on J
                if J <= 1.75:
                    column = 0
                elif J == 2.:
                    column = 1
                elif J == 2.125:
                    column = 2
                elif J >= 2.25:
                    column = 3

            # Get corresponding CT at each data point
            CTs[i] = CT_data[row][column]

        # Get thrust from thrust coefficient ---------------
        # CT = T/rho/n^2/D^4
        T = CTs * rho * n ** 2 * self.D ** 4  # N - Per propeller!

        # Thrust coefficient for correction (With propeller area as reference area)
        Tc = T / (0.5 * rho * V ** 2 * self.Sp)

        # Correction factor (multiplied by 2 because there are 2 propellers)
        e_ss = 2 * (-Tc / (2 * np.sqrt(1 + 2 * Tc)) * self.Sp / self.C)

        return e_ss

    def CL_W(self, data_point):

        # For each datapoint, keep everything constant apart from AoA, CL and CD
        data = select_data_txt(['AoA', 'AoS', 'Re'],
                               [data_point[2], data_point[3], data_point[6]], ['AoA', 'CL', 'CD', 'run'],
                               file_name='tail_off_data.txt')
        cl = np.mean(data[:, 1])
        # cd = data[:, 2]
        return cl

    def CL_alpha(self, data_point):

        # For each datapoint, keep everything constant apart from AoA, CL and CD
        data = select_data_txt(['AoS', 'Re'],
                               [data_point[3], data_point[6]], ['AoA', 'CL', 'CD', 'run'],
                               file_name='tail_off_data.txt')

        alpha = data[:, 0]
        cl   = data[:, 1]

        alpha = alpha[~np.isnan(cl)]
        cl = cl[~np.isnan(cl)]

        plt.plot(alpha, cl)
        plt.grid()
        # plt.show()

        # Fit a line through the CL^2 - CD graph to find cd0
        poly_cl_alpha = np.polyfit(alpha, cl, deg=1)
        x = np.linspace(-5, 5, 100)
        cl_a = np.gradient(x, poly_cl_alpha[0] * x + poly_cl_alpha[1])

        return np.mean(cl_a)

    def lift_interference_main_wing(self):
        b_v = self.b_wing * 0.78
        be = (self.b_wing + b_v) / 2
        delta = 0.105

        N_pts = len(self.data[:, 0])

        alpha_up = np.zeros(N_pts)

        for i in range(N_pts):
            CL_w = self.CL_W(self.data[i, :])
            alpha_up[i] = delta * self.S / CL_w  # clw
            # alpha_up[i] = self.S * cd_0 / (4 * self.C)

        tau_2 = 0.135
        alpha_sc = 0.5 * self.wing_mac * alpha_up * tau_2

        CD_W = np.zeros(N_pts)
        for i in range(1):
            CL_w = self.CL_W(self.data[i, :])
            CD_W[i] = delta * self.S / self.C * CL_w ** 2

        CM = np.zeros(N_pts)
        for i in range(N_pts):
            CL_a = self.CL_alpha(self.data[i, :])
            CM[i] = 1/8 * alpha_sc[i] * CL_a

        return alpha_up + alpha_sc, CD_W, CM

    def down_wash(self):
        N_pts = len(self.data[:, 0])

        alpha_up = np.zeros(N_pts)

        for i in range(N_pts):
            delta = 0.105
            CL_w = self.CL_W(self.data[i, :])
            alpha_up[i] = delta * self.S / CL_w  # clw

        tau_2_t = 0.75
        alpha_sc = 3.22 * 0.165 * alpha_up * tau_2_t

        cm_a = 0.7777
        CM = cm_a * (alpha_up + alpha_sc)
        return CM


if __name__ == '__main__':
    # Import the data
    unc_data = np.genfromtxt('test_data_corr_thrust.txt')
    corr = Corrections(unc_data)
    corr.lift_interference_main_wing()

    model_off()


    # Test run
    # corr.wake_blockage()
    # corr.solid_blockage()
    # corr.slipstream_blockage()
