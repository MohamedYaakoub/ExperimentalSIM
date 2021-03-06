import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from select_data import select_data_txt


def model_off(file_name='Data_txt/test_data_corr_thrust.txt'):
    K = 0.9  # Correction factor
    C = 2.07  # Wind tunnel cross section
    V_wing_str = 0.001765 * 2  # Front strut
    V_aft = 0.000196  # Aft strut
    V = 2 * V_wing_str + V_aft

    # Correction factor for solid blockage
    e_sb = K * V / (C ** (3 / 2))

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
                                          ['CD', 'Cy', 'CL', 'CMpitch', 'CMyaw'], file_name='Data_txt/model_off_aoa'
                                                                                            '.txt'), axis=0)

        off_aos = np.mean(select_data_txt(['AoS'],
                                          [data_point[header_names.index('AoS')]],
                                          ['CD', 'Cy', 'CL', 'CMpitch', 'CMyaw'], file_name='Data_txt/model_off_aos'
                                                                                            '.txt'), axis=0)

        # Add corrections together
        off = off_aoa + off_aos

        # Correct model off data
        off_corr = off / ((1 + e_sb) ** 2)
        data[i, header_names.index('CD_uncorr')] -= off_corr[0]
        data[i, header_names.index('CY')] -= off_corr[1]
        data[i, header_names.index('CL_uncorr')] -= off_corr[2]
        data[i, header_names.index('CMpitch')] -= off_corr[3]
        data[i, header_names.index('CMyaw')] -= off_corr[4]

    np.savetxt('test_data_thrust_model_off_corr.txt', data, header='        '.join(header_names), fmt='%10.5f')


class Corrections:
    # ===== Model geometry ====
    # Part volumes
    V_wing_strut = 0.001765 * 2  # Front strut
    V_aft_strut = 0.000196  # Aft strut
    V_fuselage = 0.01606  # Fuselage
    V_wing = 0.00302  # Wing (all of it)
    V_vtail = 0.000355  # Vertical tail
    V_htail = 0.000975  # Horizontal tail
    V_nacelle = 0.000792 * 2  # Nacelles (both)

    S = 0.2172  # [m^2] Reference area for the model

    # ==== Factors for solid blockage ====
    tau = 0.88
    K_3_fuse = 0.9  # Fuselage
    K_3_nacelle = 0.94  # Nacelle
    K_1_htail = 1.02  # Horizontal tail
    K_1_vtail = 1.04  # Vertical tail
    K_th_strut = 0.9 / tau
    C = 2.07  # [m^2] Tunnel cross sectional area

    # Propeller dimensions for slipstream blockage ------------
    D = 0.2032  # m
    Sp = np.pi / 4 * D ** 2  # m^2

    # lift int
    b_wing = 1.397
    wing_mac = 0.165

    def __init__(self, data_for_correction, file_location, tail_on = True):
        self.data = data_for_correction
        self.file_location = file_location
        self.tail = tail_on

    def solid_blockage_element(self, K, part_volume):
        return K * self.tau * part_volume / (self.C ** (3 / 2))

    def solid_blockage(self):

        e_sb_f = self.solid_blockage_element(self.K_3_fuse, self.V_fuselage)
        e_sb_n = self.solid_blockage_element(self.K_3_nacelle, self.V_nacelle)
        e_sb_ht = self.solid_blockage_element(self.K_1_htail, self.V_htail)
        e_sb_vt = self.solid_blockage_element(self.K_1_vtail, self.V_vtail)
        e_sb_w = self.solid_blockage_element(self.K_th_strut, self.V_wing)  # Using Thom's approximation
        e_sb_st = self.solid_blockage_element(self.K_th_strut, self.V_aft_strut + self.V_wing_strut)

        return e_sb_f + e_sb_n + e_sb_ht + e_sb_vt + e_sb_w + e_sb_st

    def zero_lift_drag(self, data_point):

        # For each datapoint, keep everything constant apart from AoA, CL and CD
        if self.tail:
            data = select_data_txt(['AoS', 'Re', 'J_M1', 'dr'],
                                   [data_point[3], data_point[6], data_point[4], data_point[1]],
                                   ['AoA', 'CL_uncorr', 'CD_uncorr', 'run'], file_name=self.file_location)
        else:
            header_names = open(self.file_location, 'r').readlines()[0].split()[1:]

            data = select_data_txt(['AoS', 'Re'],
                                   [data_point[header_names.index('AoS')], data_point[header_names.index('Re')]],
                                   ['AoA', 'CL', 'CD', 'run'], file_name=self.file_location)

        cl = data[:, 1]
        cd = data[:, 2]

        cd = cd[~np.isnan(cl)]
        cl = cl[~np.isnan(cl)]

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
                               file_name='Data_txt/tail_off_data.txt')

        cl = np.mean(data[:, 1])
        # cd = data[:, 2]
        return cl

    def CL_alpha(self, data_point):

        # For each datapoint, keep everything constant apart from AoA, CL and CD
        data = select_data_txt(['AoS', 'Re', 'J_M1', 'dr', 'de'],
                               [data_point[3], data_point[6], data_point[4],
                                data_point[1], data_point[16]], ['AoA', 'CL', 'CD', 'run'],
                               file_name='Data_txt/test_data_thrust_model_off_corr.txt')

        alpha = data[:, 0]
        cl = data[:, 1]

        alpha = alpha[~np.isnan(cl)]
        cl = cl[~np.isnan(cl)]


        # Fit a line through the CL^2 - CD graph to find cd0
        poly_cl_alpha = np.polyfit(alpha, cl, deg=1)[0]
        # x = np.linspace(-5, 5, 100)
        # cl_a = np.gradient(x, poly_cl_alpha[0] * x + poly_cl_alpha[1])

        return poly_cl_alpha

    def CY_b(self, data_point):

        # For each datapoint, keep everything constant apart from AoA, CL and CD
        data = select_data_txt(['AoA', 'Re', 'J_M1', 'dr', 'de'],
                               [data_point[2], data_point[6], data_point[4], data_point[1], data_point[16]],
                               ['AoS', 'CY', 'run'],
                               file_name='Data_txt/test_data_thrust_model_off_corr.txt')

        aos = data[:, 0]
        cy = data[:, 1]

        # aos = aos[~np.isnan(cy)]
        # cl = cy[~np.isnan(cy)]


        # Fit a line through the CL^2 - CD graph to find cd0
        cy_beta = np.polyfit(aos, cy, deg=1)[0]
        # x = np.linspace(-5, 5, 100)
        # cl_a = np.gradient(x, poly_cl_alpha[0] * x + poly_cl_alpha[1])

        return cy_beta

    def CY_vW(self, data_point):

        # For each datapoint, keep everything constant apart from AoA, CL and CD
        data = select_data_txt(['AoA', 'Re', 'J_M1', 'dr', 'de', 'AoS'],
                               [data_point[2], data_point[6], data_point[4],
                                data_point[1], data_point[16], data_point[3]],
                               ['AoS', 'CY', 'run'],
                               file_name='Data_txt/test_data_thrust_model_off_corr.txt')
        CY = np.mean(data[:, 1])
        return CY

    def lift_interference_main_wing(self):
        b_v = self.b_wing * 0.78
        be = (self.b_wing + b_v) / 2
        delta = 0.105

        N_pts = len(self.data[:, 0])

        alpha_up = np.zeros(N_pts)

        for i in range(N_pts):
            CL_w = self.CL_W(self.data[i, :])
            alpha_up[i] = delta * self.S / self.C * CL_w  # clw
            # alpha_up[i] = self.S * cd_0 / (4 * self.C)

        tau_2 = 0.135
        alpha_sc = 0.5 * self.wing_mac * alpha_up * tau_2

        CD_W = np.zeros(N_pts)
        for i in range(N_pts):
            CL_w = self.CL_W(self.data[i, :])
            CD_W[i] = delta * self.S / self.C * CL_w ** 2

        CM = np.zeros(N_pts)
        for i in range(N_pts):
            CL_a = self.CL_alpha(self.data[i, :])
            CM[i] = 1 / 8 * alpha_sc[i] * CL_a

        # ### Directional
        # beta_up = np.zeros(N_pts)
        # delta_beta =
        # for i in range(N_pts):
        #     Cy_b = self.CY_b(self.data[i, :])
        #     beta_up[i] = delta_beta * self.S / self.C * Cy_b
        #     # alpha_up[i] = self.S * cd_0 / (4 * self.C)
        #
        # tau_2 = 0.135
        # beta_sc = 0.5 * self.wing_mac * alpha_up * tau_2
        #
        # CD_W = np.zeros(N_pts)
        # for i in range(1):
        #     CY_v = self.CY_vW(self.data[i, :])
        #     CD_W[i] = delta * self.S / self.C * CL_w ** 2
        #
        # CM = np.zeros(N_pts)
        # for i in range(N_pts):
        #     CL_a = self.CL_alpha(self.data[i, :])
        #     CM[i] = 1 / 8 * alpha_sc[i] * CL_a

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

        cm_a = -0.0842 * 180 / np.pi
        CM = cm_a * (alpha_up + alpha_sc)
        return CM


if __name__ == '__main__':

    # Import the data
    unc_data = np.genfromtxt('Data_txt/tail_off_data_uncorr.txt')
    data_columns_tail_off = open('Data_txt/tail_off_data_uncorr.txt', 'r').readlines()[0].split()[1:]
    data_f_tail_off = pd.DataFrame(unc_data, columns=data_columns_tail_off)
    corr_tail_off = Corrections(unc_data, 'Data_txt/tail_off_data_uncorr.txt', tail_on=False)

    e_tail_off = corr_tail_off.solid_blockage() + corr_tail_off.wake_blockage()

    print((1 + e_tail_off) ** -2)

    # epsilon
    data_f_tail_off['V'] = data_f_tail_off['V'] * (1 + e_tail_off)
    data_f_tail_off['CL'] = data_f_tail_off['CL'] * (1 + e_tail_off) ** -2
    data_f_tail_off['CD'] = data_f_tail_off['CD'] * (1 + e_tail_off) ** -2
    data_f_tail_off['CY'] = data_f_tail_off['CY'] * (1 + e_tail_off) ** -2
    data_f_tail_off['CMpitch'] = data_f_tail_off['CMpitch'] * (1 + e_tail_off) ** -2
    data_f_tail_off['CMyaw'] = data_f_tail_off['CMyaw'] * (1 + e_tail_off) ** -2

    data_f_values_tail_off = data_f_tail_off.values
    np.savetxt('Data_txt/tail_off_data.txt', data_f_values_tail_off,
               header='        '.join(data_columns_tail_off), fmt='%10.5f')

    # Import the data
    unc_data = np.genfromtxt('Data_txt/test_data_thrust_model_off_corr.txt')
    data_columns = open('Data_txt/test_data_thrust_model_off_corr.txt', 'r').readlines()[0].split()[1:]
    data_f = pd.DataFrame(unc_data, columns=data_columns)
    corr = Corrections(unc_data, 'Data_txt/test_data_thrust_model_off_corr.txt')

    e = corr.solid_blockage() + corr.wake_blockage()
    e_slip = corr.slipstream_blockage()
    alpha_l, CD_W_l, CM_l = corr.lift_interference_main_wing()
    CM_d = corr.down_wash()

    # epsilon
    data_f['V'] = data_f['V'] * (1 + e)
    data_f['V'] = data_f['V'] * (1 + e_slip)  # slip stream
    data_f['CL'] = data_f['CL_uncorr'] * (1 + e) ** -2
    data_f['CD'] = data_f['CD_uncorr'] * (1 + e) ** -2
    data_f['CY'] = data_f['CY'] * (1 + e) ** -2
    data_f['CMpitch'] = data_f['CMpitch'] * (1 + e) ** -2
    data_f['CMyaw'] = data_f['CMyaw'] * (1 + e) ** -2

    # lift interference
    data_f['AoA'] = data_f['AoA'] + np.rad2deg(alpha_l)

    # print(np.rad2deg(alpha_l))
    data_f['CD'] = data_f['CD'] + CD_W_l
    print(CD_W_l, alpha_l, CM_l)
    # down wash
    data_f['CMpitch'] = data_f['CMpitch'] + CM_l + CM_d
    data_f_values = data_f.values

    # np.savetxt('Data_txt\Analysis_data.txt', data_columns, fmt='%s')
    np.savetxt('Data_txt\Analysis_data.txt', data_f_values, header='        '.join(data_columns), fmt='%10.5f')

    # Test run
    # corr.wake_blockage()
    # corr.solid_blockage()
    # corr.slipstream_blockage()
