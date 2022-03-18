import numpy as np
from select_data import select_data_txt

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
        data = select_data_txt(['AoS', 'Re', 'J_M1', 'dr'], [data_point[3], data_point[6], data_point[4], data_point[1]], ['AoA', 'CL', 'CD', 'run'])

        cl   = data[:, 1]
        cd   = data[:, 2]

        # TODO: correct CL and CD for thrust

        # Fit a line through the CL^2 - CD graph to find cd0
        poly_clcd = np.polyfit(cl**2, cd, deg = 1)
        cd_0 = poly_clcd[1]

        return cd_0

    def wake_blockage(self):

        # Number of data points
        N_pts = len(self.data[:, 0])

        # Array to store correction
        e_wb  = np.zeros(N_pts)

        # Loop through all the points and find the zero lift drag for that specific case
        for i in range(N_pts):

            cd_0 = self.zero_lift_drag(self.data[i, :])

            e_wb[i] = self.S*cd_0/(4*self.C)

        return e_wb


if __name__ == '__main__':

    # Import the data
    unc_data = np.genfromtxt('test_data.txt')
    corr = Corrections(unc_data)

    # Test run
    corr.wake_blockage()
    corr.solid_blockage()

