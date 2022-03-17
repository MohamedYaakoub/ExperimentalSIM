import numpy as np


class Corrections:

    # Factors for solid blockage
    tau = 0.88
    K_3_fuse    = 0.9   # Fuselage
    K_3_nacelle = 0.94  # Nacelle
    K_1_htail   = 1.02  # Horizontal tail
    K_1_vtail   = 1.04  # Vertical tail
    K_th_strut  = 0.9/tau
    C           = 2.07  # [m^2] Tunnel cross sectional area

    # Part volumes
    V_wing_strut = 0.001765*2    # Front strut
    V_aft_strut  = 0.000196
    V_fuselage   = 0.016    # Fuselage
    V_wing       = 0.00302  # Wing (all of it)
    V_vtail      = 0.00035  # Vertical tail
    V_htail      = 0.00045  # Horizontal tail
    V_nacelle    = 0.000792*2    # Nacelles (both)

    def solid_blockage(self, K, part_volume):
        return K*self.tau*part_volume/(self.C**(3/2))