import numpy as np
from select_data import select_data_txt
import matplotlib.pyplot as plt


def trim(AoS, AoA, Re, J):
    """
    Finds the lift and drag coefficients in trimmed condition at a certain angle of attack and sideslip.
    This is done using interpolation.
    :param J:
    :param Re:
    :param AoS:
    :param AoA:
    :return:
    """

    # ==== Find stability derivatives ====

    # Find the aerodynamic coefficients at the given conditions
    # this should return data for at least two different rudder angles
    coeffs = select_data_txt(['AoA', 'AoS', 'Re', 'J_M1'], [AoA, AoS, Re, J],
                             ['dr', 'CL_uncorr', 'CD_uncorr', 'CY', 'CMpitch', 'CMyaw'], file_name='test_data_corr_thrust.txt')

    dr = coeffs[:, 0]
    CL = coeffs[:, 1]
    CD = coeffs[:, 2]
    CY = coeffs[:, 3]
    Cm = coeffs[:, 4]
    Cn = coeffs[:, 5]

    # Fit a line through the points for Cn and dr, the slope of this line is the stability derivative Cn_dr
    p = np.polyfit(dr, Cn, deg = 1)
    Cn_dr = p[0]    # Yawing moment derivative wrt rudder deflection
    Cn0   = p[1]    # Yawing moment at zero rudder deflection

    # Finding the rudder angle for zero yawing moment
    dr_eq = -Cn0/Cn_dr

    # Finding the variation of lift and drag with rudder deflection (assuming a linear relationship)
    D = np.polyfit(dr, CD, deg = 1)
    L = np.polyfit(dr, CL, deg = 1)

    print('CD', CD)

    # Lift and drag at equilibrium
    CD_eq = D[0]*dr_eq + D[1]
    CL_eq = L[0]*dr_eq + L[1]
    #
    # plt.plot(dr, Cn, 'x')
    # r = np.linspace(0, -10, 20)
    # plt.plot(r, Cn_dr*r + p[1])
    # plt.show()

    return dr_eq, CD_eq, CL_eq


if __name__ == '__main__':

    beta = np.array([0, 5, 10])
    clcd = np.zeros(len(beta))
    r_eq = np.zeros(len(beta))

    for i in range(len(beta)):
        dreq, cd_eq, cl_eq = trim(beta[i], 5, 339769, 1.75)
        r_eq[i] = dreq
        print(i, cl_eq, cd_eq)
        clcd[i] = cl_eq/cd_eq

    plt.plot(beta, clcd, 'x-')
    plt.show()

    plt.plot(beta, r_eq, 'o-')
    plt.show()


