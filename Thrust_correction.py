import numpy as np

# File format:
# run, dr, AoA, AoS, J_M1, J_M2, Re, CL, CD, CY, CMpitch, CMyaw, rho, V, rpsM1, rpsM2

# Read original data from text file
data = np.genfromtxt('balance_data.txt')

# Retrieve lift and drag coefficients and other necessary data (uncorrected for thrust)
AoA = np.deg2rad(data[:, 2])
J1 = data[:, 4]
J2 = data[:, 5]
CL = data[:, 7]
CD = data[:, 8]
rho = data[:, 12]
V = data[:, 13]
n1 = data[:, 14]
n2 = data[:, 15]
n = (n1 + n2)/2  # Average n from both engines to get T

# Thrust coefficient at each condition (from graph)
CT_data = np.array([[0.30155, 0.23681, 0.14191, 0.044346],   # V = 30 m/s - J: [1.6,  1.75, 2,     2.25]
                    [0.25455, 0.15610, 0.10554, 0.051441]])  # V = 40 m/s - J: [1.75, 2,    2.125, 2.25]

# Construct array of thrust coefficient with same shape as data arrays
CTs = np.zeros_like(V)


# Round function to round to needed precision for J
# (Taken from https://stackoverflow.com/questions/2272149/round-to-5-or-other-number-in-python)
def myround(x, prec=2, base=.05):
    return np.round(base * np.round(x/base), prec)


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
D = 0.2032   # m
T = 2 * CTs * rho * n**2 * D**4  # N - (Multiply by 2 because of 2 propellers)

# Convert to thrust coefficient that can be added/subtracted from aerodynamic coefficients
S = 0.2172                          # m^2 - Model reference area (wing surface)
Tc = T / (1/2 * rho * V**2 * S)     # [-]


# Correct lift and drag coefficients for thrust (uncorrected are actual CL and CD after correcting for thrust, but
# without boundary corrections)
CD_uncorr = CD + Tc * np.cos(AoA)
CL_uncorr = CL - Tc * np.sin(AoA)

# Reshape arrays to add new columns to data
CL_uncorr = np.expand_dims(CL_uncorr, axis=0)
CD_uncorr = np.expand_dims(CD_uncorr, axis=0)

# Append columns of corrected coefficients to array
data = np.hstack((data, CL_uncorr.T))
data = np.hstack((data, CD_uncorr.T))

# Save to text file
variables = ['run', 'dr', 'AoA', 'AoS', 'J_M1', 'J_M2', 'Re', 'CL', 'CD', 'CY', 'CMpitch', 'CMyaw', 'rho', 'V', 'rpsM1',
             'rpsM2', 'CL_uncorr', 'CD_uncorr']
np.savetxt('test_data_corr_thrust.txt', data, header='        '.join(variables), fmt='%10.5f')
