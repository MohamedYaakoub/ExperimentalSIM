import numpy as np
import matplotlib.pyplot as plt

# Constants
p_ref = 20 * 10**(-6)  # Pa
D = 0.2032             # m
B = 6                  # - (Number of blades)
R_air = 287            # J/kg/K
gamma = 1.4            # -
Temp = 286             # K
L_fs = 27.16           # m (Length of reference full scale aircraft, ATR72)
L_mo = 1.342           # m (Length of model)
SF = L_mo/L_fs

R = np.array([163.9, 151.9, 150.9, 152.6, 157.0, 164.7]) / 1000  # Distance from the microphones to  propeller
Theta = np.deg2rad(np.array([60, 75, 82.5, 90, 97.5, 105.0]))    # Angle from the microphones to  propeller

R_ref = R[0]            # m (Reference distance for SPL correction)

# Read original data from text file
data = np.genfromtxt('Data_txt/test_data.txt')

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


# Read noise data
mic1_data = np.genfromtxt('Data_txt/mic_1.txt')
mic2_data = np.genfromtxt('Data_txt/mic_2.txt')
mic3_data = np.genfromtxt('Data_txt/mic_3.txt')
# mic4_data = np.genfromtxt('Data_txt/mic_4.txt')
# mic5_data = np.genfromtxt('Data_txt/mic_5.txt')
mic6_data = np.genfromtxt('Data_txt/mic_6.txt')

# Save frequency
f = mic1_data[0, :]
# f = f*SF

# Calculate SPL from data -----------------
# Save to arrays
SPL_mic1 = []
SPL_mic2 = []
SPL_mic3 = []
# SPL_mic4 = []
# SPL_mic5 = []
SPL_mic6 = []

# File format is each test point stored as a row, first row of the file has f, and the rest have the SPL per test point
# Loop through all rows only with odd indices (SPL)
for i in range(1, len(mic1_data)):

    SPL_mic1.append(mic1_data[i, :])
    SPL_mic2.append(mic2_data[i, :])
    SPL_mic3.append(mic3_data[i, :])
    # SPL_mic4.append(mic4_data[i, :])
    # SPL_mic5.append(mic5_data[i, :])
    SPL_mic6.append(mic6_data[i, :])


# Convert to numpy arrays
SPL_mic1 = np.array(SPL_mic1)
SPL_mic2 = np.array(SPL_mic2)
SPL_mic3 = np.array(SPL_mic3)
# SPL_mic4 = np.array(SPL_mic4)
# SPL_mic5 = np.array(SPL_mic5)
SPL_mic6 = np.array(SPL_mic6)

# Correct measured SPL ----------------------------

# Get Mach number
M = V / np.sqrt(gamma * R_air * Temp)

# Correction distance
R_prim1 = R[0] / (-M * np.cos(Theta[0]) + np.sqrt(M**2 * np.cos(Theta[0])**2 - M**2 + 1))
R_prim2 = R[1] / (-M * np.cos(Theta[1]) + np.sqrt(M**2 * np.cos(Theta[1])**2 - M**2 + 1))
R_prim3 = R[2] / (-M * np.cos(Theta[2]) + np.sqrt(M**2 * np.cos(Theta[2])**2 - M**2 + 1))
R_prim4 = R[3] / (-M * np.cos(Theta[3]) + np.sqrt(M**2 * np.cos(Theta[3])**2 - M**2 + 1))
R_prim5 = R[4] / (-M * np.cos(Theta[4]) + np.sqrt(M**2 * np.cos(Theta[4])**2 - M**2 + 1))
R_prim6 = R[5] / (-M * np.cos(Theta[5]) + np.sqrt(M**2 * np.cos(Theta[5])**2 - M**2 + 1))

SPL_corr_mic1 = SPL_mic1 + 20 * np.log10(R_prim1[0]/R_ref) + 20 * np.log10(1/SF)
SPL_corr_mic2 = SPL_mic2 + 20 * np.log10(R_prim2[1]/R_ref) + 20 * np.log10(1/SF)
SPL_corr_mic3 = SPL_mic3 + 20 * np.log10(R_prim3[2]/R_ref) + 20 * np.log10(1/SF)
# SPL_corr_mic4 = SPL_mic4 + 20 * np.log10(R_prim4[3]/R_ref) + 20 * np.log10(1/SF)
# SPL_corr_mic5 = SPL_mic5 + 20 * np.log10(R_prim5[4]/R_ref) + 20 * np.log10(1/SF)
SPL_corr_mic6 = SPL_mic6 + 20 * np.log10(R_prim6[5]/R_ref) + 20 * np.log10(1/SF)

# print(SPL_mic1[0][0:5])
# print(SPL_corr_mic1[0][0:5])

# Calculate p_rms from corrected SPL
p_rms_mic1 = p_ref * 10 ** (SPL_corr_mic1/20)
p_rms_mic2 = p_ref * 10 ** (SPL_corr_mic2/20)
p_rms_mic3 = p_ref * 10 ** (SPL_corr_mic3/20)
# p_rms_mic4 = p_ref * 10 ** (SPL_corr_mic4/20)
# p_rms_mic5 = p_ref * 10 ** (SPL_corr_mic5/20)
p_rms_mic6 = p_ref * 10 ** (SPL_corr_mic6/20)


# Calculate thrust -----------------------------------

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

# Get thrust from thrust coefficient
# CT = T/rho/n^2/D^4
T = CTs * rho * n**2 * D**4  # N

# Reformat T array in order to be able to scale p_rms
T = np.expand_dims(T, axis=0)
T = np.repeat(T.T, repeats=np.shape(p_rms_mic1)[1], axis=1)

# Non-dimensional pressure (Pi theorem) ---------------
Pi_p_rms_mic1 = p_rms_mic1 * D**2 / T
Pi_p_rms_mic2 = p_rms_mic2 * D**2 / T
Pi_p_rms_mic3 = p_rms_mic3 * D**2 / T
# Pi_p_rms_mic4 = p_rms_mic4 * D**2 / T
# Pi_p_rms_mic5 = p_rms_mic5 * D**2 / T
Pi_p_rms_mic6 = p_rms_mic6 * D**2 / T

# Calculate SPL from non-dimensional Pi ratio
SPL_pi_1 = 20 * np.log10(Pi_p_rms_mic1)
SPL_pi_2 = 20 * np.log10(Pi_p_rms_mic2)
SPL_pi_3 = 20 * np.log10(Pi_p_rms_mic3)
# SPL_pi_4 = 20 * np.log10(Pi_p_rms_mic4)
# SPL_pi_5 = 20 * np.log10(Pi_p_rms_mic5)
SPL_pi_6 = 20 * np.log10(Pi_p_rms_mic6)


# Scale frequency ------------------------------------------------------------------------
# Reformat n array in order to be able to scale frequency with blade passage frequency
n = np.expand_dims(n, axis=0)
n = np.repeat(n.T, repeats=np.shape(p_rms_mic1)[1], axis=1)

# Blade passage frequency
BPF = n * B

# Scale frequency with blade passage frequency
f_scaled = f/BPF
# f_scaled = f/36


# Plotting --------------------------------------------
index = 0

# SPL from pi ratio
# SPL vs J
plt.subplot(2, 2, 1)
# plt.axvline(1, color='k', linestyle=(0, (5, 5)), linewidth=0.7, label='BPF')
# plt.axvline(2, color='k', linestyle=(0, (5, 5)), linewidth=0.7)
# plt.axvline(3, color='k', linestyle=(0, (5, 5)), linewidth=0.7)
plt.plot(f_scaled[0], SPL_pi_1[0], label='J = 1.6', linewidth=0.6, color='tab:blue')
plt.plot(f_scaled[1], SPL_pi_1[1], label='J = 1.75', linewidth=0.6, color='tab:green')
plt.plot(f_scaled[3], SPL_pi_1[3], label='J = 2', linewidth=0.6, color='tab:pink')
plt.plot(f_scaled[4], SPL_pi_1[4], label='J = 2.25', linewidth=0.6, color='tab:red')
plt.axvline(1, color='k', linestyle=(0, (5, 5)), linewidth=0.7, label='BPF')
plt.axvline(2, color='k', linestyle=(0, (5, 5)), linewidth=0.7)
plt.axvline(3, color='k', linestyle=(0, (5, 5)), linewidth=0.7)
# plt.plot(f, Pi_p_rms_mic1[0])

plt.title('Microphone 1')
plt.xlim((0, 4))
# plt.ylim((0, 10))
plt.xlabel('f/BPF [-]')
plt.ylabel('SPL [dB]')
# plt.grid()
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(f_scaled[0], SPL_pi_2[0], label='J = 1.6', linewidth=0.6, color='tab:blue')
plt.plot(f_scaled[1], SPL_pi_2[1], label='J = 1.75', linewidth=0.6, color='tab:green')
plt.plot(f_scaled[3], SPL_pi_2[3], label='J = 2', linewidth=0.6, color='tab:pink')
plt.plot(f_scaled[4], SPL_pi_2[4], label='J = 2.25', linewidth=0.6, color='tab:red')
plt.axvline(1, color='k', linestyle=(0, (5, 5)), linewidth=0.7, label='BPF')
plt.axvline(2, color='k', linestyle=(0, (5, 5)), linewidth=0.7)
plt.axvline(3, color='k', linestyle=(0, (5, 5)), linewidth=0.7)

plt.title('Microphone 2')
plt.xlim((0, 4))
plt.xlabel('f/BPF [-]')
plt.ylabel('SPL [dB]')
# plt.grid()
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(f_scaled[0], SPL_pi_3[0], label='J = 1.6', linewidth=0.6, color='tab:blue')
plt.plot(f_scaled[1], SPL_pi_3[1], label='J = 1.75', linewidth=0.6, color='tab:green')
plt.plot(f_scaled[3], SPL_pi_3[3], label='J = 2', linewidth=0.6, color='tab:pink')
plt.plot(f_scaled[4], SPL_pi_3[4], label='J = 2.25', linewidth=0.6, color='tab:red')
plt.axvline(1, color='k', linestyle=(0, (5, 5)), linewidth=0.7, label='BPF')
plt.axvline(2, color='k', linestyle=(0, (5, 5)), linewidth=0.7)
plt.axvline(3, color='k', linestyle=(0, (5, 5)), linewidth=0.7)

plt.title('Microphone 3')
plt.xlim((0, 4))
plt.xlabel('f/BPF [-]')
plt.ylabel('SPL [dB]')
# plt.grid()
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(f_scaled[0], SPL_pi_6[0], label='J = 1.6', linewidth=0.6, color='tab:blue')
plt.plot(f_scaled[1], SPL_pi_6[1], label='J = 1.75', linewidth=0.6, color='tab:green')
plt.plot(f_scaled[3], SPL_pi_6[3], label='J = 2', linewidth=0.6, color='tab:pink')
plt.plot(f_scaled[4], SPL_pi_6[4], label='J = 2.25', linewidth=0.6, color='tab:red')
plt.axvline(1, color='k', linestyle=(0, (5, 5)), linewidth=0.7, label='BPF')
plt.axvline(2, color='k', linestyle=(0, (5, 5)), linewidth=0.7)
plt.axvline(3, color='k', linestyle=(0, (5, 5)), linewidth=0.7)

plt.title('Microphone 6')
plt.xlim((0, 4))
plt.xlabel('f/BPF [-]')
plt.ylabel('SPL [dB]')
# plt.grid()
plt.legend()

plt.show()


# SPL vs AoA -----------------------------------------------------------------------------
plt.subplot(2, 2, 1)
# plt.axvline(1, color='k', linestyle=(0, (5, 5)), linewidth=0.7, label='BPF')
# plt.axvline(2, color='k', linestyle=(0, (5, 5)), linewidth=0.7)
# plt.axvline(3, color='k', linestyle=(0, (5, 5)), linewidth=0.7)
plt.plot(f_scaled[0], SPL_pi_1[0], label=r'$\alpha$ = 0 deg', linewidth=0.6, color='tab:blue')
plt.plot(f_scaled[5], SPL_pi_1[5], label=r'$\alpha$ = 5 deg', linewidth=0.6, color='tab:red')
plt.axvline(1, color='k', linestyle=(0, (5, 5)), linewidth=0.7, label='BPF')
plt.axvline(2, color='k', linestyle=(0, (5, 5)), linewidth=0.7)
plt.axvline(3, color='k', linestyle=(0, (5, 5)), linewidth=0.7)
# plt.plot(f, Pi_p_rms_mic1[0])

plt.title('Microphone 1')
plt.xlim((0, 4))
# plt.ylim((0, 10))
plt.xlabel('f/BPF [-]')
plt.ylabel('SPL [dB]')
# plt.grid()
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(f_scaled[0], SPL_pi_2[0], label=r'$\alpha$ = 0 deg', linewidth=0.6, color='tab:blue')
plt.plot(f_scaled[5], SPL_pi_2[5], label=r'$\alpha$ = 5 deg', linewidth=0.6, color='tab:red')
plt.axvline(1, color='k', linestyle=(0, (5, 5)), linewidth=0.7, label='BPF')
plt.axvline(2, color='k', linestyle=(0, (5, 5)), linewidth=0.7)
plt.axvline(3, color='k', linestyle=(0, (5, 5)), linewidth=0.7)

plt.title('Microphone 2')
plt.xlim((0, 4))
plt.xlabel('f/BPF [-]')
plt.ylabel('SPL [dB]')
# plt.grid()
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(f_scaled[0], SPL_pi_3[0], label=r'$\alpha$ = 0 deg', linewidth=0.6, color='tab:blue')
plt.plot(f_scaled[5], SPL_pi_3[5], label=r'$\alpha$ = 5 deg', linewidth=0.6, color='tab:red')
plt.axvline(1, color='k', linestyle=(0, (5, 5)), linewidth=0.7, label='BPF')
plt.axvline(2, color='k', linestyle=(0, (5, 5)), linewidth=0.7)
plt.axvline(3, color='k', linestyle=(0, (5, 5)), linewidth=0.7)

plt.title('Microphone 3')
plt.xlim((0, 4))
plt.xlabel('f/BPF [-]')
plt.ylabel('SPL [dB]')
# plt.grid()
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(f_scaled[0], SPL_pi_6[0], label=r'$\alpha$ = 0 deg', linewidth=0.6, color='tab:blue')
plt.plot(f_scaled[5], SPL_pi_6[5], label=r'$\alpha$ = 5 deg', linewidth=0.6, color='tab:red')
plt.axvline(1, color='k', linestyle=(0, (5, 5)), linewidth=0.7, label='BPF')
plt.axvline(2, color='k', linestyle=(0, (5, 5)), linewidth=0.7)
plt.axvline(3, color='k', linestyle=(0, (5, 5)), linewidth=0.7)

plt.title('Microphone 6')
plt.xlim((0, 4))
plt.xlabel('f/BPF [-]')
plt.ylabel('SPL [dB]')
# plt.grid()
plt.legend()

plt.show()

# SPL vs AoS
plt.subplot(2, 2, 1)

plt.plot(f_scaled[49], SPL_pi_1[49], label=r'$\beta$ = -5 deg', linewidth=0.6, color='tab:blue')
plt.plot(f_scaled[57], SPL_pi_1[57], label=r'$\beta$ = 0 deg', linewidth=0.6, color='tab:green')
plt.plot(f_scaled[65], SPL_pi_1[65], label=r'$\beta$ = 5 deg', linewidth=0.6, color='tab:pink')
plt.plot(f_scaled[73], SPL_pi_1[73], label=r'$\beta$ = 10 deg', linewidth=0.6, color='tab:red')
plt.axvline(1, color='k', linestyle=(0, (5, 5)), linewidth=0.7, label='BPF')
plt.axvline(2, color='k', linestyle=(0, (5, 5)), linewidth=0.7)
plt.axvline(3, color='k', linestyle=(0, (5, 5)), linewidth=0.7)
# plt.plot(f, Pi_p_rms_mic1[0])

plt.title('Microphone 1')
plt.xlim((0, 4))
# plt.ylim((0, 10))
plt.xlabel('f/BPF [-]')
plt.ylabel('SPL [dB]')
# plt.grid()
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(f_scaled[49], SPL_pi_2[49], label=r'$\beta$ = -5 deg', linewidth=0.6, color='tab:blue')
plt.plot(f_scaled[57], SPL_pi_2[57], label=r'$\beta$ = 0 deg', linewidth=0.6, color='tab:green')
plt.plot(f_scaled[65], SPL_pi_2[65], label=r'$\beta$ = 5 deg', linewidth=0.6, color='tab:pink')
plt.plot(f_scaled[73], SPL_pi_2[73], label=r'$\beta$ = 10 deg', linewidth=0.6, color='tab:red')
plt.axvline(1, color='k', linestyle=(0, (5, 5)), linewidth=0.7, label='BPF')
plt.axvline(2, color='k', linestyle=(0, (5, 5)), linewidth=0.7)
plt.axvline(3, color='k', linestyle=(0, (5, 5)), linewidth=0.7)

plt.title('Microphone 2')
plt.xlim((0, 4))
plt.xlabel('f/BPF [-]')
plt.ylabel('SPL [dB]')
# plt.grid()
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(f_scaled[49], SPL_pi_3[49], label=r'$\beta$ = -5 deg', linewidth=0.6, color='tab:blue')
plt.plot(f_scaled[57], SPL_pi_3[57], label=r'$\beta$ = 0 deg', linewidth=0.6, color='tab:green')
plt.plot(f_scaled[65], SPL_pi_3[65], label=r'$\beta$ = 5 deg', linewidth=0.6, color='tab:pink')
plt.plot(f_scaled[73], SPL_pi_3[73], label=r'$\beta$ = 10 deg', linewidth=0.6, color='tab:red')
plt.axvline(1, color='k', linestyle=(0, (5, 5)), linewidth=0.7, label='BPF')
plt.axvline(2, color='k', linestyle=(0, (5, 5)), linewidth=0.7)
plt.axvline(3, color='k', linestyle=(0, (5, 5)), linewidth=0.7)

plt.title('Microphone 3')
plt.xlim((0, 4))
plt.xlabel('f/BPF [-]')
plt.ylabel('SPL [dB]')
# plt.grid()
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(f_scaled[49], SPL_pi_6[49], label=r'$\beta$ = -5 deg', linewidth=0.6, color='tab:blue')
plt.plot(f_scaled[57], SPL_pi_6[57], label=r'$\beta$ = 0 deg', linewidth=0.6, color='tab:green')
plt.plot(f_scaled[65], SPL_pi_6[65], label=r'$\beta$ = 5 deg', linewidth=0.6, color='tab:pink')
plt.plot(f_scaled[73], SPL_pi_6[73], label=r'$\beta$ = 10 deg', linewidth=0.6, color='tab:red')
plt.axvline(1, color='k', linestyle=(0, (5, 5)), linewidth=0.7, label='BPF')
plt.axvline(2, color='k', linestyle=(0, (5, 5)), linewidth=0.7)
plt.axvline(3, color='k', linestyle=(0, (5, 5)), linewidth=0.7)

plt.title('Microphone 6')
plt.xlim((0, 4))
plt.xlabel('f/BPF [-]')
plt.ylabel('SPL [dB]')
# plt.grid()
plt.legend()

plt.show()

# SPL vs V -----------------------------------------------------------------------------
plt.subplot(2, 2, 1)
# plt.axvline(1, color='k', linestyle=(0, (5, 5)), linewidth=0.7, label='BPF')
# plt.axvline(2, color='k', linestyle=(0, (5, 5)), linewidth=0.7)
# plt.axvline(3, color='k', linestyle=(0, (5, 5)), linewidth=0.7)
plt.plot(f_scaled[3], SPL_pi_1[3], label='V = 30 m/s', linewidth=0.6, color='tab:blue')
plt.plot(f_scaled[26], SPL_pi_1[26], label='V = 40 m/s', linewidth=0.6, color='tab:red')
plt.axvline(1, color='k', linestyle=(0, (5, 5)), linewidth=0.7, label='BPF')
plt.axvline(2, color='k', linestyle=(0, (5, 5)), linewidth=0.7)
plt.axvline(3, color='k', linestyle=(0, (5, 5)), linewidth=0.7)
# plt.plot(f, Pi_p_rms_mic1[0])

plt.title('Microphone 1')
plt.xlim((0, 4))
# plt.ylim((0, 10))
plt.xlabel('f/BPF [-]')
plt.ylabel('SPL [dB]')
# plt.grid()
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(f_scaled[3], SPL_pi_2[3], label='V = 30 m/s', linewidth=0.6, color='tab:blue')
plt.plot(f_scaled[26], SPL_pi_2[26], label='V = 40 m/s', linewidth=0.6, color='tab:red')
plt.axvline(1, color='k', linestyle=(0, (5, 5)), linewidth=0.7, label='BPF')
plt.axvline(2, color='k', linestyle=(0, (5, 5)), linewidth=0.7)
plt.axvline(3, color='k', linestyle=(0, (5, 5)), linewidth=0.7)

plt.title('Microphone 2')
plt.xlim((0, 4))
plt.xlabel('f/BPF [-]')
plt.ylabel('SPL [dB]')
# plt.grid()
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(f_scaled[3], SPL_pi_3[3], label='V = 30 m/s', linewidth=0.6, color='tab:blue')
plt.plot(f_scaled[26], SPL_pi_3[26], label='V = 40 m/s', linewidth=0.6, color='tab:red')
plt.axvline(1, color='k', linestyle=(0, (5, 5)), linewidth=0.7, label='BPF')
plt.axvline(2, color='k', linestyle=(0, (5, 5)), linewidth=0.7)
plt.axvline(3, color='k', linestyle=(0, (5, 5)), linewidth=0.7)

plt.title('Microphone 3')
plt.xlim((0, 4))
plt.xlabel('f/BPF [-]')
plt.ylabel('SPL [dB]')
# plt.grid()
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(f_scaled[3], SPL_pi_6[3], label='V = 30 m/s', linewidth=0.6, color='tab:blue')
plt.plot(f_scaled[26], SPL_pi_6[26], label='V = 40 m/s', linewidth=0.6, color='tab:red')
plt.axvline(1, color='k', linestyle=(0, (5, 5)), linewidth=0.7, label='BPF')
plt.axvline(2, color='k', linestyle=(0, (5, 5)), linewidth=0.7)
plt.axvline(3, color='k', linestyle=(0, (5, 5)), linewidth=0.7)

plt.title('Microphone 6')
plt.xlim((0, 4))
plt.xlabel('f/BPF [-]')
plt.ylabel('SPL [dB]')
# plt.grid()
plt.legend()

plt.show()


# SPL vs d_r -----------------------------------------------------------------------------
plt.subplot(2, 2, 1)

plt.plot(f_scaled[0], SPL_pi_1[0], label=r'$\delta_r$ = 0 deg', linewidth=0.6, color='tab:blue')
plt.plot(f_scaled[57], SPL_pi_1[57], label=r'$\delta_r$ = -10 deg', linewidth=0.6, color='tab:red')
plt.axvline(1, color='k', linestyle=(0, (5, 5)), linewidth=0.7, label='BPF')
plt.axvline(2, color='k', linestyle=(0, (5, 5)), linewidth=0.7)
plt.axvline(3, color='k', linestyle=(0, (5, 5)), linewidth=0.7)
# plt.plot(f, Pi_p_rms_mic1[0])

plt.title('Microphone 1')
plt.xlim((0, 4))
# plt.ylim((0, 10))
plt.xlabel('f/BPF [-]')
plt.ylabel('SPL [dB]')
# plt.grid()
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(f_scaled[0], SPL_pi_2[0], label=r'$\delta_r$ = 0 deg', linewidth=0.6, color='tab:blue')
plt.plot(f_scaled[57], SPL_pi_2[57], label=r'$\delta_r$ = -10 deg', linewidth=0.6, color='tab:red')
plt.axvline(1, color='k', linestyle=(0, (5, 5)), linewidth=0.7, label='BPF')
plt.axvline(2, color='k', linestyle=(0, (5, 5)), linewidth=0.7)
plt.axvline(3, color='k', linestyle=(0, (5, 5)), linewidth=0.7)

plt.title('Microphone 2')
plt.xlim((0, 4))
plt.xlabel('f/BPF [-]')
plt.ylabel('SPL [dB]')
# plt.grid()
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(f_scaled[0], SPL_pi_3[0], label=r'$\delta_r$ = 0 deg', linewidth=0.6, color='tab:blue')
plt.plot(f_scaled[57], SPL_pi_3[57], label=r'$\delta_r$ = -10 deg', linewidth=0.6, color='tab:red')
plt.axvline(1, color='k', linestyle=(0, (5, 5)), linewidth=0.7, label='BPF')
plt.axvline(2, color='k', linestyle=(0, (5, 5)), linewidth=0.7)
plt.axvline(3, color='k', linestyle=(0, (5, 5)), linewidth=0.7)

plt.title('Microphone 3')
plt.xlim((0, 4))
plt.xlabel('f/BPF [-]')
plt.ylabel('SPL [dB]')
# plt.grid()
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(f_scaled[0], SPL_pi_6[0], label=r'$\delta_r$ = 0 deg', linewidth=0.6, color='tab:blue')
plt.plot(f_scaled[57], SPL_pi_6[57], label=r'$\delta_r$ = -10 deg', linewidth=0.6, color='tab:red')
plt.axvline(1, color='k', linestyle=(0, (5, 5)), linewidth=0.7, label='BPF')
plt.axvline(2, color='k', linestyle=(0, (5, 5)), linewidth=0.7)
plt.axvline(3, color='k', linestyle=(0, (5, 5)), linewidth=0.7)

plt.title('Microphone 6')
plt.xlim((0, 4))
plt.xlabel('f/BPF [-]')
plt.ylabel('SPL [dB]')
# plt.grid()
plt.legend()

plt.show()

# SPL at original vs remeasured point -----------------------------------------------------------------------------
plt.subplot(2, 2, 1)

plt.plot(f_scaled[9], SPL_pi_1[9], label='Original', linewidth=0.6, color='tab:blue')
plt.plot(f_scaled[113], SPL_pi_1[113], label='Repeated', linewidth=0.6, color='tab:red')
plt.axvline(1, color='k', linestyle=(0, (5, 5)), linewidth=0.7, label='BPF')
plt.axvline(2, color='k', linestyle=(0, (5, 5)), linewidth=0.7)
plt.axvline(3, color='k', linestyle=(0, (5, 5)), linewidth=0.7)
# plt.plot(f, Pi_p_rms_mic1[0])

plt.title('Microphone 1')
plt.xlim((0, 4))
# plt.ylim((0, 10))
plt.xlabel('f/BPF [-]')
plt.ylabel('SPL [dB]')
# plt.grid()
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(f_scaled[9], SPL_pi_2[9], label='Original', linewidth=0.6, color='tab:blue')
plt.plot(f_scaled[113], SPL_pi_2[113], label='Repeated', linewidth=0.6, color='tab:red')
plt.axvline(1, color='k', linestyle=(0, (5, 5)), linewidth=0.7, label='BPF')
plt.axvline(2, color='k', linestyle=(0, (5, 5)), linewidth=0.7)
plt.axvline(3, color='k', linestyle=(0, (5, 5)), linewidth=0.7)

plt.title('Microphone 2')
plt.xlim((0, 4))
plt.xlabel('f/BPF [-]')
plt.ylabel('SPL [dB]')
# plt.grid()
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(f_scaled[9], SPL_pi_3[9], label='Original', linewidth=0.6, color='tab:blue')
plt.plot(f_scaled[113], SPL_pi_3[113], label='Repeated', linewidth=0.6, color='tab:red')
plt.axvline(1, color='k', linestyle=(0, (5, 5)), linewidth=0.7, label='BPF')
plt.axvline(2, color='k', linestyle=(0, (5, 5)), linewidth=0.7)
plt.axvline(3, color='k', linestyle=(0, (5, 5)), linewidth=0.7)

plt.title('Microphone 3')
plt.xlim((0, 4))
plt.xlabel('f/BPF [-]')
plt.ylabel('SPL [dB]')
# plt.grid()
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(f_scaled[9], SPL_pi_6[9], label='Original', linewidth=0.6, color='tab:blue')
plt.plot(f_scaled[113], SPL_pi_6[113], label='Repeated', linewidth=0.6, color='tab:red')
plt.axvline(1, color='k', linestyle=(0, (5, 5)), linewidth=0.7, label='BPF')
plt.axvline(2, color='k', linestyle=(0, (5, 5)), linewidth=0.7)
plt.axvline(3, color='k', linestyle=(0, (5, 5)), linewidth=0.7)

plt.title('Microphone 6')
plt.xlim((0, 4))
plt.xlabel('f/BPF [-]')
plt.ylabel('SPL [dB]')
# plt.grid()
plt.legend()

plt.show()

