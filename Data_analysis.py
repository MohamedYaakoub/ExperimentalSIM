import numpy as np
import pandas as pd
from trim_conditions import trim
import matplotlib.pyplot as plt
from select_data import select_data_txt

# plt.rcParams.update({'font.size': 16})


def directional_derivatives(filename, Re, J, AoA):
    dat = select_data_txt(['Re', 'J_M1', 'AoA', 'de', 'dr'], [Re, J, AoA, 0, 0], ['AoS', 'CY', 'CMyaw'],
                          file_name=filename)

    dat = dat[dat[:, 0].argsort()]
    aos = dat[:, 0]
    cy = dat[:, 1]
    cn = dat[:, 2]

    print(aos)
    p_beta = np.polyfit(aos, cn, deg=1)
    cn_b = p_beta[0]

    p_y_beta = np.polyfit(aos, cy, deg=1)
    cy_b = p_y_beta[0]

    # cn_b = np.zeros(len(aos))
    # cy_b = np.zeros(len(aos))

    # for k, AoS in enumerate(aos):
    #     dat = select_data_txt(['Re', 'J_M1', 'AoA', 'de', 'dr'], [Re, J, AoA, 0, 0],
    #                           ['CY', 'CMyaw', 'AoS', 'run'],
    #                           file_name=filename)
    #
    #     cy = dat[:, 0]
    #     cn = dat[:, 1]
    #     sideslip = dat[:, 2]
    #     print(sideslip, cy, cn)
    #
    #     # Find cn dr and cy dr
    #     p_beta = np.polyfit(sideslip, cn, deg=1)
    #     cn_b[k] = p_beta[0]
    #
    #     p_y_beta = np.polyfit(sideslip, cy, deg=1)
    #     cy_b[k] = p_y_beta[0]

    return cn_b, cy_b, aos


def Direction_stability():
    data = np.genfromtxt('Data_txt/Analysis_data.txt', skip_header=1)
    data_columns = open('Data_txt/Analysis_data.txt', 'r').readlines()[0].split()[1:]
    data_f = pd.DataFrame(data, columns=data_columns)

    fig, ax = plt.subplots()
    fig1, ax1 = plt.subplots()

    # cn_b_unc, cy_b_unc, aos_unc = directional_derivatives('Data_txt/test_data_thrust_model_off_corr.txt', 339000, 1.75,
    #                                                       0)
    # cn_b_cor, cy_b_cor, aos_cor = directional_derivatives('Data_txt/Analysis_data.txt', 339000, 1.75, 0.2)

    aoa_levels = np.unique(np.round(data[:, 2], 1))
    re_levels = np.unique(np.round(data[:, 6] / 1e5, 1)) * 1e5

    # Plotting conventions
    markers = ['x', 'o', 'D', '|']

    # Loop through the different Reynolds numbers
    for aoa in aoa_levels:

        for re in re_levels:

            # Preparing different subplots, so everything can be plotted while only looping once
            fig, ax = plt.subplots()
            fig1, ax1 = plt.subplots()

            # Find the different J-levels
            temp = select_data_txt(['Re'], [re], ['J_M1'], file_name='Data_txt/Analysis_data.txt')
            j_levels = np.unique(np.round(temp, 1))
            for i, j in enumerate(j_levels):
                dat = select_data_txt(['Re', 'J_M1', 'AoA', 'de', 'dr'], [re, j, aoa, 0, 0], ['AoS', 'CY', 'CMyaw'],
                                      file_name='Data_txt/Analysis_data.txt')

                # Sort the data to avoid artifacts in the plots
                dat = dat[dat[:, 0].argsort()]
                aos = dat[:, 0]

                # Array to store stability derivatives
                cy = dat[:, 1]
                cn = dat[:, 2]

                ax.plot(aos, cy, marker=markers[i], label='J =' + str(j), markerfacecolor='none')
                ax1.plot(aos, cn, marker=markers[i], label='J =' + str(j), markerfacecolor='none')

            ax.set_xlim([-0.5, 10.5])
            ax.set_ylim([0, 0.25])
            ax.set_xlabel('Angle of Sideslip [deg]')
            ax.set_ylabel('$C_{Y}$ [-]')
            ax.grid()
            ax.legend()
            ax1.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

            fig.tight_layout()
            fig.savefig('Figures/Directional_stability/cy_beta' + str(re) + '_aoa_' + str(aoa) + '.pdf')

            ax1.set_xlim([-0.5, 10.5])
            ax1.set_ylim([-0.06, 0])
            ax1.set_xlabel('Angle of Sideslip [deg]')
            ax1.set_ylabel('$C_{n}$ [-]')
            ax1.grid()
            ax1.legend()
            ax1.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            fig1.tight_layout()
            fig.savefig('Figures/Directional_stability/cn_beta' + str(re) + '_aoa_' + str(aoa) + '.pdf')

            fig.show()
            fig1.show()


def control_derivatives(filename, Re, J, AoA):
    dat = select_data_txt(['Re', 'J_M1', 'AoA', 'de'], [Re, J, AoA, 0], ['AoS', 'CY', 'CMyaw', 'dr'],
                          file_name=filename)

    # Remove negative angles of sideslip, they give errors because they don't have rudder deflection
    dat = dat[dat[:, 0] >= 0]

    # Sort the data to avoid artifacts in the plots
    dat = dat[dat[:, 0].argsort()]
    aos = dat[:, 0]

    # Find control derivatives at each sideslip angle

    # Array to store stability derivatives
    cn_dr = np.zeros(len(aos))
    cy_dr = np.zeros(len(aos))

    for k, AoS in enumerate(aos):
        dat = select_data_txt(['Re', 'J_M1', 'AoA', 'de', 'AoS'], [Re, J, AoA, 0, AoS],
                              ['CY', 'CMyaw', 'dr', 'run'],
                              file_name=filename)

        cy = dat[:, 0]
        cn = dat[:, 1]
        dr = dat[:, 2]

        # Find cn dr and cy dr
        p_dr = np.polyfit(dr, cn, deg=1)
        cn_dr[k] = p_dr[0]

        p_y = np.polyfit(dr, cy, deg=1)
        cy_dr[k] = p_y[0]

    return cn_dr, cy_dr, aos


def Rudder_effectiveness():
    # Import the data
    data = np.genfromtxt('Data_txt/Analysis_data.txt', skip_header=1)

    fig, ax = plt.subplots()
    fig1, ax1 = plt.subplots()

    cn_dr_unc, cy_dr_unc, aos_unc = control_derivatives('Data_txt/test_data_thrust_model_off_corr.txt', 339000, 1.75, 0)
    cn_dr_cor, cy_dr_cor, aos_cor = control_derivatives('Data_txt/Analysis_data.txt', 339000, 1.75, 0.2)

    ax.plot(aos_unc, cy_dr_unc, marker='d', label='Uncorrected', markerfacecolor='none')
    ax.plot(aos_cor, cy_dr_cor, marker='s', label='Corrected', markerfacecolor='none')
    ax.set_xlim([-0.5, 10.5])
    ax.set_ylim([0.0034, 0.0045])
    ax.set_xlabel('Angle of Sideslip [deg]')
    ax.set_ylabel('$C_{Y_dr}$ [-]')

    ax.grid()
    ax.legend()
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    fig.tight_layout()
    fig.savefig('Figures/Rudder_effectiveness/cy_dr_comp.pdf')

    ax1.plot(aos_unc, cn_dr_unc, marker='d', label='Uncorrected', markerfacecolor='none')
    ax1.plot(aos_cor, cn_dr_cor, marker='s', label='Corrected', markerfacecolor='none')
    ax1.set_xlim([-0.5, 10.5])
    ax1.set_ylim([-0.002, -0.0014])
    ax1.set_xlabel('Angle of Sideslip [deg]')
    ax1.set_ylabel('$C_{n_dr}$ [-]')
    ax1.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax1.grid()
    ax1.legend()

    fig1.tight_layout()
    fig1.savefig('Figures/Rudder_effectiveness/cn_dr_comp.pdf')

    fig.show()
    fig1.show()

    # Get the levels for each of the parameters of interest
    aoa_levels = np.unique(np.round(data[:, 2], 1))
    re_levels = np.unique(np.round(data[:, 6] / 1e5, 1)) * 1e5

    # Plotting conventions
    markers = ['x', 'o', 'D', '|']

    # Loop through the different Reynolds numbers
    for aoa in aoa_levels:

        for re in re_levels:

            # Preparing different subplots, so everything can be plotted while only looping once
            fig, ax = plt.subplots()
            fig1, ax1 = plt.subplots()

            # Find the different J-levels
            temp = select_data_txt(['Re'], [re], ['J_M1'], file_name='Data_txt/Analysis_data.txt')
            j_levels = np.unique(np.round(temp, 1))

            for i, j in enumerate(j_levels):

                dat = select_data_txt(['Re', 'J_M1', 'AoA', 'de'], [re, j, aoa, 0], ['AoS', 'CY', 'CMyaw', 'dr'],
                                      file_name='Data_txt/Analysis_data.txt')

                # Remove negative angles of sideslip, they give errors because they don't have rudder deflection
                dat = dat[dat[:, 0] >= 0]

                # Sort the data to avoid artifacts in the plots
                dat = dat[dat[:, 0].argsort()]
                aos = dat[:, 0]

                # Find control derivatives at each sideslip angle

                # Array to store stability derivatives
                cn_dr = np.zeros(len(aos))
                cy_dr = np.zeros(len(aos))

                for k, AoS in enumerate(aos):
                    dat = select_data_txt(['Re', 'J_M1', 'AoA', 'de', 'AoS'], [re, j, aoa, 0, AoS],
                                          ['CY', 'CMyaw', 'dr', 'run'],
                                          file_name='Data_txt/Analysis_data.txt')

                    cy = dat[:, 0]
                    cn = dat[:, 1]
                    dr = dat[:, 2]

                    # Find cn dr and cy dr
                    p_dr = np.polyfit(dr, cn, deg=1)
                    cn_dr[k] = p_dr[0]

                    p_y = np.polyfit(dr, cy, deg=1)
                    cy_dr[k] = p_y[0]

                ax.plot(aos, cy_dr, marker=markers[i], label='J =' + str(j), markerfacecolor='none')
                ax1.plot(aos, cn_dr, marker=markers[i], label='J =' + str(j), markerfacecolor='none')

            ax.set_xlim([-0.5, 10.5])
            ax.set_ylim([0.0026, 0.004])
            ax.set_xlabel('Angle of Sideslip [deg]')
            ax.set_ylabel('$C_{Y_dr}$ [-]')
            ax.grid()
            ax.legend()
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            # ax.yaxis.set_major_formatter(formatter)

            fig.tight_layout()
            fig.savefig('Figures/Rudder_effectiveness/cy_dr_re' + str(re) + '_aoa_' + str(aoa) + '.pdf')

            ax1.set_xlim([-0.5, 10.5])
            ax1.set_ylim([-0.0017, -0.0010])
            ax1.set_xlabel('Angle of Sideslip [deg]')
            ax1.set_ylabel('$C_{n_dr}$ [-]')
            ax1.grid()
            ax1.legend()
            ax1.ticklabel_format(axis="y",style="sci", scilimits=(0,0))
            # ax1.yaxis.set_major_formatter(formatter)
            fig1.tight_layout()
            fig1.savefig('Figures/Rudder_effectiveness/cn_dr_re' + str(re) + '_aoa_' + str(aoa) + '.pdf')
            fig.show()



def Trimming():
    data = np.genfromtxt('Data_txt/Analysis_data.txt', skip_header=1)
    re_levels = np.unique(np.round(data[:, 6] / 1e5, 1)) * 1e5  # Re  = [340000. 450000.]
    aoa_levels = np.unique(np.round(data[:, 2], 1))  # AoA = [0. 5.]
    aos_levels = np.unique(np.round(data[:, 3], 1))  # AoS = [-5.  0.  5. 10.]
    aos = aos_levels[aos_levels >= 0]  # Check if >= 0 then keep it

    for i in range(len(aoa_levels)):  # Loop the AoA

        for j in range(len(re_levels)):  # Loop the Reynolds

            markers = ['x', 'o', 'D', '|']

            fig, ax = plt.subplots()

            j_m1 = select_data_txt(['Re'], [re_levels[j]], ['J_M1'],
                                   file_name='Data_txt/Analysis_data.txt')  # select j based on Re
            j_levels = np.unique(np.round(j_m1, 1))

            for k in range(len(j_levels)):  # Loop the j for 4 different setup according to the Reynolds

                dr_trim = np.zeros(len(aos))  # Store the dr in each sideslip angle run
                for m in range(len(aos)):  # Loop the AoS
                    dr = trim(aos[m], aoa_levels[i], re_levels[j],
                              j_levels[k])  # find trimmed rudder deflection of certain beta
                    dr_trim[m] = dr
                ax.plot(aos, dr_trim, marker=markers[k], label='J =' + str(j_levels[k]),
                        markerfacecolor='none')  # plot the

            ax.set_xlim([-0.5, 10.5])
            ax.set_ylim([-6, -35])
            ax.set_xlabel('Angle of Sideslip [deg]')
            ax.set_ylabel('$\delta r$ [-]')
            ax.grid()
            ax.legend()
            fig.tight_layout()
            fig.savefig('Figures/dr' + str(re_levels[j]) + '_aoa_' + str(aoa_levels[i]) + '.pdf')

            plt.show()


if __name__ == '__main__':
    # Rudder_effectiveness()
    Direction_stability()