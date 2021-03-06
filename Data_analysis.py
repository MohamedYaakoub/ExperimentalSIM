import numpy as np
import pandas as pd
from trim_conditions import trim
import matplotlib.pyplot as plt
from select_data import select_data_txt

plt.rcParams.update({'font.size': 18})



def Direction_stability():
    data = np.genfromtxt('Data_txt/Analysis_data.txt', skip_header=1)
    data_columns = open('Data_txt/Analysis_data.txt', 'r').readlines()[0].split()[1:]
    data_f = pd.DataFrame(data, columns=data_columns)

    def crr_vs_uncrr(filename, Re, J, AoA):
        fig, ax = plt.subplots()
        fig1, ax1 = plt.subplots()

        dat = select_data_txt(['Re', 'J_M1', 'AoA', 'de', 'dr'], [Re, J, AoA, 0, 0], ['AoS', 'CY', 'CMyaw'],
                              file_name=filename)

        # Sort the data to avoid artifacts in the plots
        dat = dat[dat[:, 0].argsort()]
        aos = dat[:, 0]

        # Array to store stability derivatives
        cy = dat[:, 1]
        cn = dat[:, 2]

        return cy, cn, aos

    cy_b_unc, cn_b_unc, aos_unc = crr_vs_uncrr('Data_txt/test_data_thrust_model_off_corr.txt', 340000, 1.75, 0)
    cy_b_cor, cn_b_cor, aos_cor = crr_vs_uncrr('Data_txt/Analysis_data.txt', 340000, 1.75, 0.2)

    fig, ax = plt.subplots()
    fig1, ax1 = plt.subplots()

    ax.plot(aos_unc, cy_b_unc, marker='d', label='Uncorrected', markerfacecolor='none')
    ax.plot(aos_cor, cy_b_cor, marker='s', label='Corrected', markerfacecolor='none')
    ax.set_xlim([-0.5, 10.5])
    ax.set_ylim([0, 0.25])
    ax.set_xlabel('Angle of Sideslip [deg]')
    ax.set_ylabel('$C_{Y}$ [-]')
    ax.grid()
    ax.legend()
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    fig.tight_layout()
    fig.savefig('Figures/Directional_stability/cy_b_comp.pdf')

    ax1.plot(aos_unc, cn_b_unc, marker='d', label='Uncorrected', markerfacecolor='none')
    ax1.plot(aos_cor, cn_b_cor, marker='s', label='Corrected', markerfacecolor='none')
    ax1.set_xlim([-0.5, 10.5])
    ax1.set_ylim([-0.06, 0])
    ax1.set_xlabel('Angle of Sideslip [deg]')
    ax1.set_ylabel('$C_{n}$ [-]')
    ax1.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax1.grid()
    ax1.legend()

    fig1.tight_layout()
    fig1.savefig('Figures/Directional_stability/cn_b_comp.pdf')

    fig.show()
    fig1.show()

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

            # fig.show()
            # fig1.show()


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

    cn_dr_unc, cy_dr_unc, aos_unc = control_derivatives('Data_txt/test_data_thrust_model_off_corr.txt', 450000, 1.99, 5)
    cn_dr_cor, cy_dr_cor, aos_cor = control_derivatives('Data_txt/Analysis_data.txt', 450000, 1.99, 5.4)

    ax.plot(aos_unc, cy_dr_unc, marker= 'd', label='Uncorrected', markerfacecolor='none')
    ax.plot(aos_cor, cy_dr_cor, marker= 's', label='Corrected', markerfacecolor='none')
    ax.set_xlim([-0.5, 10.5])
    ax.set_ylim([0.0024, 0.0035])
    ax.set_xlabel('Angle of Sideslip [deg]')
    ax.set_ylabel('$C_{Y_{\delta r}}$ [-]')

    ax.grid()
    ax.legend()
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    fig.tight_layout()
    fig.savefig('Figures/Rudder_effectiveness/cy_dr_comp.pdf')

    ax1.plot(aos_unc, cn_dr_unc, marker= 'd', label='Uncorrected', markerfacecolor='none')
    ax1.plot(aos_cor, cn_dr_cor, marker= 's', label='Corrected', markerfacecolor='none')
    ax1.set_xlim([-0.5, 10.5])
    ax1.set_ylim([-0.0016, -0.0012])
    ax1.set_xlabel('Angle of Sideslip [deg]')
    ax1.set_ylabel('$C_{n_{\delta r}}$ [-]')
    ax1.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax1.grid()
    ax1.legend()
    fig1.tight_layout()
    fig1.savefig('Figures/Rudder_effectiveness/cn_dr_comp.pdf')

    fig.show()
    fig1.show()

    # Get the levels for each of the parameters of interest
    aoa_levels  = np.unique(np.round(data[:, 2], 1))
    re_levels   = np.unique(np.round(data[:, 6]/1e5, 1))*1e5

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
                    p_dr  = np.polyfit(dr, cn, deg = 1)
                    cn_dr[k] = p_dr[0]

                    p_y   = np.polyfit(dr, cy, deg = 1)
                    cy_dr[k] = p_y[0]

                ax.plot(aos, cy_dr, marker = markers[i], label = 'J =' + str(j), markerfacecolor='none')
                ax1.plot(aos, cn_dr, marker = markers[i], label = 'J =' + str(j), markerfacecolor='none')

            ax.set_xlim([-0.5, 10.5])
            ax.set_ylim([0.0026, 0.004])
            ax.set_xlabel('Angle of Sideslip [deg]')
            ax.set_ylabel('$C_{Y_{\delta r}}$ [-]')
            ax.grid()
            ax.legend()
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            # ax.yaxis.set_major_formatter(formatter)

            fig.tight_layout()
            fig.savefig('Figures/Rudder_effectiveness/cy_dr_re' + str(re) + '_aoa_' + str(aoa) + '.pdf')

            ax1.set_xlim([-0.5, 10.5])
            ax1.set_ylim([-0.0017, -0.0010])
            ax1.set_xlabel('Angle of Sideslip [deg]')
            ax1.set_ylabel('$C_{n_{\delta r}}$ [-]')
            ax1.grid()
            ax1.legend()
            ax1.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            # ax1.yaxis.set_major_formatter(formatter)
            fig1.tight_layout()
            fig1.savefig('Figures/Rudder_effectiveness/cn_dr_re' + str(re) + '_aoa_' + str(aoa) + '.pdf')
            fig.show()


def Trimming():
    data = np.genfromtxt('Data_txt/Analysis_data.txt', skip_header=1)
    re_levels = np.unique(np.round(data[:, 6] / 1e5, 1)) * 1e5  # Re  = [340000. 450000.]
    aoa_levels = np.unique(np.round(data[:, 2], 1))  # AoA = [0.2 5.5]
    aos_levels = np.unique(np.round(data[:, 3], 1))  # AoS = [-5.  0.  5. 10.]
    aos = aos_levels[aos_levels >= 0]  # Check if >= 0 then keep it

    for i in range(len(aoa_levels)):  # Loop the AoA

        for j in range(len(re_levels)):  # Loop the Reynolds

            markers = ['x', 'o', 'D', '|']

            fig, ax = plt.subplots()
            fig1, ax1 = plt.subplots()

            j_m1 = select_data_txt(['Re'], [re_levels[j]], ['J_M1'],
                                   file_name='Data_txt/Analysis_data.txt')  # select j based on Re
            j_levels = np.unique(np.round(j_m1, 1))

            for k in range(len(j_levels)):  # Loop the j for 4 different setup according to the Reynolds

                dr_trim   = np.zeros(len(aos))  # Store the dr in each sideslip angle run
                cdcl_trim = np.zeros(len(aos))  # Store the cd in each sideslip angle run
                for m in range(len(aos)):  # Loop the AoS

                    # find trimmed rudder deflection of certain beta
                    dr = trim(aos[m], aoa_levels[i], re_levels[j],
                              j_levels[k], filename = 'Data_txt/Analysis_data.txt')
                    dr_trim[m] = dr

                    # find the cd of certain beta
                    result = select_data_txt(['AoA', 'Re', 'J_M1', 'AoS', 'de', 'dr'],
                                         [aoa_levels[i], re_levels[j], j_levels[k], aos[m], 0, 0], ['CD', 'CL'],
                                         file_name='Data_txt/Analysis_data.txt')

                    cdcl_trim[m] = np.mean(result[:, 1]/result[:, 0])

                ax.plot(aos, dr_trim, marker=markers[k], label='J =' + str(j_levels[k]),
                        markerfacecolor='none')
                ax1.plot(aos, cdcl_trim, marker=markers[k], label='J =' + str(j_levels[k]),
                        markerfacecolor='none')

            ax.set_xlim([-0.5, 10.5])
            ax.set_ylim([-6, -35])
            ax.set_xlabel('Angle of Sideslip [deg]')
            ax.set_ylabel('$\delta r$ [-]')
            ax.grid()
            ax.legend()
            fig.tight_layout()
            fig.savefig('Figures/Trimming/dr' + str(re_levels[j]) + '_aoa_' + str(aoa_levels[i]) + '.pdf')

            ax1.set_xlim([-0.5, 10.5])
            ax1.set_ylim([0, 20])
            ax1.set_xlabel('Angle of Sideslip [deg]')
            ax1.set_ylabel('$C_{L}/C_{D}$ [-]')
            ax1.grid()
            ax1.legend()
            fig1.tight_layout()
            fig1.savefig('Figures/Trimming/clcd' + str(re_levels[j]) + '_aoa_' + str(aoa_levels[i]) + '.pdf')
            fig.show()
            fig1.show()

    # Corrected vs Uncorrected data
    CDCL_uncorr = np.zeros(len(aos))
    CDCL_corr = np.zeros(len(aos))
    dr_trim_uncorr = np.zeros(len(aos))
    dr_trim_corr = np.zeros(len(aos))
    for a in range(len(aos)):

        uncoor_data = select_data_txt(['AoA', 'Re', 'J_M1', 'AoS', 'de', 'dr'],
                                [5, 450000, 2, aos[a], 0, 0], ['CD_uncorr', 'CL_uncorr'],
                                file_name='Data_txt/test_data_thrust_model_off_corr.txt')

        coor_data = select_data_txt(['AoA', 'Re', 'J_M1', 'AoS', 'de', 'dr'],
                                [5.5, 450000, 2, aos[a], 0, 0], ['CD', 'CL'],
                                file_name='Data_txt/Analysis_data.txt')

        dr_uncorr = trim(aos[a], 5, 450000, 2, filename='Data_txt/test_data_thrust_model_off_corr.txt')
        dr_corr = trim(aos[a], 5.5, 450000, 2, filename='Data_txt/Analysis_data.txt')

        CDCL_uncorr[a] = np.mean(uncoor_data[:, 1]/uncoor_data[:, 0])
        CDCL_corr[a] = np.mean(coor_data[:, 1]/coor_data[:, 0])
        dr_trim_uncorr[a] = dr_uncorr
        dr_trim_corr[a] = dr_corr

    fig2, ax2 = plt.subplots()
    fig3, ax3 = plt.subplots()
    ax2.plot(aos, CDCL_uncorr, marker='d', label='Uncorrected', markerfacecolor='none')
    ax2.plot(aos, CDCL_corr, marker='s', label='Corrected', markerfacecolor='none')
    ax3.plot(aos, dr_trim_uncorr, marker='d', label='Uncorrected', markerfacecolor='none')
    ax3.plot(aos, dr_trim_corr, marker='s', label='Corrected', markerfacecolor='none')

    ax2.set_xlim([-0.5, 10.5])
    ax2.set_ylim([0, 20])
    ax2.set_xlabel('Angle of Sideslip [deg]')
    ax2.set_ylabel('$C_{L}/C_{D}$ [-]')
    ax2.grid()
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig('Figures/Trimming/clcd_aos_comp.pdf')

    ax3.set_xlim([-0.5, 10.5])
    ax3.set_ylim([-6, -35])
    ax3.set_xlabel('Angle of Sideslip [deg]')
    ax3.set_ylabel('$\delta r$ [-]')
    ax3.grid()
    ax3.legend()
    fig3.tight_layout()
    fig3.savefig('Figures/Trimming/dr_aos_comp.pdf')

    fig2.show()
    fig3.show()


if __name__ == '__main__':
    Trimming()
    Rudder_effectiveness()
    Direction_stability()