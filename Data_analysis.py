import numpy as np
import pandas as pd
from select_data import select_data_txt
from trim_conditions import trim
import matplotlib.pyplot as plt
from select_data import select_data_txt

plt.rcParams.update({'font.size': 16})



def Direction_stability():
    data = np.genfromtxt('Data_txt/Analysis_data.txt', skip_header=1)
    data_columns = open('Data_txt/Analysis_data.txt', 'r').readlines()[0].split()[1:]
    data_f = pd.DataFrame(data, columns=data_columns)
    #print(np.unique(data_f['AoA']))


def Rudder_effectiveness():

    # Import the data
    data = np.genfromtxt('Data_txt/Analysis_data.txt', skip_header=1)

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
            ax.set_ylabel('$C_{Y_dr}$ [-]')
            ax.grid()
            ax.legend()
            fig.tight_layout()
            fig.savefig('Figures/cy_dr_re' + str(re) + '_aoa_' + str(aoa) + '.pdf')

            ax1.set_xlim([-0.5, 10.5])
            ax1.set_ylim([-0.0017, -0.0010])
            ax1.set_xlabel('Angle of Sideslip [deg]')
            ax1.set_ylabel('$C_{n_dr}$ [-]')
            ax1.grid()
            ax1.legend()
            fig1.tight_layout()
            fig.savefig('Figures/cn_dr_re' + str(re) + '_aoa_' + str(aoa) + '.pdf')

            plt.show()



def Trimming():
    data = np.genfromtxt('Data_txt/Analysis_data.txt', skip_header=1)
    re_levels  = np.unique(np.round(data[:, 6] / 1e5, 1)) * 1e5          # Re  = [340000. 450000.]
    aoa_levels = np.unique(np.round(data[:, 2], 1))                      # AoA = [0. 5.]
    aos_levels = np.unique(np.round(data[:, 3], 1))                      # AoS = [-5.  0.  5. 10.]
    aos        = aos_levels[aos_levels >= 0]                             # Check if >= 0 then keep it

    for i in range(len(aoa_levels)):                        # Loop the AoA

        for j in range(len(re_levels)):                     # Loop the Reynolds

            markers = ['x', 'o', 'D', '|']

            fig, ax = plt.subplots()

            j_m1 = select_data_txt(['Re'], [re_levels[j]], ['J_M1'], file_name='Data_txt/Analysis_data.txt')    # select j based on Re
            j_levels = np.unique(np.round(j_m1, 1))

            for k in range(len(j_levels)):                  # Loop the j for 4 different setup according to the Reynolds

                dr_trim = np.zeros(len(aos))                # Store the dr in each sideslip angle run
                for m in range(len(aos)):                   # Loop the AoS
                    dr = trim(aos[m], aoa_levels[i], re_levels[j], j_levels[k])     # find trimmed rudder deflection of certain beta
                    dr_trim[m] = dr
                ax.plot(aos, dr_trim, marker=markers[k], label='J =' + str(j_levels[k]), markerfacecolor='none')    # plot the

            ax.set_xlim([-0.5, 10.5])
            ax.set_ylim([-6, -35])
            ax.set_xlabel('Angle of Sideslip [deg]')
            ax.set_ylabel('$\delta r$ [-]')
            ax.grid()
            ax.legend()
            fig.tight_layout()
            # fig.savefig('Figures/cy_dr_re' + str(re) + '_aoa_' + str(aoa) + '.pdf')

            plt.show()

    pass


if __name__ == '__main__':
    Trimming()
