import numpy as np
import matplotlib.pyplot as plt
from open_files import Balance, Pressure, Tail_off


def write_to_text_bal():

    # Datasets and variables that need to be stored
    data_sets = ['rudder0deg', 'rudderminus10deg', 'rudder0deg_remeasure', 'rudder0deg_restart', 'rudder0deg_elevator5deg']
    rudder    = [0, -10, 0, 0, 0]
    elevator  = [0, 0, 0, 0, 5]
    variables = ['run', 'dr', 'AoA', 'AoS', 'J_M1', 'J_M2', 'Re', 'CL', 'CD', 'CY', 'CMpitch', 'CMyaw', 'rho', 'V', 'rpsM1', 'rpsM2', 'de']

    Balance_Data = Balance()

    # Array to store all the data in
    data = np.zeros((1, len(variables)))

    for i in range(len(data_sets)):

        # Make an empty array to store the data for this dataset
        set_array = np.zeros((len(Balance_Data.windOn(data_sets[i], variables[0])[:, 0]), len(variables)))

        for j in range(len(variables)):
            if j == 1:
                set_array[:, j] = rudder[i]
            elif j == 16:
                set_array[:, j] = elevator[i]
            else:
                # Store the data for each variable
                set_array[:, j] = Balance_Data.windOn(data_sets[i], variables[j])[:, 0]

        # Append data from dataset to big array
        data = np.vstack((data, set_array))

    # Make header for the file

    np.savetxt('test_data.txt', data[1:, :], header = '        '.join(variables), fmt = '%10.5f')


def write_to_text_bal_tailoff():

    # Datasets and variables that need to be stored
    data_sets = ['tailOff_alfa0_balance', 'tailOff_alfa0_V50_balance', 'tailOff_alfa5_betaSweep_balance',
                 'tailOff_alfa10_betaSweep_balance', 'tailOff_beta0_balance', 'tailOff_beta0_V50_balance']
    variables = ['run', 'AoA', 'AoS', 'Re', 'CL', 'CD', 'CY', 'CMpitch', 'CMyaw', 'rho', 'V', 'rpsM1', 'rpsM2']

    Balance_Data = Tail_off()

    # Array to store all the data in
    data = np.zeros((1, len(variables)))

    for i in range(len(data_sets)):

        # Make an empty array to store the data for this dataset
        set_array = np.zeros((len(Balance_Data.windOn(data_sets[i], variables[0])[:, 0]), len(variables)))

        for j in range(len(variables)):
            set_array[:, j] = Balance_Data.windOn(data_sets[i], variables[j])[:, 0]

        # Append data from dataset to big array
        data = np.vstack((data, set_array))

    # Make header for the file

    np.savetxt('tail_off_data.txt', data[1:, :], header = '        '.join(variables), fmt = '%10.5f')


def write_to_text_prs():

    # Datasets and variables that need to be stored
    data_sets = ['rudder0deg', 'rudderminus10deg', 'rudder0deg_remeasure', 'rudder0deg_restart']
    rudder    = [0, -10, 0, 0]
    variables = ['run', 'dr', 'AoA', 'AoS', 'pTaps', 'CpTaps']

    Pressure_Data = Pressure()

    # Array to store all the data in
    data = np.zeros((1, len(variables)+10))

    for i in range(len(data_sets)):

        # Make an empty array to store the data for this dataset
        set_array = np.zeros((len(Pressure_Data.find(data_sets[i], variables[0])[:, 0]), len(variables)-2))

        for j in range(len(variables)):
            if j == 1:
                set_array[:, j] = rudder[i]
            # Pressure data contains six fields, these are appended to the end of the array
            elif variables[j] == 'pTaps' or variables[j] == 'CpTaps':
                set_array = np.hstack((set_array, Pressure_Data.find(data_sets[i], variables[j])))
            else:
                # Store the data for each variable
                set_array[:, j] = Pressure_Data.find(data_sets[i], variables[j])[:, 0]

        # Append data from dataset to big array
        data = np.vstack((data, set_array))

    # Make header for the file
    p_hdr = []
    cp_hdr = []
    for i in range(6):
        p_hdr.append('pTaps' + str(i))
        cp_hdr.append('CpTaps' + str(i))

    header = variables[:-2] + p_hdr + cp_hdr

    np.savetxt('pressure_data.txt', data[1:, :], header = '         '.join(header), fmt = '%10.5f')


def select_data_txt(const_name, const_value, var_name, file_name = 'test_data.txt'):

    # Import data
    data            = np.genfromtxt(file_name, skip_header = 1)
    data_return     = np.zeros((len(data[:, 0]), len(var_name)))
    header_names = open(file_name, 'r').readlines()[0].split()[1:]

    # Make an empty array to store the data
    idx = np.ones(len(data[:, 0]), dtype=bool)

    # Go through the list of data that is kept constant and find the indices of where they are equal to the preset value
    for i in range(len(const_name)):

        const_idx = header_names.index(const_name[i])

        # Find indices of where the data should be selected
        if np.isclose(const_value[i], 0, atol = 1e-2):
            idx = idx * np.isclose(data[:, const_idx], const_value[i], atol=1e-2)

        else:
            idx = idx * np.isclose(data[:, const_idx], const_value[i], rtol=1e-2)

    # Store all the data needed in one array
    for j in range(len(var_name)):
        index   = header_names.index(var_name[j])
        data_return[:, j] = data[:, index]

    # Clip data and return
    return data_return[idx]


if __name__ == '__main__':
    """
    All possible variables:
    ('run', 'O'), ('hr', 'O'), ('min', 'O'), ('sec', 'O'), ('AoA', 'O'), ('AoS', 'O'), ('dPb', 'O'), ('pBar', 'O'), (
    'temp', 'O'), ('B', 'O'), ('B1', 'O'), ('B2', 'O'), ('B3', 'O'), ('B4', 'O'), ('B5', 'O'), ('B6', 'O'), (
    'rpmWT', 'O'), ('rho', 'O'), ('q', 'O'), ('V', 'O'), ('Re', 'O'), ('rpsM1', 'O'), ('rpsM2', 'O'), ('iM1', 'O'), (
    'iM2', 'O'), ('dPtQ', 'O'), ('pInf', 'O'), ('nu', 'O'), ('J_M1', 'O'), ('J_M2', 'O'), ('B16zeroed', 'O'), (
    'FX', 'O'), ('FY', 'O'), ('FZ', 'O'), ('MX', 'O'), ('MY', 'O'), ('MZ', 'O'), ('CFX', 'O'), ('CFY', 'O'), (
    'CFZ', 'O'), ('CMX', 'O'), ('CMY', 'O'), ('CMZ', 'O'), ('CN', 'O'), ('CT', 'O'), ('CY', 'O'), ('CL', 'O'), (
    'CD', 'O'), ('CYaw', 'O'), ('CMroll', 'O'), ('CMpitch', 'O'), ('CMpitch25c', 'O'), ('CMyaw', 'O'), ('b', 'O'), (
    'c', 'O'), ('S', 'O')]))
    """
    write_to_text_bal()
    write_to_text_bal_tailoff()

    # write_to_text_prs()

    J1 = select_data_txt(['AoA', 'Re', 'J_M1', 'dr'], [5, 339000, 1.6, 0], ['AoS', 'CMyaw'])
    J2 = select_data_txt(['AoA', 'Re', 'J_M1', 'dr'], [5, 339000, 1.6, -10], ['AoS', 'CMyaw'])

    aos1 = J1[:, 0]
    cn1  = J1[:, 1]

    aos2 = J2[:, 0]
    cn2  = J2[:, 1]

    plt.plot(aos1, cn1, label = 'rps = 92')
    plt.plot(aos2, cn2, label = 'rps = 66')
    plt.legend()
    plt.show()

    # write_to_text()

# def select_data(data_set, const_name, const_value, var_name):
#     """
#     :param data_set:        Dataset, for example, 'rudder0deg', 'rudderminus10deg', ...
#     :param const_name:      Name of the variables that are kept constant
#     :param const_value:     Value of the constant variable, for example const_name = 'rpsM1' and const_value = '92'
#                             only returns data points for which the rps of the first motor equals 92
#     :param var_name:        Names of the data that has to be returned
#     :return:
#     """
#
#     # Make an empty array to store the data
#     data = np.zeros((len(Balance().windOn(data_set, 'AoS')), len(var_name)))
#     idx  = np.ones(len(Balance().windOn(data_set, 'AoS')), dtype = bool)
#
#     # Get the balance data
#     Balance_Data = Balance()
#
#     # Go through the list of data that is kept constant and find the indices of where they are equal to the preset value
#     for i in range(len(const_name)):
#
#         # Find indices of where the data should be selected
#         if const_value[i] == 0:
#             idx = idx*np.isclose(Balance_Data.windOn(data_set, const_name[i])[:, 0], const_value[i], atol = 1e-2)
#         else:
#             idx = idx * np.isclose(Balance_Data.windOn(data_set, const_name[i])[:, 0], const_value[i], rtol=1e-2)
#
#     # Store all the data needed in one array
#     for j in range(len(var_name)):
#
#         data[:, j] = Balance_Data.windOn(data_set, var_name[j])[:, 0]
#
#     # Clip data and return
#     return data[idx]
