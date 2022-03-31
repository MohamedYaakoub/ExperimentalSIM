from scipy.io import loadmat
import numpy as np
# from open_files import Balance, Pressure, Tail_off, Microphone


class Microphone:
    def __init__(self):
        self.Microphone_data = loadmat('Data/MICROPHONE.mat')['Mic']

    def find(self, key1, key2):
        vec = self.Microphone_data[key1][0, 0][0, 0][key2]

        return vec


# SPL -> [0][Test point][:, Microphone number] (Whole range of frequencies covered by '[:')
# data_sets = ['rudder0deg', 'rudderminus10deg', 'rudder0deg_remeasure', 'rudder0deg_restart']
# print(Microphone().find(data_sets[0], 'SPL')[0][1][:, 1])
# print(Microphone().find(data_sets[0], 'f')[0][0][:, 1])


def write_data_mic(mic_number):

    # Datasets and variables that need to be stored
    data_sets = ['rudder0deg', 'rudderminus10deg', 'rudder0deg_remeasure', 'rudder0deg_restart', 'rudder0deg_elevator5deg']
    # rudder    = [0, -10, 0, 0]
    variables = ['SPL', 'f']

    # Load microphone data
    Mic_Data = Microphone()

    # Array to store all the data in, initialise it with frequency
    # data = np.zeros((1, len(Mic_Data.find('rudder0deg', 'SPL')[0][0][:, 0])))
    data = Mic_Data.find('rudder0deg', 'f')[0][0][:, 0]

    for i in range(len(data_sets)):

        # # Make an empty array to store the data for this dataset
        # # set_array = np.zeros((len(Mic_Data.find(data_sets[i], variables[0])[:, 0]), len(variables)-2))
        # set_array = np.zeros((2, len(Mic_Data.find(data_sets[i], 'SPL')[0][0][:, mic_number])))

        # SPL -> [0][Test point][:, Microphone number] (Range of frequencies -> [:)
        # Microphone().find('rudder0deg', 'SPL')[0][0][:, 0]

        # Loop through all test points
        for j in range(len(Mic_Data.find(data_sets[i], 'SPL')[0])):
            # Make an empty array to store the data for this dataset
            # set_array = np.zeros((len(Mic_Data.find(data_sets[i], variables[0])[:, 0]), len(variables)-2))
            set_array = np.zeros((1, len(Mic_Data.find(data_sets[i], 'SPL')[0][j][:, mic_number])))

            # # First row is f
            # # set_array[0, 1] = 'f'
            # set_array[0, :] = Mic_Data.find(data_sets[i], 'f')[0][j][:, 0]

            # Second row is SPL
            # set_array[0, 0] = 'SPL'
            set_array[0, :] = Mic_Data.find(data_sets[i], 'SPL')[0][j][:, mic_number]

            # for j in range(len(variables)):
            #     if j == 1:
            #         set_array[:, j] = rudder[i]
            #     # Pressure data contains six fields, these are appended to the end of the array
            #     elif variables[j] == 'SPL' or variables[j] == 'f':
            #         set_array = np.hstack((set_array, Mic_Data.find(data_sets[i], variables[j])[:, mic_number]))
            #     else:
            #         # Store the data for each variable
            #         set_array[:, j] = Mic_Data.find(data_sets[i], variables[j])[:, 0]

            # Append data from dataset to big array
            data = np.vstack((data, set_array))

    # Make header for the file
    SPL_hdr = []
    f_hdr = []
    for i in range(6):
        SPL_hdr.append('SPL' + str(i))
        f_hdr.append('f' + str(i))

    header = variables[:-2] + SPL_hdr + f_hdr

    np.savetxt('Data_txt\mic_' + str(mic_number + 1) + '.txt', data, fmt='%10.5f')  # [1:, :], header='         '.join(header), fmt='%10.5f')


# Save files for the 6 microphones
for i in range(6):
    write_data_mic(i)
