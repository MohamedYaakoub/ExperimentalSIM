import numpy as np
from scipy.io import loadmat
from matplotlib import pyplot as plt


class Balance:
    def __init__(self):
        self.Balance_data = loadmat('Data/BALANCE.mat')['BAL']

    def windOff(self, key1, key2):
        vec = self.Balance_data["windOff"][0, 0][key1][0, 0][0, 0][key2]
        return vec

    def windOn(self, key1, key2):
        vec = self.Balance_data["windOn"][0, 0][key1][0, 0][0, 0][key2]
        return vec


class Pressure:
    def __init__(self):
        self.Pressure_data = loadmat('Data/PRESSURE.mat')['PRS']

    def find(self, key1, key2):
        vec = self.Pressure_data[key1][0, 0][0, 0][key2]

        return vec


class Microphone:
    def __init__(self):
        self.Microphone_data = loadmat('Data/MICROPHONE.mat')['Mic']

    def find(self, key1, key2):
        vec = self.Microphone_data[key1][0, 0][0, 0][key2]

        return vec


def main():

    # Import these classes into your code and get the values the following examples

    # For getting Balance example
    Balance_Data = Balance()
    my_bal_values = Balance_Data.windOn('rudderminus10deg', 'FZ')
    print(f'FZ in -10 deg rudder Wind On condition -- {my_bal_values}')

    # For pressure value

    Pressure_Data = Pressure()
    my_press_values = Pressure_Data.find('rudder0deg_restart', 'rho')
    print(f'Density values for 0 deg rudder during the restarted part of the experiment -- {my_press_values}')


if __name__ == '__main__':
    main()


# class Balance:
#     def __init__(self):
#         self.Balance_data = loadmat('BALANCE.mat')['BAL']
#
#         self.windOff_rudder0deg = pd.DataFrame(
#             np.transpose(np.array(list(self.Balance_data["windOff"][0, 0]['rudder0deg'][0, 0][0, 0])).reshape((14, 8))),
#             columns=['run', 'hr', 'min', 'sec', 'AoA', 'AoS', 'pBar', 'temp', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6'])
#
#         self.windOff_rudderminus10deg = pd.DataFrame(
#             np.transpose(np.array(list(self.Balance_data["windOff"][0, 0]['rudderminus10deg'][0, 0][0, 0])).reshape((14, 8))),
#             columns=['run', 'hr', 'min', 'sec', 'AoA', 'AoS', 'pBar', 'temp', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6'])
#
#         self.windOff_rudder0deg_restart = pd.DataFrame(
#             np.transpose(np.array(list(self.Balance_data["windOff"][0, 0]['rudder0deg_restart'][0, 0][0, 0])).reshape((14, 8))),
#             columns=['run', 'hr', 'min', 'sec', 'AoA', 'AoS', 'pBar', 'temp', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6'])
#
#         self.windOff_rudder0deg_elevator5deg= pd.DataFrame(
#             np.transpose(np.array(list(self.Balance_data["windOff"][0, 0]['rudder0deg_elevator5deg'][0, 0][0, 0])).reshape((14, 8))),
#             columns=['run', 'hr', 'min', 'sec', 'AoA', 'AoS', 'pBar', 'temp', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6'])
#
#         self.windOn_rudder0deg = pd.DataFrame(
#             np.transpose(np.array(list(self.Balance_data["windOn"][0, 0]['rudder0deg'][0, 0][0, 0])).reshape((14, 49))),
#             columns=['run', 'hr', 'min', 'sec', 'AoA', 'AoS', 'pBar', 'temp', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6'])
#
#         self.windOn_rudderminus10deg = pd.DataFrame(
#             np.transpose(
#                 np.array(list(self.Balance_data["windOn"][0, 0]['rudderminus10deg'][0, 0][0, 0])).reshape((14, 49))),
#             columns=['run', 'hr', 'min', 'sec', 'AoA', 'AoS', 'pBar', 'temp', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6'])
#
#         self.windOn_rudder0deg_restart = pd.DataFrame(
#             np.transpose(
#                 np.array(list(self.Balance_data["windOn"][0, 0]['rudder0deg_restart'][0, 0][0, 0])).reshape((14, 49))),
#             columns=['run', 'hr', 'min', 'sec', 'AoA', 'AoS', 'pBar', 'temp', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6'])
#
#         self.windOn_rudder0deg_elevator5deg = pd.DataFrame(
#             np.transpose(
#                 np.array(list(self.Balance_data["windOn"][0, 0]['rudder0deg_elevator5deg'][0, 0][0, 0])).reshape(
#                     (14, 49))),
#             columns=['run', 'hr', 'min', 'sec', 'AoA', 'AoS', 'pBar', 'temp', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6'])
import numpy as np
from scipy.io import loadmat
import pandas as pd
from matplotlib import pyplot as plt


class Balance:
    def __init__(self):
        self.Balance_data = loadmat('BALANCE.mat')['BAL']

    def windOff(self, key1, key2):
        vec = self.Balance_data["windOff"][0, 0][key1][0, 0][0, 0][key2]
        return vec

    def windOn(self, key1, key2):
        vec = self.Balance_data["windOn"][0, 0][key1][0, 0][0, 0][key2]
        return vec


class Pressure:
    def __init__(self):
        self.Pressure_data = loadmat('PRESSURE.mat')['PRS']

    def find(self, key1, key2):
        vec = self.Pressure_data[key1][0, 0][0, 0][key2]

        return vec


def main():

    # Import these classes into your code and get the values the following examples

    # For getting Balance example
    Balance_Data = Balance()
    my_bal_values = Balance_Data.windOn('rudderminus10deg', 'FZ')
    print(f'FZ in -10 deg rudder Wind On condition -- {my_bal_values}')

    # For pressure value

    Pressure_Data = Pressure()
    my_press_values = Pressure_Data.find('rudder0deg_restart', 'rho')
    print(f'Density values for 0 deg rudder during the restarted part of the experiment -- {my_press_values}')


if __name__ == '__main__':
    main()





# class Balance:
#     def __init__(self):
#         self.Balance_data = loadmat('BALANCE.mat')['BAL']
#
#         self.windOff_rudder0deg = pd.DataFrame(
#             np.transpose(np.array(list(self.Balance_data["windOff"][0, 0]['rudder0deg'][0, 0][0, 0])).reshape((14, 8))),
#             columns=['run', 'hr', 'min', 'sec', 'AoA', 'AoS', 'pBar', 'temp', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6'])
#
#         self.windOff_rudderminus10deg = pd.DataFrame(
#             np.transpose(np.array(list(self.Balance_data["windOff"][0, 0]['rudderminus10deg'][0, 0][0, 0])).reshape((14, 8))),
#             columns=['run', 'hr', 'min', 'sec', 'AoA', 'AoS', 'pBar', 'temp', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6'])
#
#         self.windOff_rudder0deg_restart = pd.DataFrame(
#             np.transpose(np.array(list(self.Balance_data["windOff"][0, 0]['rudder0deg_restart'][0, 0][0, 0])).reshape((14, 8))),
#             columns=['run', 'hr', 'min', 'sec', 'AoA', 'AoS', 'pBar', 'temp', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6'])
#
#         self.windOff_rudder0deg_elevator5deg= pd.DataFrame(
#             np.transpose(np.array(list(self.Balance_data["windOff"][0, 0]['rudder0deg_elevator5deg'][0, 0][0, 0])).reshape((14, 8))),
#             columns=['run', 'hr', 'min', 'sec', 'AoA', 'AoS', 'pBar', 'temp', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6'])
#
#         self.windOn_rudder0deg = pd.DataFrame(
#             np.transpose(np.array(list(self.Balance_data["windOn"][0, 0]['rudder0deg'][0, 0][0, 0])).reshape((14, 49))),
#             columns=['run', 'hr', 'min', 'sec', 'AoA', 'AoS', 'pBar', 'temp', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6'])
#
#         self.windOn_rudderminus10deg = pd.DataFrame(
#             np.transpose(
#                 np.array(list(self.Balance_data["windOn"][0, 0]['rudderminus10deg'][0, 0][0, 0])).reshape((14, 49))),
#             columns=['run', 'hr', 'min', 'sec', 'AoA', 'AoS', 'pBar', 'temp', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6'])
#
#         self.windOn_rudder0deg_restart = pd.DataFrame(
#             np.transpose(
#                 np.array(list(self.Balance_data["windOn"][0, 0]['rudder0deg_restart'][0, 0][0, 0])).reshape((14, 49))),
#             columns=['run', 'hr', 'min', 'sec', 'AoA', 'AoS', 'pBar', 'temp', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6'])
#
#         self.windOn_rudder0deg_elevator5deg = pd.DataFrame(
#             np.transpose(
#                 np.array(list(self.Balance_data["windOn"][0, 0]['rudder0deg_elevator5deg'][0, 0][0, 0])).reshape(
#                     (14, 49))),
#             columns=['run', 'hr', 'min', 'sec', 'AoA', 'AoS', 'pBar', 'temp', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6'])