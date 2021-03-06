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


class Tail_off:
    def __init__(self):
        self.Balance_data = loadmat('Data/Tail_off_balance.mat')['BAL']

    def windOn(self, key1, key2):
        vec = self.Balance_data["windOn"][0, 0][key1][0, 0][0, 0][key2]
        return vec


def main():

    # Import these classes into your code and get the values the following examples

    Tail_off_data = Tail_off()
    print(Tail_off_data.windOn('tailOff_alfa0_balance', 'FZ'))

    # For getting Balance example
    Balance_Data = Balance()
    my_bal_values = Balance_Data.windOn('rudderminus10deg', 'FZ')
    print(f'FZ in -10 deg rudder Wind On condition -- {my_bal_values}')

    # For pressure value
    Pressure_Data = Pressure()
    my_press_values = Pressure_Data.find('rudder0deg_restart', 'pTaps')
    print(f'Density values for 0 deg rudder during the restarted part of the experiment -- {my_press_values}')


if __name__ == '__main__':
    main()

