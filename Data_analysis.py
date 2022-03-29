import numpy as np
import pandas as pd
from select_data import select_data_txt
from trim_conditions import trim
import matplotlib as plt


def Direction_stability():
    data = np.genfromtxt('Data_txt/Analysis_data.txt', skip_header=1)
    data_columns = open('Data_txt/Analysis_data.txt', 'r').readlines()[0].split()[1:]
    data_f = pd.DataFrame(data, columns=data_columns)
    print(np.unique(data_f['AoA']))




def Rudder_effectiveness():
    pass


def Trimming():
    data = np.genfromtxt('Data_txt/Analysis_data.txt', skip_header=1)
    re_levels  = np.unique(np.round(data[:, 6] / 1e5, 1)) * 1e5          # Re  = [340000. 450000.]
    aoa_levels = np.unique(np.round(data[:, 2], 1))                      # AoA = [0. 5.]
    aos_levels = np.unique(np.round(data[:, 3], 1))                      # AoS = [-5.  0.  5. 10.]
    j_levels   = np.unique(np.round(data[:, 4], 1))                      # j   = [1.6 1.8 2.  2.1 2.2 2.3]

    # Reynolds low
    for i in range(len(aos_levels)):
        dr_trim = trim(aos_levels[i], aoa_levels[i], re_levels[i], j_levels[i])     # find rudder deflection

    print(dr_trim)

    pass


if __name__ == '__main__':
    Trimming()

