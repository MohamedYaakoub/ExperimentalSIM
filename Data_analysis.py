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
    aos_levels = aos_levels[aos_levels >= 0]                             # Check if >= 0 then keep it

    dr_trim = np.zeros(len())
    for i in range(len(aos_levels)):
        for k in range(len(re_levels)):
            j_m1 = select_data_txt(['Re'], [re_levels[k]], ['J_M1'], file_name='Data_txt/Analysis_data.txt')    # select j based on Re
            j_levels = np.unique(np.round(j_m1, 1))
            for l in range(len(j_levels)):
                dr = trim(aos_levels[i], aoa_levels[1], re_levels[k], j_levels[l])     # find rudder deflection



    pass


if __name__ == '__main__':
    Trimming()

