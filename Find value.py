import numpy as np
import pandas as pd

data            = np.genfromtxt('Data_txt/test_data.txt', skip_header = 1)
data_columns = open('Data_txt/test_data.txt', 'r').readlines()[0].split()[1:]
data_f = pd.DataFrame(data, columns=data_columns)

data_tail_off = np.genfromtxt('Data_txt/test_data.txt', skip_header = 1)
data_columns_off = open('Data_txt/test_data.txt', 'r').readlines()[0].split()[1:]
data_off = pd.DataFrame(data_tail_off, columns=data_columns_off)

# print(data_f['AoA'])
print(data_off)
