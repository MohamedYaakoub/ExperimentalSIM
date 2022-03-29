import numpy as np
import pandas as pd
import matplotlib as plt


def Direction_stability():
    data = np.genfromtxt('Data_txt/Analysis_data.txt', skip_header=1)
    data_columns = open('Data_txt/Analysis_data.txt', 'r').readlines()[0].split()[1:]
    data_f = pd.DataFrame(data, columns=data_columns)
    pass


def Rudder_effectiveness():
    pass


def Trimming():
    pass


if __name__ == '__main__':
    pass
