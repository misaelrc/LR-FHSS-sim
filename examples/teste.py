import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import EngFormatter
import scienceplots



#     prop_cycle = plt.rcParams['axes.prop_cycle']
#     colors = prop_cycle.by_key()['color']
#     pparam = dict(ylabel='Average success delivery ratio',xlabel='Number of nodes')
#     fig, ax = plt.subplots()
#     ax.xaxis.set_major_formatter(f_x)
#     ax.plot(nNodes*8, example_1['Success'],color='red', label='LR-FHSS')
#     ax.plot(nNodes*8, example_2['Success'],color='blue', label='ACRDA')
    


nNodes_points = 20
nNodes_min = 1000
nNodes_max = 100000
nNodes = np.linspace(nNodes_min, nNodes_max, nNodes_points, dtype=int)//8

example_1 = pd.read_pickle('example_1.data')
example_2 = pd.read_pickle('example_2.data')

f_x = EngFormatter()

with plt.style.context(['science', 'ieee', 'no-latex']):
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    pparam = dict(ylabel='Network average success',xlabel='Average number of end-devices')
    fig, ax = plt.subplots()

    ax.xaxis.set_major_formatter(f_x)
    plt.plot(nNodes * 8, example_1['Success'], '-o', color='red', label='Exponential')  
    plt.plot(nNodes * 8, example_2['Success'], '--o', color='red', label='Exponential ACRDA')

    leg = ax.legend(loc=3)

    ax.grid(ls='--', color='lightgray')
    ax.autoscale(tight=True)
    ax.set(**pparam)
    ax.set_ylim(top=1)
    plt.tight_layout()
    plt.show()
    plt.close()