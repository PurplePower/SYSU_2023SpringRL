import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from utils.plots import plot_return_trace_area





if __name__ == '__main__':
    cur_path = Path(r'SYSU_2023SpringRL\Assignment2')
    save_path = Path(r'SYSU_2023SpringRL\Assignment2\saves\good_to_save')

    # load trivial MADDPG
    trivial_trace = np.load(save_path / r'[2]n_eps=20000-eps_len=25-a_lr=0.0001-c_lr=0.001/returns.npz')['arr_0']

    # load PER MADDPG
    per_trace = np.load(save_path / r'PER-n_eps=20000-eps_len=25-a_lr=0.0001-c_lr=0.001/returns.npz')['arr_0']

    # load ReLo 
    relo_trace = np.load(save_path / r'Relo-n_eps=20000-eps_len=25-a_lr=0.0001-c_lr=0.001\returns.npz')['arr_0']


    plot_return_trace_area([per_trace, relo_trace], ['PER MADDPG', 'ReLo'])
    # plot_return_trace_area([trivial_trace], ['Trivial MADDPG'])

    plt.savefig(cur_path / 'training_trace.png')
    plt.savefig(cur_path / 'training_trace.pdf')


    pass







