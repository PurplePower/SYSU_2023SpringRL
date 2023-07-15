import torch
import numpy as np
import matplotlib.pyplot as plt
from pettingzoo.mpe import simple_spread_v3
from time import sleep, time
from pathlib import Path

import os
os.environ['SUPPRESS_MA_PROMPT'] = "1"

import pygame
from  moviepy.video.io.ImageSequenceClip import ImageSequenceClip

from imgcat import imgcat

from utils.plots import plot_return_trace_area


if __name__ == '__main__':

    returns = np.load(r'SYSU_2023SpringRL\Assignment2\saves\good_to_save\[2]n_eps=20000-eps_len=25-a_lr=0.0001-c_lr=0.001\returns.npz')
    returns = returns['arr_0']

    plot_return_trace_area([returns], ['MADDPG'])

    plt.savefig('trace_area.png')









