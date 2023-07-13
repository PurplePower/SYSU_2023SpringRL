import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

from typing import List

def plot_return_trace_area(episode_returns:List[list], labels=None, window=10):
    
    def interpolate_trace(x, y, new_x):
        spl = CubicSpline(x, y)
        return spl(new_x)
    

    if window % 2 == 0:
        window += 1
    
    plt.figure()
    mean_lines = []
    for i, eps_return in enumerate(episode_returns):
        
        mean_trace = np.correlate(eps_return, np.ones(window)/window, mode='valid')
        t = np.arange(len(mean_trace))
        cutted_eps = eps_return[window//2:-window//2+1]
        full_trace = np.vstack([cutted_eps, t]).T
        upper_bound = full_trace[cutted_eps >= mean_trace]
        lower_bound = full_trace[cutted_eps < mean_trace]
        
        upper_bound = interpolate_trace(upper_bound[:, 1], upper_bound[:, 0], t)
        lower_bound = interpolate_trace(lower_bound[:, 1], lower_bound[:, 0], t)
        
        
        line_alpha, linewidth = 0.5, 1
        line = plt.plot(t, upper_bound, alpha=0.5, linewidth=linewidth)[0]
        plt.plot(t, lower_bound, color=line.get_color(), alpha=0.5, linewidth=linewidth)
        plt.fill_between(t, upper_bound, lower_bound, color=line.get_color(), alpha=0.2)
        mean_line, = plt.plot(t, mean_trace, color=line.get_color())
        mean_lines.append(mean_line)
        
    plt.legend(mean_lines, labels)