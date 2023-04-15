import os
import sys
import time
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt

from env.grid_scenarios import MiniWorld
from TemporalDifference import TemporalDifference as TD
from utils import print_grid_policy, plot_return_trace_area


GAMMA = 0.9
EPSILON = 0.1
LR = 0.1
BLOCKS = [14, 15, 21, 27]


def temporal_difference(env):
    pass


def show_variance_of_steps():
    n_steps = list(reversed([1, 16, np.inf]))    # the larger n_step, the larger variance
    drop_remainder = True
    n_episode = 1000
    
    env = MiniWorld()
    n_state = env.n_height * env.n_width
    n_action = env.action_space.n
    
    total_returns = []
    
    for n_step in n_steps:
        print(f'Running {n_step=} TD with {n_episode=}...')
        agent = TD(
            n_step=n_step, n_state=n_state, n_action=n_action, env=env, drop_remainder=drop_remainder,
            gamma=GAMMA, alpha=LR, epsilon=EPSILON,
            log_stdout=False
        )
        
        return_history = agent.policy_iteration(n_episode, record_return=True)
        total_returns.append(return_history)
        
    # plt.figure()
    # for n_step, return_history in zip(n_steps, total_returns):
    #     plt.plot(np.arange(len(return_history)), return_history)
    # plt.legend([f'{n_step=}' if n_step != np.inf else 'MC' for n_step in n_steps])
    # plt.show()
    
    labels = [f'{n_step=}' if n_step != np.inf else 'MC' for n_step in n_steps]
    plot_return_trace_area(total_returns, window=10, labels=labels)
    plt.show()


if __name__ == "__main__":
    np.set_printoptions(4)
    np.random.seed(1977)
    
    if True:
    
        env = MiniWorld()
        n_state = env.n_height * env.n_width
        n_action = env.action_space.n
        n_step = 16
        
        # agent = TD(
        #     n_step=n_step, n_state=n_state, n_action=n_action, env=env, 
        #     gamma=GAMMA, alpha=LR, epsilon=EPSILON,
        #     log_file=f'logs/TD({n_step}) {{time}}.log'
        # ) 
        agent = TD(
            n_step=n_step, n_state=n_state, n_action=n_action, env=env, 
            gamma=GAMMA, alpha=LR, epsilon=EPSILON, on_policy=True
        )  # no log
        
        start = time.time()
        return_history = agent.policy_iteration(1000, record_return=True)
        print(f'Cost {time.time()-start:.2f} s')
        
        print_grid_policy(agent.q, env)
        
        v = agent.v
        env.render(v, pi=agent.pi, title=f'Final TD{n_step}')
        
        print(np.flipud(v.reshape((env.n_height, -1))))
        
        plt.plot(np.arange(len(return_history)), return_history)
        plt.xlabel('Episodes')
        plt.ylabel('Return')
        plt.show()
        
    else:
        show_variance_of_steps()
    
    pass
    
    
    
    
    
