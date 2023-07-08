import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import shutil

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from env.grid_scenarios import MiniWorld, GridWorldEnv
from utils import print_grid_policy
from agents.DynamicProgramming import ValueIteration, PolicyIteration



# Hypar-parameters that could be helpful.
GAMMA = 0.9
EPSILON = 0.001
BLOCKS = [14, 15, 21, 27]
R = [
    0, 0, 0, 0, 0, 0,
    0, 0, 0, 1, 0, 0,
    0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, -1,
    0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0,
]




if __name__ == "__main__":
    env = MiniWorld()
    n_state = env.observation_space.n
    n_action = env.action_space.n

    ######################################################
    # write your code to get a convergent value table v. #
    ######################################################
    
    np.set_printoptions(3)
    
    VALUE_ITERATION, POLICY_ITERATION = 0, 1
    run_what = POLICY_ITERATION
    theta = 1e-3
    
    save_path = Path('images')
    
    if run_what == VALUE_ITERATION:
        p = save_path / 'VI'
        if p.exists():
            shutil.rmtree(p)
        agent = ValueIteration(env, theta, GAMMA, p)
        agent.value_iteration()
    else:
        p = save_path / 'PI'
        if p.exists():
            shutil.rmtree(p)
        agent = PolicyIteration(env, theta, GAMMA, p)
        agent.policy_iteration()
    

    # env.render(agent.v.flatten())
    
    pass
