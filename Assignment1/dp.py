import os
import sys
from copy import deepcopy
from itertools import product

import numpy as np
from gymnasium.spaces import Discrete
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from env.grid_scenarios import MiniWorld, GridWorldEnv
from utils import print_grid_policy




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




class PolicyIteration:
    """
        Initial policy is random (equally take each action).
    """

    def __init__(self, env: MiniWorld, theta, gamma) -> None:
        self.env = env
        self.theta = theta
        self.gamma = gamma
        self.n_action = env.action_space.n
        
        # value function V(s) of current policy pi
        self.v = np.zeros((env.n_height, env.n_width)) 
        # policy pi(a|s) = pi(*s, a)
        self.pi = np.ones((env.n_height, env.n_width, self.n_action)) * (1 / self.n_action) 
        
        # we don't store q(s, a) explicitly, but compute it from v when needed
            
        
        
    def policy_evaluation(self, max_iter=1000, policy_print_period=5):
        # for each state, update V(s)
        env = self.env
        
        for iter in range(max_iter):
            new_v = np.zeros_like(self.v)
            # if iter % policy_print_period == 0:
            #     print(f'V(s) before iter {iter}:')
            #     print(f'{self.v}')
            #     print(f'Policy greedy of this V(s):\n {print_policy(self.get_policy_from_v(self.v))}\n')
                
                
            for y, x in product(range(env.n_height), range(env.n_width)):
                if (x, y, 1) in env.types or (x, y) in self.env.ends:
                    continue    # skip wall cells, wall cell values are never accessed since we never step on them.
                
                for act in range(self.n_action):
                    
                    # a single next_state is returned for deterministic env., otherwise a list of possible next states
                    env.state = env.xy_to_state(x, y)
                    next_state, reward, done, info = env.step(act)
                    next_state = env.state_to_xy(next_state)
                    next_x, next_y = next_state
                    new_v[y, x] += self.pi[y, x, act] * (
                        reward + self.gamma * 1 * self.v[next_y, next_x] * (1-done)) 

            diff = np.max(np.absolute(new_v - self.v))
            self.v = new_v
            if diff < self.theta:
                break
            
            
            
            
    def get_policy_from_v(self, v):
        """
            for each state s, take the action a with largest q(s, a)
            if multiple actions has the same largest q(s, a), then divide probability uniformly
        """
        new_pi = np.zeros_like(self.pi)
        env = self.env
        
        for y, x in product(range(env.n_height), range(env.n_width)):
            qsa_list = []
            for act in range(self.n_action):
                env.state = env.xy_to_state(x, y)
                next_state, reward, done, info = env.step(act)
                next_state = env.state_to_xy(next_state)
                next_x, next_y = next_state
                qsa = 1 * (reward + self.gamma * v[next_y, next_x] * (1 - done))
                qsa_list.append(qsa)
                
            max_qsa = max(filter(lambda x: x is not None, qsa_list))
            max_qsa_cnt = qsa_list.count(max_qsa)
            
            new_pi[y, x] = np.array([1 / max_qsa_cnt if qsa == max_qsa else 0 for qsa in qsa_list])
            
        return new_pi
        
        
        
    def policy_improvement(self):
        new_pi = self.get_policy_from_v(self.v)
        self.pi = new_pi
            
        
        
    def policy_iteration(self):
        max_iter = 100
        for i in range(max_iter):
            print(f'Running policy iteration {i}...')
            print(f'Before iteration {i}, V(s) = \n{np.flipud(self.v)}\n\n')
            print(f'Policy is ')
            print_grid_policy(self.pi.reshape((-1, self.pi.shape[-1])), self.env)
            self.env.render(
                self.v.flatten(), pi=self.pi.reshape((-1, self.pi.shape[-1])), title=f'PolicyIter {i}')
            
            self.policy_evaluation()
            old_pi = deepcopy(self.pi)
            self.policy_improvement()
            
            if np.all(old_pi == self.pi):
                break


              
class ValueIteration:
    
    def __init__(self, env: GridWorldEnv, theta, gamma) -> None:
        self.env = env
        self.theta = theta
        self.gamma = gamma
        self.n_action = env.action_space.n
        
        self.v = np.zeros((env.n_height, env.n_width))
        
        
    def value_iteration(self, max_iter=1000):
        env = self.env

        for i in range(max_iter):
            print(f'Before iter {i}, V*(s)=\n{np.flipud(self.v)}')
            print(f'Policy is ')
            print_grid_policy(self.get_policy(), self.env)
            env.render(self.v.flatten(), pi=self.get_policy(), title=f'ValueIter {i}')
            
            new_v = np.zeros_like(self.v)
            for y, x in product(range(env.n_height), range(env.n_width)):
                qsa_list = []
                for act in range(self.n_action):
                    env.state = env.xy_to_state(x, y)
                    next_state, reward, done, info = env.step(act)
                    next_x, next_y = env.state_to_xy(next_state)
                    
                    qsa = reward + self.gamma * self.v[next_y, next_x] * (1 - done)
                    qsa_list.append(qsa)
                    
                new_v[y, x] = max(qsa_list)
                
            diff = np.max(np.absolute(new_v - self.v))
            self.v = new_v
            if diff < self.theta:
                break

            
    def get_policy(self, flatten=True):
        env = self.env
        pi = np.zeros((env.n_height, env.n_width, self.n_action))
        for y, x in product(range(env.n_height), range(env.n_width)):
            qsa_list = []
            for a in range(self.n_action):
                env.state = env.xy_to_state(x, y)
                next_state, reward, done, info = env.step(a)
                next_x, next_y = env.state_to_xy(next_state)
                qsa = reward + self.gamma * self.v[next_y, next_x] * (1 - done)
                qsa_list.append(qsa)
                
            max_q = max(qsa_list)
            cnt_q = qsa_list.count(max_q)
            pi[y, x] = np.array([1 / cnt_q if q == max_q else 0 for q in qsa_list])
        
        if flatten:
            pi = pi.reshape((-1, pi.shape[-1])) # in shape (n_state, n_action)
        return pi




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
    
    if run_what == VALUE_ITERATION:
        agent = ValueIteration(env, theta, GAMMA)
        agent.value_iteration()
    else:
    
        agent = PolicyIteration(env, theta, GAMMA)
        agent.policy_iteration()
    

    # env.render(agent.v.flatten())
    
    pass
