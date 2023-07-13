import numpy as np
from pettingzoo.mpe._mpe_utils.simple_env import SimpleEnv
from pettingzoo import AECEnv
from pettingzoo.utils.wrappers import BaseWrapper
import gymnasium as gym

import sys
import os

sys.path.append('multiagent-particle-envs')

from multiagent.environment import MultiAgentEnv



"""
AECEnv is the most basic environment class.
SimpleEnv is subclass of AECEnv, includes additional stuffs like 
rendering.

BaseWrapper is subclass of AECEnv, all wrappers are subclass of BaseWrapper.
wrappers hold the wrapped raw_env in `env`, raw_env is subclass of SimpleEnv.


"""

def get_simple_env(env) -> SimpleEnv:
    if env.__class__ == AECEnv:
        raise f'env is the most basic class'
    
    if isinstance(env, BaseWrapper):
        while not isinstance(env, SimpleEnv):
            env = env.env

    return env


class ClassicalEnv:
    """
    Classical environment. Use it just like single agent env except that
    you pass actions and get observations of all agents.
    """

    def __init__(self) -> None:
        self.action_spaces = self.state_space = self.action_space = None
        self.action_dim = self.state_dim = 0
        self.discrete_action = True
        pass

    def reset(self):
        pass


    def step(self, actions):
        pass


    def close(self):
        pass

    
    def render(self):
        pass


    @property
    def num_agents(self):
        pass

    @property
    def agents(self):
        pass





class PettingZooWrapperEnv(ClassicalEnv):

    def __init__(self, pz_env: BaseWrapper|SimpleEnv) -> None:
        super().__init__()
        pz_env.reset()

        self.action_spaces = pz_env.action_spaces
        self.action_space = pz_env.action_space
        self.state_space = pz_env.state_space
        self.env = pz_env

        self.discrete_action = True
        act_s = pz_env.action_space(pz_env.agents[0])
        if isinstance(act_s, gym.spaces.Box):
            # continuous
            self.action_dim = act_s.shape[0] # velocity in 4 directions, plus a no_action
            self.discrete_action = False
        else:
            self.action_dim = pz_env.action_space(pz_env.agents[0]).n
        self.state_dim = pz_env.state_space.shape[0]


    def _get_obs(self):
        return [self.env.observe(a) for a in self.agents]
    

    def _get_rewards(self):
        return [self.env.rewards[a] for a in self.agents]

    def reset(self):
        self.env.reset()
        return self._get_obs()


    def step(self, actions):
        for act in actions:
            if isinstance(act, np.ndarray):
                act = np.squeeze(act)
            self.env.step(act)

        # return next_states, rewards, done, infos
        next_states = self._get_obs()
        rewards = self._get_rewards()
        all_terminated = [self.env.terminations[a] for a in self.agents]
        all_truncated = [self.env.truncations[a] for a in self.agents]
        done = all(all_terminated) or all(all_truncated)
        infos = self.env.infos

        return next_states, rewards, done, infos
    

    def close(self):
        return self.env.close()
    

    def render(self):
        return self.env.render()


    @property
    def num_agents(self):
        return self.env.num_agents
    
    @property
    def agents(self):
        return self.env.agents

    pass



class OpenAIWrapperEnv(ClassicalEnv):

    def __init__(self, env:MultiAgentEnv) -> None:
        super().__init__()
        self.env = env
        self.discrete_action = False    # only continuous
        self.action_space = self.action_spaces = env.action_space
        self.state_space = env.observation_space
        self.action_dim = env.action_space[0].n    # env.action_space is a list of Space
        self.state_dim = sum([s.shape[0] for s in env.observation_space])


    def _process_dtype(self, s):
        return [_s.astype(np.float32) for _s in s]
    

    def reset(self):
        return self._process_dtype(self.env.reset())
    
    
    def step(self, actions):
        # actions are continuous actions
        s, r, done_list, i = self.env.step(actions)
        s = self._process_dtype(s)
        r = self._process_dtype(r)
        done = all(done_list)
        return s, r, done, i
    

    def close(self):
        return self.env.close()
    

    def render(self, mode='rgb_array'):
        return self.env.render(mode)[0]


    @property
    def num_agents(self):
        return self.env.n
    

    @property
    def agents(self):
        return self.env.agents



