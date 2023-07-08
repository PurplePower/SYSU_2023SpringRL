from pettingzoo.mpe._mpe_utils.simple_env import SimpleEnv
from pettingzoo import AECEnv
from pettingzoo.utils.wrappers import BaseWrapper



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

