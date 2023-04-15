import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from typing import Union

import numpy as np
from loguru import logger

from GeneralPolicyIteration import GeneralPolicyIteration
from env.grid_scenarios import GridWorldEnv

class TemporalDifference(GeneralPolicyIteration):
    """
    n-step TD on-policy.
    """
    
    def __init__(
        self, n_step, n_state, n_action, env: GridWorldEnv, on_policy=True, gamma=0.9, alpha=0.1, epsilon=0.1,
        drop_remainder=False, random_s0=True, log_file=None, log_file_level='INFO', log_stdout=False
    ) -> None:
        super().__init__(gamma)
        self.n_step = n_step
        self.n_state = n_state
        self.n_action = n_action
        self.env = env
        
        self.on_policy = on_policy
        self.drop_remainder = drop_remainder
        self.random_s0 = random_s0
        
        self.alpha = alpha
        self.epsilon = epsilon
        
        self.q = np.zeros((n_state, n_action)) / n_action
        
        logger.remove()
        if log_stdout:
            logger.add(sys.stdout, format='{message}', colorize=True, level='INFO')
        if log_file is not None:
            logger.add(log_file, format='{time} {message}', colorize=True, level=log_file_level)
        
        
    def _check_param(self):
        super()._check_param()
        assert self.n_step >= 1 and self.n_state >= 1 and self.n_action >= 1
        assert self.epsilon <= 1
    
        
    def take_behavioural_action(self, s):
        """
        Take an action from behavioural policy (epsilon-greedy).
        """
        greedy_act, greedy_act_prob = self.take_greedy_action(s)
        eps = self.epsilon
        if np.random.rand() < self.epsilon:
            # take random action
            a = np.random.choice(self.n_action)
            p = eps / self.n_action
            if self.q[s, a] == self.q[s].max():
                p += (1 - eps) / np.sum(self.q[s] == self.q[s].max())
        else:
            # return greedy action
            a, p = greedy_act, (1 - eps) / np.sum(self.q[s] == self.q[s].max()) + eps / self.n_action
            
        return a, p
            
            
    def take_greedy_action(self, s):
        """
        If multiple actions have the same max q(s, a), randomly draw one.
        """
        t = self.q[s] == self.q[s].max()
        n_max = np.sum(t)
        a = np.random.choice(np.flatnonzero(t))
        return a, 1 / n_max
        
        
        
    def take_target_action(self, s):
        """
        Take action from target policy. 
        """
        if self.on_policy:
            return self.take_behavioural_action(s)
        else:
            return self.take_greedy_action(s)


    def _get_target_action_prob(self, s, a):
        """
        Get target_pi(a).
        """
        if self.on_policy:      # prob from a epsilon-greedy policy
            eps = self.epsilon
            p = eps / self.n_action
            if self.q[s, a] == self.q[s].max():
                p += (1 - eps) / np.sum(self.q[s] == self.q[s].max())

            return p      
        else:       # prob from a greedy policy
            if self.q[s, a] == self.q[s].max():
                return 1 / np.sum(self.q[s] == self.q[s].max())
            else:
                return 0
        
    
    def _policy_evaluate_and_improve(self, history, s_next, a_next, im_next, history_return):
        env = self.env
        n_step_return, importance_sampling_ratio = 0, 1  # TODO: accumulative update
        for i, (hs, ha, hr, im) in enumerate(reversed(history)):
            n_step_return = n_step_return * self.gamma + hr
            if i < self.n_step - 1:
                importance_sampling_ratio *= im     # one step later shifts to match q[s_next, a_next]
        
        importance_sampling_ratio *= im_next
            
        s0, a0, r1, im0 = history[0]    # q[s, a] to be updated
        logger.debug(f'Updating s0={env.state_to_xy(s0)}, {a0=} with {history=}, G={n_step_return}')
        
        self.q[s0, a0] += self.alpha * importance_sampling_ratio *\
                (n_step_return + self.gamma ** self.n_step * self.q[s_next, a_next] - self.q[s0, a0])
        
        
    def policy_iteration(self, n_episode, record_return=False, log_period=100):
        """
        Currently running on-policy control.
        """
        env = self.env
        
        eps_returns = []
        for i_eps in range(n_episode):
            # initialize starting state
            s = env.reset()
            x, y = env.state_to_xy(s)
            if self.random_s0:
                s = np.random.randint(0, self.n_state)
                x, y = env.state_to_xy(s)
                while env.is_end_state(x, y) or (x, y, 1) in env.types:
                    s = np.random.randint(0, self.n_state)
                    x, y = env.state_to_xy(s)
            
            env.state = s
            
            # running episode
            a, p = self.take_behavioural_action(s)
            im = self._get_target_action_prob(s, a) / p     # importance of (a, s)
            
            done = False
            history, history_return, last_reward = [], 0, 0
            eps_return = 0       
            
            logger.debug(f'Starting episode {i_eps}/{n_episode} at {(x, y)=}, {a=}')
            
            while not done:
                assert not self.on_policy or im == 1
                s_next, reward, done, _ = env.step(a)
                a_next, p_next = self.take_behavioural_action(s_next)
                im_next = self._get_target_action_prob(s_next, a_next) / p_next
                history_return = (history_return - last_reward * (len(history) >= self.n_step)) / self.gamma + \
                    reward * self.gamma ** min((self.n_step - 1), len(history))
                history.append((s, a, reward, im))
                
                eps_return += reward    # no discount for recording
                logger.debug(f'From (x, y)={env.state_to_xy(s)}, {a=} to {env.state_to_xy(s_next)}')
                
                # update q if we have n steps
                if len(history) >= self.n_step:
                    self._policy_evaluate_and_improve(history, s_next, a_next, im_next)
                    history.pop(0)
                    
                s, a, im, p, last_reward = s_next, a_next, im_next, p_next, reward
                
            if len(history) > 0 and not self.drop_remainder:
                # then the all (s, a) in remaining history are updated, used in Monte-Carlo
                # that is: use as large k-step to update as possible, 1 <= k < n
                # TD target = n-step reward + discounted q[s_n+1,a_n+1], while after k-step, reward and q are 0
                logger.debug('Updating remaining episode steps')
                n_step_return, importance_sampling_ratio = 0, 1
                for s, a, r, im in reversed(history):
                    n_step_return  = n_step_return * self.gamma + r
                    self.q[s, a] += self.alpha * importance_sampling_ratio * (n_step_return - self.q[s, a])
                    importance_sampling_ratio *= im
                    logger.debug(f'Updating s={env.state_to_xy(s)}, {a=} with G={n_step_return}')
                    
            eps_returns.append(eps_return)
            if i_eps % log_period == 0 and i_eps > 0:
                logger.info(
                    f'Episode {i_eps-log_period}-{i_eps} mean return: {np.mean(eps_returns[-log_period:]):.5f}'
                )
                
        if record_return:
            return np.array(eps_returns)
                    
                    
    @property
    def v(self):
        """
        Compute value from q by:
        V(s) = sum for all act a( pi(a|s) * q[s, a] )
        """
        # v = np.zeros(self.n_state)
        return np.max(self.q, axis=-1) # by greedy policy
        
    @property
    def pi(self):
        """
        Derive a greedy policy from q. If on-policy, an epsilon-greedy pi is returned, 
        otherwise a deterministic one is return.
        """
        
        if self.on_policy:
            non_opt_prob = self.epsilon / self.n_action
            if False:
                t_pi = np.zeros((self.n_state, self.n_action))
                for s in range(self.n_state):
                    if not self.env.is_end_state(s) and (*self.env.state_to_xy(s), 1) not in self.env.types:
                        max_q = self.q[s].max()
                        cnt_max = np.sum(self.q[s] == max_q)
                        t_pi[s] = ((self.q[s] == max_q) * (1-self.epsilon) / cnt_max) + non_opt_prob
            
            # faster numpy array operation
            t = (self.q == np.max(self.q, axis=-1, keepdims=True))
            return t * (1-self.epsilon) / np.sum(t, axis=-1, keepdims=True) + non_opt_prob
            
        else:
            t = self.q == np.max(self.q, axis=-1, keepdims=True)
            return (t + 0) / np.sum(t, axis=-1, keepdims=True)
            
        
        
                    
                    
                
            

