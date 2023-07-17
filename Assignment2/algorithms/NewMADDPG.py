import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

from algorithms.actor_critic import Actor, Critic

from multiagent.environment import MultiAgentEnv
from utils.ReplayBuffer import Experience, ReplayBuffer, PERBuffer, ReloPERBuffer

from typing import List, Tuple


class DDPGAgent:
    def __init__(self, args:dict={}, agent_id=None) -> None:

        if not args.get('empty_construct', False):
            self._construct_agent(args, agent_id)
        
        pass


    def _construct_agent(self, args: dict, agent_id):
        self.agent_id = agent_id
        self.args = args.copy()

        self.obs_dim = args['obs_dims'][agent_id]
        self.state_dim = args['state_dim']
        self.action_dim = args['action_dims'][agent_id]
        self.total_action_dim = args.get('total_action_dim', sum(args['action_dims']))
        
        self.high_act = args['high_act']
        self.device = args['device']

        # hidden_dim = args.get('hidden_dim', 64)
        # n_layer = args.get('n_layer', 4)

        self.actor = Actor(self.obs_dim, self.action_dim, self.high_act).to(self.device)
        self.target_actor = Actor(self.obs_dim, self.action_dim, self.high_act).to(self.device)

        self.critic = Critic(self.state_dim, self.total_action_dim, self.high_act).to(self.device)
        self.target_critic = Critic(self.state_dim, self.total_action_dim, self.high_act).to(self.device)

        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=args['actor_lr'])
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=args['critic_lr'])


    def take_single_action(self, obs: np.ndarray, noise_rate, epsilon) -> np.ndarray:
        if np.random.uniform() < epsilon:
            action = np.random.uniform(-self.high_act, self.high_act, self.action_dim)
        else:
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
            action = self.actor(obs).cpu().numpy()
            noise = noise_rate * self.high_act * np.random.randn(*action.shape)
            action += noise
            action = np.clip(action, -self.high_act, self.high_act)

        return action
    
    def soft_update(self):
        tau = self.args['tau']
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


    def state_dict(self) -> dict:
        t = {
            'actor': self.actor.state_dict(), 'target_actor': self.target_actor.state_dict(),
            'critic': self.critic.state_dict(), 'target_critic': self.target_critic.state_dict(),
            'actor_optim': self.actor_optim.state_dict(), 'critic_optim': self.critic_optim.state_dict(),
        }

        for k, v in self.__dict__.items():
            if k not in t:
                t[k] = v

        return t
    
    def load_state_dict(self, state_dict:dict):
        self._construct_agent(state_dict['args'], state_dict['agent_id'])
        reserved_attrs = set(['actor', 'critic', 'target_actor', 'target_critic', 'actor_optim', 'critic_optim'])
        for k in reserved_attrs:
            getattr(self, k).load_state_dict(state_dict[k])
        
        for k, v in state_dict.items():
            if k not in reserved_attrs:
                setattr(self, k, v)

        return self



class NewMADDPG:

    def __init__(self, env: MultiAgentEnv, args:dict) -> None:
        self.env = env
        if not args.get('empty_construct'):
            self._construct(env, args)

        pass


    def _construct(self, env: MultiAgentEnv, args:dict):
        self.env = env
        self.n_agents = env.n

        self.args = args

        self.device = args['device']

        self.action_dims = args['action_dims']
        self.obs_dims = args['obs_dims']
        self.state_dim = sum(self.obs_dims)

        self.gamma = args.get('gamma', 0.95)
        self.trained_step = 0

        self.agents: List[DDPGAgent] = [
            DDPGAgent(args, agent_id)
            for agent_id in range(self.n_agents)
        ]

        self.prioritized_replay = args.get('prioritized_replay', False)
        self.relo = args.get('relo', False)
        if self.relo:
            self.prioritized_replay = False  

        mem_cls = PERBuffer if self.prioritized_replay else ReplayBuffer
        mem_cls = ReloPERBuffer if self.relo else mem_cls

        # TODO: hybrid action spaces, obs spaces
        state_shape = (self.n_agents, self.obs_dims[0])
        act_shape = (self.n_agents, self.action_dims[0])
        rew_shape = done_shape = (self.n_agents,)
        self.buffer = mem_cls(
            state_shape=state_shape, act_shape=act_shape, rew_shape=rew_shape, done_shape=done_shape,
        )
        pass


    def store_experience(self, states, actions, rewards, next_states, done):
        self.buffer.store(states, actions, rewards, next_states, done)
        pass


    def sample_experience(self, batch_size):
        return self.buffer.sample(batch_size)
    

    def train(self, sample:Experience|Tuple[Experience, np.ndarray, np.ndarray], agent_id):
        device = self.device
        if self.prioritized_replay or self.relo:
            sample, is_weights, batch_idx = sample
            is_weights = torch.tensor(is_weights, device=device).view([-1, 1])

        states = torch.tensor(sample.state, device=device) # shape of (batch_sz, n_agent, obs_dim)
        actions = torch.tensor(sample.act, device=device)  # shape of (batch_sz, n_agent, act_dim)
        rewards = torch.tensor(sample.reward[:, [agent_id]], device=device)  # only take this agent's 
        next_states = torch.tensor(sample.next_state, device=device)
        dones = torch.tensor(sample.done, device=device)

        split_actions = list(torch.split(actions, 1, dim=1))  # split by agent: n_agent * (batch_sz, 1, act_dim)
        observations = list(torch.split(states, 1, dim=1))    # split by agent, too
        next_observations = list(torch.split(next_states, 1, dim=1))

        for i in range(self.n_agents):
            split_actions[i] = torch.squeeze(split_actions[i])
            observations[i] = torch.squeeze(observations[i])
            next_observations[i] = torch.squeeze(next_observations[i])

        cur_agent = self.agents[agent_id]

        # build critic loss
        next_actions = []
        with torch.no_grad():
            for aid, agent in enumerate(self.agents):
                next_actions.append(agent.target_actor(observations[aid]))

            q_next = cur_agent.target_critic(next_observations, next_actions).detach()
            target_q = (self.gamma * q_next + rewards).detach()

        q = cur_agent.critic(observations, split_actions)
        critic_loss = torch.pow(target_q - q, 2)
        if self.prioritized_replay:
            td_errors = critic_loss.detach().cpu().numpy()
            critic_loss *= is_weights

        critic_loss = torch.mean(critic_loss)

        # build actor loss
        split_actions[agent_id] = cur_agent.actor(observations[agent_id])
        actor_loss = -1.0 * cur_agent.critic(observations, split_actions).mean()

        cur_agent.actor_optim.zero_grad()
        actor_loss.backward()
        cur_agent.actor_optim.step()

        cur_agent.critic_optim.zero_grad()
        critic_loss.backward()
        cur_agent.critic_optim.step()

        # after critic learns this batch, compare reducible loss
        if self.relo:
            q_cur_target = cur_agent.target_critic(observations, split_actions).detach()
            target_critic_loss = torch.pow(target_q - q_cur_target, 2)
            relo = (critic_loss.detach() - target_critic_loss).cpu().numpy() # new_loss - old_loss
            pass

        cur_agent.soft_update()
        self.trained_step += 1

        if self.prioritized_replay:
            self.buffer.update_batch(batch_idx, td_errors)
        elif self.relo:
            self.buffer.update_batch(batch_idx, relo)

        return actor_loss.item(), critic_loss.item()
        


    def state_dict(self) -> dict: 
        t = {
            'agents': [a.state_dict() for a in self.agents],
        }
        reserved_attrs = ['env', 'buffer'] + list(t.keys()) # clear buffer TODO
        for k, v in self.__dict__.items():
            if k not in reserved_attrs:
                t[k] = v

        return t
    

    def load_state_dict(self, state_dict:dict):
        reserved_attrs = ['agents', 'env', 'buffer']
        for k, v in state_dict.items():
            if k not in reserved_attrs:
                setattr(self, k, v)
        
        self.args['empty_construct'] = True
        self.agents = [
            DDPGAgent(self.args, i).load_state_dict(state_dict['agents'][i]) for i in range(self.n_agents)
        ]
        self.args['empty_construct'] = False
        return self


    def enter_train(self):
        for a in self.agents:
            a.actor.train()
            a.target_actor.train()
            a.critic.train()
            a.target_critic.train()


    def enter_eval(self):
        for a in self.agents:
            a.actor.eval()
            a.target_actor.eval()
            a.critic.eval()
            a.target_critic.eval()




