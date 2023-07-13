import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from pettingzoo.mpe._mpe_utils.simple_env import SimpleEnv

from typing import List

from utils.env_helper import ClassicalEnv

from algorithms.actor_critic import Actor, Critic


# class SimpleModel(torch.nn.Module):
#     def __init__(self, num_in, num_out, hidden_dim,  n_layer=3, device='cuda'):
#         super().__init__()
#         self.device = device
#         self.layers = torch.nn.ModuleList()
#         self.layers.append(nn.Linear(num_in, hidden_dim, device=device))    # first
        
#         for _ in range(n_layer - 2):
#             self.layers.append(nn.Linear(hidden_dim, hidden_dim, device=device))

#         self.layers.append(nn.Linear(hidden_dim, num_out, device=device))

#     def forward(self, x):
#         for layer in self.layers[:-1]:
#             x = F.relu(layer(x))
#         return self.layers[-1](x)
    

# class Actor(SimpleModel):
#     def __init__(self, num_in, num_out, hidden_dim, n_layer=3, device='cuda', env_args:dict={}):
#         super().__init__(num_in, num_out, hidden_dim, n_layer, device)
#         self.high_act = env_args.get('high_act', 1)

#     def forward(self, state):
#         for layer in self.layers[:-1]:
#             state = F.relu(layer(state))
#         return torch.tanh(self.layers[-1](state)) * self.high_act


# class Critic(SimpleModel):
#     def __init__(self, num_in, num_out, hidden_dim, n_layer=3, device='cuda', env_args:dict={}):
#         super().__init__(num_in, num_out, hidden_dim, n_layer, device)
#         self.high_act = env_args.get('high_act', 1)


#     # def forward(self, con_states, con_actions):
#     #     con_actions /= self.high_act
#     #     x = torch.cat([con_states, con_actions], dim=1)

#     #     # do not relu the last layer
#     #     for layer in self.layers[:-1]:
#     #         x = F.relu(layer(x))
#     #     return self.layers[-1](x)

#     def forward(self, x):
#         # do not relu the last layer
#         for layer in self.layers[:-1]:
#             x = F.relu(layer(x))
#         return self.layers[-1](x)



class DDPGAgent:
    """
    A single agent in MADDPG, runs DDPG to learn.

    Each agent has its own actor and critic, by using ideas from double DQN, 
    results in 4 network models.

    Assume using discrete action space.
    
    """

    def __init__(
        self, state_dim=None, action_dim=None, critic_input_dim=None, hidden_dim=64, n_layer=3,
        actor_lr=1e-3, critic_lr=1e-3, epsilon=0.01, #eps_decay=0.99, 
        device='cuda', 
        lr_scheduler=None, lr_scheduler_args=None, continuous_action=False, 
        noise_rate=0.1, env_args:dict={}
    ) -> None:
        """
        state_dim: dimensions of state
        action_dim: dimensions of action, default to discrete actions
        critic_input_dim: dimensions of states and actions from all agents
        
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.critic_input_dim = critic_input_dim
        self.epsilon = epsilon
        # self.eps_decay = eps_decay

        self.device = device
        self.continuous_action = continuous_action
        self.noise_rate = noise_rate

        # networks
        self.actor = self.critic = self.target_actor = self.target_critic = None
        self.actor_optimizer = self.critic_optimizer = None
        self.actor_scheduler = self.critic_scheduler= None

        # self.actor = Actor(state_dim, action_dim, hidden_dim, n_layer, device)
        # self.critic = Critic(critic_input_dim, 1, hidden_dim, n_layer, device)
        self.actor = Actor(state_dim, action_dim, high_action=env_args['high_act']).to(device)
        self.critic = Critic(critic_input_dim-action_dim, action_dim, high_action=env_args['high_act']).to(device)

        # target networks used in generating traces
        self.target_actor = Actor(state_dim, action_dim, high_action=env_args['high_act']).to(device)
        self.target_critic = Critic(critic_input_dim-action_dim, action_dim, high_action=env_args['high_act']).to(device)

        # self.target_actor.load_state_dict(self.actor.state_dict())
        # self.target_critic.load_state_dict(self.critic.state_dict())
        self.soft_update(self.actor, self.target_actor, tau=1)
        self.soft_update(self.critic, self.target_critic, tau=1)

        # optimizers
        self.l2_reg_factor = 0
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), actor_lr, weight_decay=self.l2_reg_factor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), critic_lr, weight_decay=self.l2_reg_factor)
        

        if lr_scheduler is not None:
            self.actor_scheduler = lr_scheduler(optimizer=self.actor_optimizer, **lr_scheduler_args)
            self.critic_scheduler = lr_scheduler(optimizer=self.critic_optimizer, **lr_scheduler_args)

        self.env_args = env_args
        self.high_act = env_args.get('high_act', 1)
        self.low_act = env_args.get('low_act', 0)
        pass


    def state_dict(self) -> dict:
        # models, optimizers and schedulers
        t = {
            'actor': self.actor.state_dict(), 'target_actor': self.target_actor.state_dict(),
            'critic': self.critic.state_dict(), 'target_critic': self.target_critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(), 'critic_optimizer': self.critic_optimizer.state_dict(),
        }
        if self.actor_scheduler is not None:
            t['actor_scheduler'] = self.actor_scheduler.state_dict()
            t['critic_scheduler'] = self.critic_scheduler.state_dict()

        # others
        for k, v in self.__dict__.items():
            if k not in t:
                t[k] = v
        
        return t

        
    def load_state_dict(self, state_dict:dict):
        torch_model_keys = set(['actor', 'critic', 'target_actor', 'target_critic', 'actor_optimizer', 'critic_optimizer'])
        for k in torch_model_keys:
            getattr(self, k).load_state_dict(state_dict[k])
            # self.__dict__[k].load_state_dict(state_dict[k])

        if 'actor_scheduler' in state_dict:
            self.actor_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.actor_optimizer, 1)
            self.critic_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.critic_optimizer, 1)
            self.actor_scheduler.load_state_dict(state_dict['actor_scheduler'])
            self.critic_scheduler.load_state_dict(state_dict['critic_scheduler'])

        for k, v in state_dict.items():
            if k not in torch_model_keys and 'scheduler' not in k:
                setattr(self, k, v)
        
        return self


    def take_discrete_actions(self, obs: np.ndarray, explore=False) -> torch.Tensor:
        """
        Take an action. The output probability from actor is cast to 
        discrete action by Gumbel Softmax

        obs: np.ndarray with shape (batch_sz, self.state_dim(of single agent))
        """
        assert 0
        batch_size = obs.shape[0]
        assert obs.shape[1] == self.state_dim, "State dimensions not match"

        logits = self.actor(obs)  # of shape (batch_sz, action_dim)
        if explore:
            # explore by epsilon
            best_actions = (logits == torch.max(logits, axis=1, keepdim=True)[0]).float()
            random_actions = torch.eye(self.action_dim, requires_grad=False)
            random_actions = random_actions[np.random.choice(self.action_dim, size=batch_size)].to(self.device)
            actions = torch.where(
                torch.rand(batch_size, device=self.device) < self.epsilon, random_actions, best_actions)
        else:
            # sample from given logits
            actions = F.gumbel_softmax(logits, tau=0.5, hard=True)

        assert actions.shape[0] == batch_size
        return actions
    

    def take_continuous_actions(self, obs: np.ndarray, explore=False) -> np.ndarray:
        batch_size = obs.shape[0]
        assert obs.shape[1] == self.state_dim, "State dimensions not match"

        logits = self.actor(obs).detach().cpu().numpy()
        if explore:
            # explore by epsilon-greedy
            choose = np.random.uniform(size=(batch_size, 1)) < self.epsilon
            rand_act = np.random.uniform(self.low_act, self.high_act, logits.shape)
            actions = logits * (~choose) + rand_act * choose

            # add some noise, too
            noise = self.noise_rate * self.high_act * np.random.randn(*logits.shape)
            actions += noise
        else:
            actions = logits

        actions = np.squeeze(np.clip(actions, self.low_act, self.high_act), axis=0)
        return actions.copy()

    def take_action(self, obs: np.ndarray, explore=False) -> np.ndarray:
        if self.continuous_action:
            return self.take_continuous_actions(obs, explore)
        else:
            return self.take_discrete_actions(obs, explore)
        
    
    def soft_update(self, net:torch.nn.Module, target_net:torch.nn.Module, tau):
        """
        Update target network slowly towards network.
        """
        assert tau >= 0 and tau <= 1
        for target_param, param in zip(target_net.parameters(), net.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)




class MADDPGTrainer:
    """
    MADDPG trainer class. Given environment and agent parameters, MADDPG constructs and trains
    all agents. The outer-most training loop is left to user for flexibility.
    
    """

    def __init__(
        self, env:ClassicalEnv, actor_lr=1e-3, critic_lr=1e-3, hidden_dim=64, n_layer=3,
        # state_dim=None, action_dim=None, 
        gamma=0.99, tau=0.01, epsilon=0.01, eps_decay=0.998, eps_decrease=5e-7, device='cuda', 
        lr_scheduler=None, lr_scheduler_args=None, noise_decrease=5e-7, env_args={}
    ) -> None:
        
        self.env = env
        self.actor_lr, self.critic_lr = actor_lr, critic_lr
        self.state_dim, self.action_dim = env.state_dim, env.action_dim
        self.state_dim_per_agent = self.state_dim // env.num_agents
        self.critic_criterion = torch.nn.MSELoss()
        self.hidden_dim = hidden_dim

        self.discrete_action = env.discrete_action

        self.gamma = gamma
        self.tau = tau
        self.epsilon = epsilon
        self.eps_decrease = eps_decrease
        self.device = device

        self.noise_decrease = noise_decrease

        self.reward_scaling_factor = 1

        # sum of states and actions from all agents
        # self.critic_input_dim = 0
        # for act_s in env.action_spaces.values():
        #     self.critic_input_dim += act_s.n

        # self.critic_input_dim += env.state_space.shape[0]

        self.critic_input_dim = env.action_dim * env.num_agents + env.state_dim


        self.agents: List[DDPGAgent] = [
            DDPGAgent(
                self.state_dim_per_agent, self.action_dim, 
                self.critic_input_dim, hidden_dim, n_layer, 
                actor_lr=actor_lr, critic_lr=critic_lr, 
                epsilon=epsilon, # eps_decay=eps_decay,
                device=device,
                lr_scheduler=lr_scheduler, lr_scheduler_args=lr_scheduler_args,
                continuous_action=not self.discrete_action,
                env_args=env_args
            )
            for _ in range(env.num_agents)
        ]
        
        self.env_args = env_args
        pass


    def take_action(self, agent_id:int, observation: np.ndarray, explore) -> np.ndarray:
        """
        Take action for one agent.
        """
        if len(observation.shape) == 1:
            assert observation.shape[0] == self.state_dim_per_agent
            observation = observation.reshape((1, -1))  # batch_size=1
        else:
            assert len(observation.shape) == 2 and observation.shape[1] == self.state_dim_per_agent

        observation = torch.from_numpy(observation).to(device=self.device)
        action = self.agents[agent_id].take_action(observation, explore)
        if self.discrete_action:
            action = np.argmax(action)    # to action space
        return action.astype(np.float32)
    

    @property
    def target_policies(self):
        return [agent.target_actor for agent in self.agents]
    

    @property
    def policies(self):
        return [agent.actor for agent in self.agents]
    

    def update_new(self, sample, agent_id: int):
        states, actions, next_states, rewards, done = sample
        batch_size = states.shape[0]
        # actions of shape (batch_sz, n_agent, action_dim)

        states = torch.split(states, self.state_dim_per_agent, dim=1)
        next_states = torch.split(next_states, self.state_dim_per_agent, dim=1)
        rewards = torch.split(rewards, 1, dim=1)[agent_id]  # only needs reward for this agent

        cur_agent = self.agents[agent_id]

        # critic loss
        actions_next = []
        with torch.no_grad():
            for _aid in range(self.env.num_agents):
                actions_next.append(self.agents[_aid].target_actor(next_states[_aid]))

            # q_next = cur_agent.target_critic(torch.cat([*next_states, *actions_next], dim=1)).detach()
            q_next = cur_agent.target_critic(next_states, actions_next).detach()
            target_q = (rewards + self.gamma * q_next).detach()

        # q_value = cur_agent.critic(torch.cat([*states, actions.view([batch_size, -1])], dim=1))
        q_value = cur_agent.critic(states, list(torch.split(actions.view([batch_size, -1]), self.action_dim, dim=1)))
        critic_loss = (target_q - q_value).pow(2).mean()

        # actor loss
        #TODO
        acts_for_actor = []
        for _aid in range(len(self.agents)):
            if _aid == agent_id:
                acts_for_actor.append(cur_agent.actor(states[_aid]))
            else:
                # acts_for_actor.append(actions[:, _aid, :])
                acts_for_actor.append(self.agents[_aid].actor(states[_aid]))

        actor_loss = - cur_agent.critic(states, acts_for_actor).mean()

        cur_agent.actor_optimizer.zero_grad()
        actor_loss.backward()
        cur_agent.actor_optimizer.step()

        cur_agent.critic_optimizer.zero_grad()
        critic_loss.backward()
        cur_agent.critic_optimizer.step()

        self.update_all_targets()

        for a in self.agents:
            # a.epsilon *= a.eps_decay
            a.epsilon -= self.eps_decrease
            a.epsilon = max(a.epsilon, 0.05)
            a.noise_rate -= self.noise_decrease
            a.noise_rate = max(a.noise_rate, 0.05)

        pass


    def update(self, sample, agent_id:int):
        """
        Upon entering, all agents have taken an action and an episode step
        is performed with next_states and rewards, by which we are updating the `agent_id`-th 
        agent now.
        """

        # process data to match shapes
        states, actions, next_states, rewards, done = sample
        batch_size = states.shape[0]

        states = torch.split(states, self.state_dim_per_agent, dim=1)
        next_states = torch.split(next_states, self.state_dim_per_agent, dim=1)
        if self.discrete_action:
            actions = F.one_hot(actions, num_classes=self.action_dim)   # (batch_sz, n_agent, action_dim)
            actions_non_con = torch.swapaxes(actions, 0, 1)
            actions = torch.reshape(actions, (batch_size, -1))
        else:
            # actions of shape (batch_sz, n_agent, action_dim)
            actions_non_con = torch.swapaxes(actions, 0, 1)
            actions = actions.view((batch_size, -1))

        cur_agent = self.agents[agent_id]


        #########################
        # build critic loss

        with torch.no_grad():
            next_actions = [
                policy(next_state)
                for policy, next_state in zip(self.target_policies, next_states)
            ]
            target_critic_input = torch.cat((*next_states, *next_actions), dim=1)
            target_critic_value = rewards[:, agent_id].view((-1, 1)) * self.reward_scaling_factor\
                + self.gamma * cur_agent.target_critic(target_critic_input).detach() * (1 - done)

        critic_input = torch.cat((*states, actions), dim=1)
        critic_value = cur_agent.critic(critic_input)

        # since now we update the learning network, not the target, target_critic_value has no gradients
        # thus detach it from current graph
        # critic_loss = self.critic_criterion(critic_value, target_critic_value)
        critic_loss = (target_critic_value - critic_value).pow(2).mean()

        ###########################
        # build actor loss: to maximize Q value

        cur_actor_out = cur_agent.actor(states[agent_id])
        if self.discrete_action:
            cur_critic_in = F.gumbel_softmax(cur_actor_out, tau=0.5) # may lower tau
        else:
            cur_critic_in = cur_actor_out
        all_actor_actions = []
        for i, (pi, _state) in enumerate(zip(self.policies, states)):
            if i == agent_id:
                all_actor_actions.append(cur_critic_in)
            else:
                # all_actor_actions.append(pi(_state))    # TODO add random act by eps
                all_actor_actions.append(actions_non_con[i])

        critic_in_act = torch.cat((*states, *all_actor_actions), dim=1)
        actor_loss = - cur_agent.critic(critic_in_act).mean()
        # actor_loss += (cur_actor_out ** 2).mean() * 1e-3    #? some sort of regularization?
        # TODO actor_loss add entropy

        # optimizer step
        cur_agent.critic_optimizer.zero_grad()
        critic_loss.backward()
        cur_agent.critic_optimizer.step() 

        cur_agent.actor_optimizer.zero_grad()
        actor_loss.backward() 
        cur_agent.actor_optimizer.step()

        if cur_agent.actor_scheduler is not None:
            cur_agent.actor_scheduler.step()
            cur_agent.critic_scheduler.step()

        for a in self.agents:
            # a.epsilon *= a.eps_decay
            a.epsilon -= self.eps_decrease
            a.epsilon = max(a.epsilon, 0.05)
            a.noise_rate -= self.noise_decrease
            a.noise_rate = max(a.noise_rate, 0.05)
            

        pass


    def update_all_targets(self):
        for agent in self.agents:
            agent.soft_update(agent.actor, agent.target_actor, self.tau)
            agent.soft_update(agent.critic, agent.target_critic, self.tau)



    def state_dict(self) -> dict:
        t = {
            'agents': [ a.state_dict() for a in self.agents ],
            'critic_criterion': self.critic_criterion.state_dict(),
        }
        for k, v in self.__dict__.items():
            if k not in t and k != 'env':
                t[k] = v
        
        return t
    

    def load_state_dict(self, state_dict: dict):
        preserved_members = ['agents', 'critic_criterion', 'env']
        for k, v in state_dict.items():
            if k not in preserved_members:
                setattr(self, k, v)
        self.agents = [
            DDPGAgent(
                self.state_dim_per_agent, self.action_dim, self.critic_input_dim, 
                self.hidden_dim
            ).load_state_dict(state_dict['agents'][i]) 
            for i in range(self.env.num_agents)
        ]
        self.critic_criterion.load_state_dict(state_dict['critic_criterion'])

        return self

    pass



