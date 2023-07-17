

"""
评估时，将直接使用 algorithms.NewMADDPG 的 class NewMADDPG 。
其能操纵所有 agents 对环境反馈进行动作。 详见 new_evaluate.py，在那里
将对提交的模型进行评估，并计算耗时、生成评估时的样例视频。生成的样例视频
将在`本文件夹`下的 submission_output_dir 中。

NewMADDPG 内包含了所有 agents 及其它训练参数，其直接保存为一个文件，
并可用 NewMADDPG.load_state_dict() 加载。提交的模型是本文件夹下的
submission_model.pt，其使用最简单的 MADDPG 训练 19600 个 episodes.


但为方便使用自动化的脚本进行测试，从 NewMADDPG.py 中复制相关内容到下面。
由于可以将所有 agents 作为整体，令 NewMADDPG 作为这个 agents。

"""

# ! 注意，以下代码应及时从 algorithms/NewMADDPG.py 中获取更新，并添加 act()，删除训练相关内容

import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

# from algorithms.actor_critic import Actor, Critic

from multiagent.environment import MultiAgentEnv
# from utils.ReplayBuffer import Experience, ReplayBuffer, PERBuffer, ReloPERBuffer

from typing import List, Tuple

# borrow from starry-sky6688
# define the actor network
class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, high_action):
        super(Actor, self).__init__()
        self.max_action = high_action
        self.fc1 = nn.Linear(obs_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.action_out = nn.Linear(64, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        actions = self.max_action * torch.tanh(self.action_out(x))

        return actions


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, high_action):
        super(Critic, self).__init__()
        self.max_action = high_action
        self.fc1 = nn.Linear(state_dim + action_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.q_out = nn.Linear(64, 1)

    def forward(self, state:list, action:list):
        state = torch.cat(state, dim=1)
        for i in range(len(action)):
            action[i] /= self.max_action
        action = torch.cat(action, dim=1)
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_value = self.q_out(x)
        return q_value


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

        # mem_cls = PERBuffer if self.prioritized_replay else ReplayBuffer
        # mem_cls = ReloPERBuffer if self.relo else mem_cls

        # TODO: hybrid action spaces, obs spaces
        state_shape = (self.n_agents, self.obs_dims[0])
        act_shape = (self.n_agents, self.action_dims[0])
        rew_shape = done_shape = (self.n_agents,)
        # self.buffer = mem_cls(
        #     state_shape=state_shape, act_shape=act_shape, rew_shape=rew_shape, done_shape=done_shape,
        # )
        pass

        


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


    def act(self, obs):
        with torch.no_grad():
            actions = [
                a.take_single_action(o, noise_rate=0, epsilon=0) 
                for o, a in zip(obs, self.agents)
            ]
        return actions







