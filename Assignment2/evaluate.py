import torch
import numpy as np
import matplotlib.pyplot as plt
from pettingzoo.mpe import simple_spread_v3
from time import time
from pathlib import Path
import pygame
import pygame.camera

from algorithms.MADDPGTrainer import MADDPGTrainer
from utils.env_helper import get_simple_env

pygame.init()
pygame.camera.init()


if __name__ == '__main__':
    save_path = Path(r'SYSU_2023SpringRL\Assignment2\saves')
    model_path = save_path / Path('avg=-8.02203-i_episode=100.pt')

    num_episodes = 100
    hidden_dim = 64
    episode_length = 25
    
    env = simple_spread_v3.env(max_cycles=episode_length, render_mode='human')
    env.reset()
    simple_env = get_simple_env(env)
    
    state_dim, action_dim = env.state_space.shape[0], env.action_spaces['agent_0'].n

    trainer = MADDPGTrainer(
        env, hidden_dim=hidden_dim
    )

    state_dict = torch.load(model_path)
    print(f'Loaded model: i_episode={state_dict["i_episode"]}')

    trainer.load_state_dict(state_dict['trainer'])

    return_list = []

    for i_episode in range(num_episodes):
        episode_step = episode_return = 0
        env.reset()
        states = env.state()
        actions = [None for _ in range(env.num_agents)]

        for _i, agent in enumerate(env.agent_iter()):
            episode_step, agent_id = _i // env.num_agents, _i % env.num_agents
            _o, _cr, _terminated, _truncated, _info = env.last()
            if _terminated or _truncated:
                break

            # take actions
            obs = env.observe(agent)
            with torch.no_grad():
                action = trainer.take_action(agent_id, obs, explore=False)

            env.step(action)

            if agent_id == env.num_agents - 1:
                # a global step finished
                rewards = [env.rewards[a] for a in env.agents]
                episode_return += sum(rewards)
                pygame.image.save(simple_env.screen, save_path / 'imgs/screenshot.png')

            pass

        episode_return *= 2 / episode_length
        return_list.append(episode_return)
        print(f'Episode {i_episode}: return {episode_return}')

        pass

    print(f'Average return in {len(return_list)} episodes: {np.mean(return_list)}')

    


