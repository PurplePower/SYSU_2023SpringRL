import torch
import numpy as np
import matplotlib.pyplot as plt
from pettingzoo.mpe import simple_spread_v3
from time import time
from pathlib import Path
import pygame
from  moviepy.video.io.ImageSequenceClip import ImageSequenceClip

import os
os.environ['SUPPRESS_MA_PROMPT'] = "1"

from algorithms.MADDPGTrainer import MADDPGTrainer
from utils.env_helper import get_simple_env
from utils.env_helper import PettingZooWrapperEnv

pygame.init()

def empty_dir(path: Path):
    for file in path.glob("*"):
        if file.is_file():
            file.unlink()





if __name__ == '__main__':
    save_path = Path(r'SYSU_2023SpringRL\Assignment2\saves')
    model_path = save_path / Path(
        # r'models\pz\n_eps=10000-eps_len=25-hidden_dim=64-upd_int=100-a_lr=0.01-c_lr=0.01-lr_decay=0.9997122182804811-local_ratio=0.2-cont_act=False' +\
        r'models\pz\\n_eps=10000-eps_len=25-hidden_dim=64-upd_int=10-a_lr=0.0001-c_lr=0.001-lr_decay=1-local_ratio=0.5-cont_act=True' +\
        r'\avg=-6.18755-i_episode=9900-2023-07-11 21h50m58s.pt' )
    video_save_path = save_path / 'videos'
    video_save_path.mkdir(exist_ok=True)
    empty_dir(video_save_path)

    num_episodes = 100
    hidden_dim = 64
    episode_length = 25
    
    env = simple_spread_v3.env(
        max_cycles=episode_length, render_mode='human', continuous_actions=True)
    env.reset()
    simple_env = get_simple_env(env)
    env = PettingZooWrapperEnv(env)
    
    state_dim, action_dim = env.state_dim, env.action_dim

    state_dict = torch.load(model_path)
    print(f'Loaded model: i_episode={state_dict["i_episode"]}')


    trainer = MADDPGTrainer(
        env, hidden_dim=hidden_dim, n_layer=4,
    )

    trainer.load_state_dict(state_dict['trainer'])

    

    return_list = []

    for i_episode in range(num_episodes):
        episode_step = episode_return = 0
        states = env.reset()
        
        generate_video = i_episode % (num_episodes // 10) == 0
        frames = []

        for episode_step in range(episode_length):
            with torch.no_grad():
                actions = [
                    trainer.take_action(_i, s, explore=False) for _i, s in enumerate(states)
                ]

            next_states, rewards, done, infos = env.step(actions)
            states = next_states
            episode_return += np.sum(rewards)

            if generate_video:
                frames.append(pygame.surfarray.array3d(simple_env.screen))


        episode_return *= 2 / episode_length
        return_list.append(episode_return)
        print(f'Episode {i_episode}: return {episode_return}')

        if generate_video:
            clip = ImageSequenceClip(frames, fps=10.0)
            clip.write_videofile(str(video_save_path / f'episode {i_episode}.mp4'))

        pass

    print(f'Average return in {len(return_list)} episodes: {np.mean(return_list)}')

    


