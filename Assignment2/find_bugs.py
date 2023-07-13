import torch
import numpy as np
import matplotlib.pyplot as plt
from pettingzoo.mpe import simple_spread_v3
from time import sleep, time
from pathlib import Path

import os
os.environ['SUPPRESS_MA_PROMPT'] = "1"

import pygame
from  moviepy.video.io.ImageSequenceClip import ImageSequenceClip

from imgcat import imgcat

from algorithms.MADDPGTrainer import MADDPGTrainer
from utils.env_helper import PettingZooWrapperEnv, OpenAIWrapperEnv, get_simple_env


def get_obs_componenets(obs, n_agent=3):
    index = 0
    velocity = obs[index:index+2]
    index += len(velocity)

    position = obs[index:index+2]
    index += len(position)

    landmark_rel_pos = []
    for i in range(n_agent):
        landmark_rel_pos.append(obs[index:index+2])
        index += 2

    other_agent_rel_pos = []
    for i in range(n_agent-1):
        other_agent_rel_pos.append(obs[index:index+2])
        index += 2

    communication = obs[index:]

    return velocity, position, landmark_rel_pos, other_agent_rel_pos, communication


if '__main__' == __name__:
    save_path = Path(r'SYSU_2023SpringRL\Assignment2\saves')
    video_save_path = save_path / 'videos/bugs'
    video_save_path.mkdir(parents=True, exist_ok=True)

    action_names = ['nope', 'left', 'right', 'down', 'up']
    episode_length = 25
    env = simple_spread_v3.env(max_cycles=episode_length, continuous_actions=True, render_mode='human')    # args: render_mode, max_cycles
    senv = get_simple_env(env)
    env = PettingZooWrapperEnv(env)
    env.reset()



    for i_episode in range(10):
        env.reset()
        frames = []
        for episode_step in range(episode_length):
            # actions = [1, 0, 0]
            actions = np.array([
                [0, -1, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 1, 0]
            ]) * 0.5

            next_states, rewards, done, infos = env.step(actions)

            # print(f'Eps {i_episode} step {episode_step}: agent_0 moves {action_names[actions[0]]}')

            frames.append(pygame.surfarray.array3d(senv.screen))
            imgcat(frames[-1], width=20, height=20)

            vel, pos, lm_rel_pos, other_rel_pos, cmn = get_obs_componenets(next_states[0])
            print(f'Now agent 0: {vel=}, {pos=}, \nlm_rel={lm_rel_pos}, \nothers={other_rel_pos=}')
            pass


        clip = ImageSequenceClip(frames, fps=10.0)
        clip.write_videofile(str(video_save_path / f'episode {i_episode}.mp4'))








    pass

    









