import argparse
import time
import numpy as np
import torch
import os
os.environ['SUPPRESS_MA_PROMPT'] = "1"
from utils.make_env import make_env

#! 此处加载类
from agents.maddpg.submission import NewMADDPG


def run(config):
    env = make_env(config.env_id)

    # ! 可能需要确保以下加载路径正确
    state_dict = torch.load(r'SYSU_2023SpringRL\Assignment2\agents\maddpg\submission_model.pt')
    args = state_dict['args']
    agents = NewMADDPG(env, args)
    agents.load_state_dict(state_dict['trainer'])


    total_reward = 0.
    for ep_i in range(config.n_episodes):
        print("Episode %i of %i" % (ep_i + 1, config.n_episodes))
        obs = env.reset()
        # env.render('human')
        episode_reward = 0.
        for t_i in range(config.episode_length):
            calc_start = time.time()

            # ! 此处获取动作
            actions = agents.act(obs)

            obs, rewards, dones, infos = env.step(actions)
            episode_reward += np.array(rewards).sum()
            calc_end = time.time()
            elapsed = (calc_end - calc_start) * 1000.0
            # the elapsed should not exceed 10ms per step
            # print("Elapsed %f" % (elapsed))
            # env.render('human')
        total_reward += episode_reward/config.episode_length
        print("Episode reward: %.2f" % (episode_reward/config.episode_length))
    print("Mean reward of %d episodes: %.2f" % (config.n_episodes, total_reward/config.n_episodes))

    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", default="simple_spread", type=str)
    parser.add_argument("--n_episodes", default=100, type=int)
    parser.add_argument("--episode_length", default=25, type=int)
    config = parser.parse_args()

    run(config)
