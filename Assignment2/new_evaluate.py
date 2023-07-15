import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from time import time
import os
import argparse
import matplotlib.pyplot as plt
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

os.environ['SUPPRESS_MA_PROMPT'] = "1"

from algorithms.NewMADDPG import NewMADDPG
from utils.make_env import make_env
from utils.plots import plot_return_trace_area


if '__main__' == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", default="simple_spread", type=str)
    parser.add_argument("--n_episodes", default=100, type=int)
    parser.add_argument("--episode_length", default=25, type=int)
    config = parser.parse_args()


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ######################################
    # making environment
    ######################################
    episode_length = config.episode_length
    env = make_env('simple_spread')
    env.reset()
    assert not env.discrete_action_input

    obs_dims = [space.shape[0] for space in env.observation_space]
    state_dim = sum(obs_dims)
    action_dims = [space.n for space in env.action_space]
    total_action_dim = sum(action_dims)
    n_agent = env.n

    ######################################
    # loading model
    ######################################
    submission = True

    if submission:
        """
        ! 评估生成的视频输出在 'agents/maddpg/submission_output_dir' 中。
        评估模型在 agents/maddpg/submission_model.pt，是包含 3 个 agents 的单个文件，大小约 1MB。
        """

        cur_file_path = Path(__file__)
        save_path = cur_file_path.parent / 'agents/maddpg/'

        # load model from agents/maddpg
        model_save_path = save_path / 'submission_model.pt'
        image_save_path = save_path / 'submission_output_dir/'
        image_save_path.mkdir(exist_ok=True)
    else:

        save_path = Path('SYSU_2023SpringRL/Assignment2/saves/new')
        save_path /= 'n_eps=20000-eps_len=25-a_lr=0.0001-c_lr=0.001'
        model_save_path = save_path / 'models'
        model_save_path /= 'episode=19600-ret_per_step=-4.105-2023-07-14 01h05m52s.pt'
        image_save_path = save_path / 'imgs'


    for v in image_save_path.glob('model_eval*.mp4'):
        v.unlink()

    loaded_model = torch.load(model_save_path)
    args = loaded_model['args']

    trainer = NewMADDPG(env, args).load_state_dict(loaded_model['trainer'])


    ########################################
    # evaluating
    ########################################
    n_episode = config.n_episodes
    returns = []
    cost = 0
    for i_episode in tqdm(range(n_episode)):
        states = env.reset()
        eps_return = 0

        frames = []
        generate_video = i_episode % (n_episode // 20) == 0

        for time_step in range(episode_length):
            start = time()
            with torch.no_grad():
                actions = [
                    a.take_single_action(obs, noise_rate=0, epsilon=0) for obs, a in zip(states, trainer.agents)
                ]
            next_states, rewards, done, info = env.step(actions)
            states = next_states
            eps_return += rewards[0]
            cost += time() - start  # includes action, step() time

            if generate_video:
                frames.append(env.render(mode='rgb_array')[0])
        
        returns.append(eps_return / episode_length)

        if generate_video:
            clip = ImageSequenceClip(frames, fps=10)
            clip.write_videofile(str(image_save_path / f'model_eval-eps={i_episode}.mp4'), logger=None)
    
    avg_eval_returns = np.mean(returns)
    print(
        f'Averate return per step {avg_eval_returns:.5f}\n'
        f'Time cost per step {cost / (n_episode * episode_length) * 1e3} ms'
        
    )

    plot_return_trace_area([returns], ['MADDPG'])

    plt.savefig(image_save_path / 'eval_returns.png')
    plt.savefig(image_save_path / 'eval_returns.pdf')
    





