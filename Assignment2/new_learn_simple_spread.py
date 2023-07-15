import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
import shutil
import os
from time import time
from datetime import datetime
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

os.environ['SUPPRESS_MA_PROMPT'] = "1"

from algorithms.NewMADDPG import NewMADDPG
from utils.make_env import make_env
from utils.ReplayBuffer import ReplayBuffer





if '__main__' == __name__:


    ######################################
    # params for learning
    ######################################
    n_episodes = 20000
    episode_length = 25 # default is 25 for simple spread
    # hidden_dim = 64
    actor_lr = 1e-4
    critic_lr = 1e-3

    gamma = 0.95
    tau = 1e-2
    epsilon = 0.1
    noise_rate = 0.1

    batch_size = 256

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    ######################################
    # making environment
    ######################################
    env = make_env('simple_spread')
    env.reset()
    assert not env.discrete_action_input

    obs_dims = [space.shape[0] for space in env.observation_space]
    state_dim = sum(obs_dims)
    action_dims = [space.n for space in env.action_space]
    total_action_dim = sum(action_dims)
    n_agent = env.n
    
    ######################################
    # saving
    ######################################
    save_path = Path('SYSU_2023SpringRL/Assignment2/saves/new')
    save_path /= f'n_eps={n_episodes}-eps_len={episode_length}-a_lr={actor_lr}-c_lr={critic_lr}'
    save_path.mkdir(exist_ok=True, parents=True)
    model_save_path = save_path / 'models'
    image_save_path = save_path / 'imgs'

    continue_training = False

    if continue_training:
        #! deprecated, buffer is cleared 
        model_to_load = model_save_path / 'episode=9900-ret_per_step=-5.534-2023-07-13 16h39m02s.pt'
        print(f'Loading from {model_to_load.stem}')
    else:
        if model_save_path.exists():
            shutil.rmtree(model_save_path)
        if image_save_path.exists():
            shutil.rmtree(image_save_path)
        model_save_path.mkdir(exist_ok=True, parents=True)
        image_save_path.mkdir(exist_ok=True, parents=True)


    ######################################
    # training
    ######################################
    pretrained_episodes = 0
    if continue_training:
        loaded_models = torch.load(model_to_load)
        args = loaded_models['args']
    else:
        args = {
            'n_agent': n_agent,
            'state_dim': state_dim, 'obs_dims': obs_dims, 
            'action_dims': action_dims, 'total_action_dim': total_action_dim,
            'gamma': gamma, 'tau': tau, 'epsilon': epsilon, 
            'actor_lr': actor_lr, 'critic_lr': critic_lr, 'high_act': 1,
            'device': device,
            'prioritized_replay': True
        }

    # replay_buffer = ReplayBuffer()
    trainer = NewMADDPG(env, args)

    if continue_training:
        trainer.load_state_dict(loaded_models['trainer'])
        pretrained_episodes = 10000

    eval_interval = n_episodes // 100


    def evaluate(i_eps, n_eps=20, save_model=True):
        eval_returns = []
        for i_episode in range(n_eps):
            states = env.reset()
            eps_return = 0

            frames = []
            generate_video = i_episode % (n_eps // 2) == 0

            for time_step in range(episode_length):
                with torch.no_grad():
                    actions = [
                        a.take_single_action(obs, noise_rate=0, epsilon=0) for obs, a in zip(states, trainer.agents)
                    ]
                next_states, rewards, done, info = env.step(actions)
                states = next_states
                eps_return += rewards[0]

                if generate_video:
                    frames.append(env.render(mode='rgb_array')[0])
            
            eval_returns.append(eps_return / episode_length)

            if generate_video:
                clip = ImageSequenceClip(frames, fps=10)
                clip.write_videofile(str(image_save_path / f'eval={i_eps}-eps={i_episode}.mp4'), logger=None)
        
        avg_eval_returns = np.mean(eval_returns)
        if save_model:
            filename = model_save_path / f'episode={i_eps}-ret_per_step={avg_eval_returns:.3f}-{datetime.now().strftime("%Y-%m-%d %Hh%Mm%Ss.pt")}'
            torch.save({'trainer': trainer.state_dict(), 'args': args}, filename)
            print(f'Saved to {filename}')

        return avg_eval_returns



    returns = []

    for i_episode in tqdm(range(n_episodes)):
        states = env.reset()
        eps_return = 0
        for time_step in range(episode_length):
            with torch.no_grad():
                actions = [
                    a.take_single_action(obs, noise_rate, epsilon=epsilon) 
                    for obs, a in zip(states, trainer.agents)
                ]

            next_states, rewards, done, info = env.step(actions)
            trainer.store_experience(states, actions, rewards, next_states, done)
            states = next_states
            eps_return += rewards[0]

            if len(trainer.buffer) >= batch_size:
                batch = trainer.sample_experience(batch_size)

                for agent_id in range(n_agent):
                    _, _ = trainer.train(batch, agent_id)

                noise_rate = max(0.05, noise_rate - 5e-7)
                epsilon = max(0.05, epsilon - 5e-7)

            pass

        if i_episode % eval_interval == 0:
            eval_return = evaluate(i_episode + pretrained_episodes, save_model=True)
            print(f'Evaluated at episode={i_episode+pretrained_episodes}: return/step = {eval_return:.5f}')

        returns.append(eps_return / episode_length)


    env.close()
    np.savez(save_path / 'returns.npz', np.array(returns))
    pass


