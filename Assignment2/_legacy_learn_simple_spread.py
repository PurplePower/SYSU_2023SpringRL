import torch
# from torchrl.data import ReplayBuffer, ListStorage
import numpy as np
import matplotlib.pyplot as plt
from pettingzoo.mpe import simple_spread_v3
from time import sleep, time
from datetime import datetime
from pathlib import Path
import shutil
from tqdm import tqdm
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip


import os
os.environ['SUPPRESS_MA_PROMPT'] = "1"

from algorithms.MADDPGTrainer import MADDPGTrainer
from utils.env_helper import PettingZooWrapperEnv, OpenAIWrapperEnv
from utils.make_env import make_env


if __name__ == '__main__':

    ######################################
    # params for learning
    ######################################
    num_episodes = 10000
    episode_length = 25 # default is 25 for simple spread
    hidden_dim = 64
    actor_lr = 1e-4
    critic_lr = 1e-3
    # lr_decay = (1e-3 / actor_lr) ** (1 / 8000)
    lr_decay = 1

    gamma = 0.95    # reward decay rate
    tau = 1e-2 
    epsilon, eps_decay = 0.1, (1 / 4) ** (1 / 8000)
    batch_size, start_train_size = 256, 256
    max_buffer_size = 1e6


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    update_interval = 1 
    soft_update_per_update = 1

    eval_interval = num_episodes // 100


    ######################################
    # making environment
    ######################################
    use_pettingzoo = False
    continuous_actions = True
    if use_pettingzoo:
        continuous_actions = True
    local_ratio = 0.5
    env_args = {'continuous_act': continuous_actions,}
    
    if use_pettingzoo:
        env_args.update({
            'high_act': 1, 'low_act': 0, 
        })
        env = simple_spread_v3.env(
            max_cycles=episode_length, continuous_actions=continuous_actions, local_ratio=local_ratio,
            render_mode='rgb_array'
        )
        env = PettingZooWrapperEnv(env)
        env.reset()
        
    else:
        continuous_actions = True
        env_args.update({
            'high_act': 1, 'low_act': -1,
        })
        env = make_env('simple_spread', benchmark=True)
        env = OpenAIWrapperEnv(env)
        env.reset()
        

    state_dim, action_dim = env.state_dim, env.action_dim

    ######################################
    # saving
    ######################################

    save_path = Path('SYSU_2023SpringRL\Assignment2\saves')
    save_path.mkdir(exist_ok=True)

    save_path /= f'eps_len={episode_length}-upd_int={update_interval}-'\
        f'a_lr={actor_lr}-c_lr={critic_lr}-{lr_decay=}-{local_ratio=}-cont_act={continuous_actions}'
    
    model_save_path = save_path / ('models/pz' if use_pettingzoo else 'models/op')
    image_save_path = save_path / 'imgs'
    if model_save_path.exists():
        shutil.rmtree(model_save_path)
    if image_save_path.exists():
        shutil.rmtree(image_save_path)
    model_save_path.mkdir(exist_ok=True, parents=True)
    image_save_path.mkdir(exist_ok=True, parents=True)


    ######################################
    # training
    ######################################

    trainer = MADDPGTrainer(
        env, actor_lr, critic_lr, hidden_dim, n_layer=4,
        gamma=gamma, tau=tau, epsilon=epsilon, eps_decay=eps_decay,
        device=device,
        lr_scheduler=torch.optim.lr_scheduler.ExponentialLR, lr_scheduler_args={'gamma': lr_decay},
        env_args=env_args
    )

    assert trainer.action_dim == action_dim and trainer.state_dim == state_dim
    
    return_list = []
    replay_buffer = []  # stores global state, global next state and rewards from all agents
    total_step = 0
    best_saved_ret = - np.inf

    def evaluate(trainer, i_eps, n_episode=20):
        """
        Evaluate current policy in 100 episodes.
        """
        eval_return_list = []
        for i_episode in range(n_episode):
            episode_return = 0
            states = env.reset()
            generate_video = i_episode % (n_episode // 2) == 0
            frames = []
            for episode_step in range(episode_length):
                with torch.no_grad():
                    actions = [
                        trainer.take_action(_i, s, explore=False) for _i, s in enumerate(states)
                    ]

                next_states, rewards, done, infos = env.step(actions)
                states = next_states    #! 看我看我，我宣布个事儿：我是个傻逼！
                episode_return += np.sum(rewards) if use_pettingzoo else np.mean(rewards)
                frames.append(env.render())

            episode_return *= (1 + use_pettingzoo) / episode_length
            eval_return_list.append(episode_return)

            if generate_video:
                clip = ImageSequenceClip(frames, fps=10)
                clip.write_videofile(str(image_save_path / f'eval={i_eps}-eps={i_episode}.mp4'), logger=None)

            pass
        return np.mean(eval_return_list)


    for i_episode in tqdm(range(num_episodes)):
        states = env.reset()    # list of state of each agent
        episode_return = 0

        for episode_step in range(episode_length):
            actions = [
                trainer.take_action(_i, s, explore=True) for _i, s in enumerate(states)
            ]

            next_states, rewards, done, infos = env.step(actions)
            replay_buffer.append((
                np.concatenate(states, axis=0), actions, np.concatenate(next_states, axis=0),
                rewards, done
            ))
            if len(replay_buffer) > max_buffer_size:
                replay_buffer = replay_buffer[-max_buffer_size:]

            states = next_states
            episode_return += np.sum(rewards) if use_pettingzoo else np.mean(rewards)
            total_step += 1

            if len(replay_buffer) > start_train_size and total_step % update_interval == 0:
                batch_idx = np.random.choice(len(replay_buffer), size=batch_size, replace=False)
                batch_states, batch_actions, batch_next_states, batch_rewards, batch_dones = [], [], [], [], []

                for idx in batch_idx:
                    _s, _a, _ns, _r, _done = replay_buffer[idx]
                    batch_states.append(_s)
                    batch_actions.append(_a)
                    batch_next_states.append(_ns)
                    batch_rewards.append(_r)
                    batch_dones.append([_done])

                batch = [
                    torch.tensor(np.array(batch_states), device=device),
                    torch.tensor(np.array(batch_actions), device=device),
                    torch.tensor(np.array(batch_next_states), device=device),
                    torch.tensor(batch_rewards, device=device),
                    torch.tensor(batch_dones, dtype=torch.float32, device=device)
                ]
                # for a_i in range(env.num_agents):
                #     trainer.update(batch, a_i)

                # if total_step % (soft_update_per_update * update_interval) == 0:
                #     trainer.update_all_targets()

                for a_i in range(env.num_agents):
                    trainer.update_new(batch, a_i)
            pass

        episode_return *= (1 + use_pettingzoo) / episode_length    # to match from local_ratio=0.5 to both 1 for local and global reward
        return_list.append(episode_return)

        # save if needed
        if i_episode % eval_interval == 0:
            eval_return = evaluate(trainer, i_episode)
            print(f'Evaluated at {i_episode=}: return = {eval_return:.5f}')
            filename = model_save_path / f'avg={eval_return:.5f}-{i_episode=}-{datetime.now().strftime("%Y-%m-%d %Hh%Mm%Ss")}.pt'
            torch.save(
                {'i_episode': i_episode, 'trainer': trainer.state_dict()},
                filename
            )
            # print(f'Saved to {filename}')

            best_saved_ret = max(best_saved_ret, eval_return)

            # plot
            # plt.clf()
            # t = return_list[-500:]
            # r = np.arange(0, len(t) + 1, len(t)/10)
            # plt.plot(t)
            # plt.xticks(r, r + i_episode)
            # plt.savefig(image_save_path / 'trace.png')


    plt.clf()
    plt.plot(return_list)
    plt.savefig(image_save_path / 'overall.png')

    # plt.show()
    input('Press any key to close')

    env.close()


    pass




