import torch
# from torchrl.data import ReplayBuffer, ListStorage
import numpy as np
import matplotlib.pyplot as plt
from pettingzoo.mpe import simple_spread_v3
from time import sleep, time
from pathlib import Path

from algorithms.MADDPGTrainer import MADDPGTrainer



if __name__ == '__main__':
    save_path = Path('RL Playground\SYSU_2023SpringRL\Assignment2\saves')

    # params for learning
    num_episodes = 10000
    episode_length = 25 # default value for simple spread
    hidden_dim = 64
    actor_lr = critic_lr = 1e-2
    lr_decay = (1e-3 / actor_lr) ** (1 / 8000)

    gamma = 0.95    # reward decay rate
    tau = 1e-2 
    epsilon, eps_decay = 0.2, (1 / 4) ** (1 / 8000)
    batch_size, start_train_size, sample_start_from = 256, 1024, 2048
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    update_interval = 40   # interval to do soft update of target networks

    env = simple_spread_v3.env(max_cycles=episode_length)    # args: render_mode, max_cycles
    env.reset()
    state_dim, action_dim = env.state_space.shape[0], env.action_spaces['agent_0'].n

    trainer = MADDPGTrainer(
        env, actor_lr, critic_lr, hidden_dim,
        gamma=gamma, tau=tau, epsilon=epsilon, eps_decay=eps_decay,
        device=device,
        lr_scheduler=torch.optim.lr_scheduler.ExponentialLR, lr_scheduler_args={'gamma': lr_decay}
    )

    assert trainer.action_dim == action_dim and trainer.state_dim == state_dim
    
    return_list = []
    # replay_buffer = ReplayBuffer(storage=ListStorage(8192), batch_size=batch_size)
    replay_buffer = []  # stores global state, global next state and rewards from all agents


    # plt.show(block=False)
    # plt.ion()   # update every plot()
    total_step = 0
    recent_avg = - np.inf
    for i_episode in range(num_episodes):
        episode_step = 0
        episode_return = 0
        env.reset()
        states = env.state()    # global state of shape (state_dim,)
        actions = [None for _ in range(env.num_agents)]

        # start_time = time()

        for _i, agent in enumerate(env.agent_iter()):
            """
            This will iterate flattened n_agent*max_cycles steps, 
            `agent` takes action at (`agent_step` // n_agent) step 

            `agent` is a str like "agent_1"
            """
            episode_step, agent_id = _i // env.num_agents, _i % env.num_agents
            _o, _cr, _terminated, _truncated, _info = env.last()
            if _terminated or _truncated:
                break

            # take actions
            obs = env.observe(agent)    # take action according to observations
            with torch.no_grad():
                action = trainer.take_action(agent_id, obs, explore=True)

            actions[agent_id] = action
            env.step(action)
            # print(f'{agent} took action {action}')

            if agent_id == env.num_agents - 1:
                # all agents have taken actions and reach next episode step
                # next_states, cum_reward, termination, truncation, info = env.last()
                next_states = env.state()
                # rewards = {a: env.rewards[a] for a in env.agents}
                rewards = [env.rewards[a] for a in env.agents]  #! half of older env
                episode_return += sum(rewards)
                # print(f'step {episode_step} return: {sum(rewards)}')

                total_step += 1
                
                replay_buffer.append((
                    states, actions, next_states, rewards, episode_step >= episode_length))
                
                states = next_states

                if len(replay_buffer) > start_train_size and total_step % update_interval == 0:
                    # sample a batch from replay buffer
                    # recent_rb = replay_buffer[-sample_start_from:]
                    # batch_idx = np.random.choice(
                    #     len(recent_rb), size=batch_size, replace=False)
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
                        torch.tensor(batch_actions, device=device),
                        torch.tensor(np.array(batch_next_states), device=device),
                        torch.tensor(batch_rewards, device=device),
                        torch.tensor(batch_dones, dtype=torch.float32, device=device)
                    ]
                    

                    for a_i in range(env.num_agents):
                        trainer.update(batch, a_i, episode_step == episode_length-1)

                    trainer.update_all_targets()
                    pass


            pass

        episode_return *= 2 / episode_length    # to match from local_ratio=0.5 to both 1 for local and global reward
        return_list.append(episode_return)
        if i_episode % 10 == 0:
            recent_avg = np.mean(return_list[-100:])
            print(
                f'Episode {i_episode} return: {episode_return:.6f}, most recent avg {recent_avg:.6f}'
                # f', cost {time() - start_time:.4f} s'
            )

            # save if needed
            if recent_avg >= -8.0:
                torch.save(
                    {'i_episode': i_episode, 'trainer': trainer.state_dict()},
                    save_path / f'avg={recent_avg:.5f}-{i_episode=}.pt'
                )

        if i_episode % (num_episodes // 100) == 0:
            plt.clf()
            t = return_list[-500:]
            r = np.arange(0, len(t) + 1, 10)
            plt.plot(t)
            plt.xticks(r, r + i_episode)
            plt.savefig('trace.png')
            # plt.pause(0.05)
        
        
    plt.clf()
    plt.plot(return_list)
    plt.savefig('overall.png')

    # plt.show()
    input('Press any key to close')

    env.close()


    pass




