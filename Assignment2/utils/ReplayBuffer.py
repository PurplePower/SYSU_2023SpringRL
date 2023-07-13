import numpy as np

from typing import List


class Experience:
    def __init__(self, state, act, rew, next_state, done) -> None:
        self.state = state
        self.act = act
        self.reward = rew
        self.next_state = next_state
        self.done = done




class ReplayBuffer:

    def __init__(self, max_size=5e6) -> None:
        self.max_size = max_size
        self.buffer:List[Experience] = []
        pass

    def store(self, state, act, rew, next_state, done):
        self.buffer.append(Experience(state, act, rew, next_state, done))
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)  # pop the earliest

    
    def sample(self, batch_size) -> Experience:
        idx = np.random.choice(len(self.buffer), batch_size, replace=False)
        s, a, r, s_n, d = [], [], [], [], []
        for i in idx:
            exp = self.buffer[i]
            s.append(exp.state)
            a.append(exp.act)
            r.append(exp.reward)
            s_n.append(exp.next_state)
            d.append(exp.done)

        return Experience(
            np.array(s, np.float32), np.array(a, np.float32), 
            np.array(r, np.float32), np.array(s_n, np.float32), np.array(d, np.float32)
        )
    
    def __len__(self):
        return len(self.buffer)

