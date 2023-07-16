import numpy as np
from numba import jit, int64, float32, void
from typing import List, Tuple

class Experience:
    def __init__(self, state, act, rew, next_state, done) -> None:
        self.state = state
        self.act = act
        self.reward = rew
        self.next_state = next_state
        self.done = done


class ReplayBuffer:

    def __init__(
            self, state_shape, act_shape, rew_shape, done_shape, 
            max_size=int(1e6)
        ) -> None:
        self.max_size = max_size

        self.states = np.zeros((max_size, *state_shape), np.float32)
        self.actions = np.zeros((max_size, *act_shape), np.float32)
        self.rewards = np.zeros((max_size, *rew_shape), np.float32)
        self.next_states = np.zeros_like(self.states)
        self.dones = np.zeros((max_size, *done_shape), np.int8)

        self.index = 0
        self.size = 0
        pass

    def store(self, state, act, rew, next_state, done):
        self.states[self.index] = np.array(state)
        self.actions[self.index] = np.array(act)
        self.rewards[self.index] = rew
        self.next_states[self.index] = np.array(next_state)
        self.dones[self.index] = done

        self.index = (self.index + 1) % self.max_size   # pop earliest
        self.size = min(self.size + 1, self.max_size)

    
    def sample(self, batch_size) -> Experience:
        idx = np.random.choice(len(self.buffer), batch_size, replace=False)

        return Experience(
            self.states[idx], self.actions[idx], self.rewards[idx], self.next_states[idx],
            self.dones[idx]
        )   # copies are made when modified
    
    def __len__(self):
        return self.size
    

# =================================================================

@jit(void(float32[:], float32[:], int64, float32), nopython=True)
def _static_update_trees(sum_tree:np.ndarray, min_tree:np.ndarray, node_idx:int, prob:float):

    # add 1 in case tree array is (2*max_sz-1) long; otherwise 2*max_sz
    leaf_start_idx = (len(sum_tree) + 1) // 2 - 1   
    assert node_idx >= leaf_start_idx # node_idx doesn't start from leaves!
    sum_tree[node_idx] = prob
    min_tree[node_idx] = prob

    node_idx = (node_idx - 1) // 2  # find parent
    while node_idx >= 0:
        sum_tree[node_idx] = sum_tree[2 * node_idx + 1] + sum_tree[2 * node_idx + 2]
        min_tree[node_idx] = min(min_tree[2 * node_idx + 1, ], min_tree[2 * node_idx + 2])
        node_idx = (node_idx - 1) // 2

    pass


@jit(nopython=True)
def _static_get_leaf_sample_idx(sum_tree:np.ndarray, s, size):
    # using interval (a, b]
    leaf_start_idx = (len(sum_tree) + 1) // 2 - 1   
    idx = 0
    while idx < leaf_start_idx:
        lc, rc = idx * 2 + 1, idx * 2 + 2
        if s <= sum_tree[lc]:
            idx = lc
        else:
            idx = rc
            s -= sum_tree[lc]

    sample_idx = idx - leaf_start_idx
    if sample_idx >= size:
        sample_idx = np.random.choice(size) # this rarely happens
    return sample_idx


@jit(nopython=True)
def _static_sample(
    sum_tree:np.ndarray, min_tree:np.ndarray, 
    batch_size:int, size:int, beta:float, 
    # in/out
    batch_idx:np.ndarray, is_weights:np.ndarray
):
    leaf_start_idx = (len(sum_tree) + 1) // 2 - 1   
    tree_sum, tree_min = sum_tree[0], min_tree[0]

    # find random samples in `batch_size` intervals, or just uniform random in sum range
    seg_length = tree_sum / batch_size
    for i in range(batch_size):
        lb, ub = seg_length * i, seg_length * (i + 1)
        rand = np.random.uniform(lb, ub)
        batch_idx[i] = _static_get_leaf_sample_idx(sum_tree, rand, size)
        # assert batch_idx[i] < size

    min_prob = tree_min / tree_sum
    max_is_weight = (min_prob * size) ** (-beta)
    is_weights[:] = np.power(min_tree[batch_idx + leaf_start_idx] / tree_sum * size, -beta)
    is_weights /= max_is_weight
    pass


@jit(nopython=True)
def _static_per_process_priorities_and_update_batch(
    sum_tree:np.ndarray, min_tree:np.ndarray, batch_idx:np.ndarray, td_errors:np.ndarray,
    epsilon:float, prob_upper_bound:float, alpha:float
):
    td_errors = np.minimum(np.abs(td_errors) + epsilon, prob_upper_bound)
    max_prob = td_errors.max()
    td_errors = np.power(td_errors, alpha)

    leaf_start_idx = (len(sum_tree) + 1) // 2 - 1   
    # max_prob = td_errors.max()

    for i in range(len(batch_idx)):
        _static_update_trees(sum_tree, min_tree, batch_idx[i] + leaf_start_idx, td_errors[i])

    return max_prob


class PERBuffer(ReplayBuffer):

    def __init__(
            self, state_shape, act_shape, rew_shape, done_shape, 
            max_size=2**19
        ) -> None:
        super().__init__(state_shape, act_shape, rew_shape, done_shape, max_size)

        self.size = 0
        self.index = 0
        self.max_prob = 1.0 
        self.prob_upper_bound = 1.0

        self.beta = 0.4
        self.beta_inc = 0.0001
        self.alpha = 0.7
        self.epsilon = 0.01 # small constant adds on td error

        self.use_jit = True

        """
        index starts from 0.
        the tree has `max_size` leaves and `max_size`-1 internal nodes.

        Initially all samples has default prob, or zero if leaf is empty.
        """
        self.leaf_start_idx = self.max_size - 1
        self.sum_tree = np.zeros(max_size * 2, np.float32)
        self.min_tree = np.zeros_like(self.sum_tree) + np.inf    # min-heap to store minimum prob in leaves


    def _update_trees(self, index, prob):
        if self.use_jit:
            if isinstance(prob, np.ndarray):
                prob = prob[0]
            _static_update_trees(self.sum_tree, self.min_tree, index + self.leaf_start_idx, prob)
        else:
            node_idx = index + self.leaf_start_idx
            self.sum_tree[node_idx] = self.min_tree[node_idx] = prob
            node_idx = (node_idx - 1) // 2  # find parent
            while node_idx >= 0:
                self.sum_tree[node_idx] = self.sum_tree[2 * node_idx + 1] + self.sum_tree[2 * node_idx + 2]
                self.min_tree[node_idx] = min(self.min_tree[2 * node_idx + 1, ], self.min_tree[2 * node_idx + 2])
                node_idx = (node_idx - 1) // 2


    def _get_leaf_sample_idx(self, s) -> int:
        if self.use_jit:
            sample_idx = _static_get_leaf_sample_idx(self.sum_tree, s, self.size)
            if sample_idx >= self.size:
                self._check_trees()

        else:
            # using interval (a, b]
            idx = 0
            while idx < self.leaf_start_idx:
                lc, rc = idx * 2 + 1, idx * 2 + 2
                if s <= self.sum_tree[lc]:
                    idx = lc
                else:
                    idx = rc
                    s -= self.sum_tree[lc]

            sample_idx = idx - self.leaf_start_idx
        assert sample_idx < self.size
        return int(sample_idx)

    def _get_sum(self):
        return self.sum_tree[0]
    
    def _get_min(self):
        return self.min_tree[0]

    def store(self, state, act, rew, next_state, done):
        old_idx = self.index
        super().store(state, act, rew, next_state, done)

        self._update_trees(old_idx, self.max_prob ** self.alpha)
        # self._check_trees()


    def sample(self, batch_size) -> Tuple[Experience, np.ndarray, np.ndarray]:
        batch_idx = np.empty(batch_size, np.int64)
        if self.use_jit:
            is_weights = np.zeros(batch_size, np.float32)
            _static_sample(
                self.sum_tree, self.min_tree, batch_size, self.size, self.beta, batch_idx, is_weights)
        else:
            # find random samples in `batch_size` intervals, or just uniform random in sum range
            seg_length = self._get_sum() / batch_size
            for i in range(batch_size):
                lb, ub = seg_length * i, seg_length * (i + 1)
                rand = np.random.uniform(lb, ub)
                batch_idx[i] = self._get_leaf_sample_idx(rand)

            min_prob = self._get_min() / self._get_sum()
            max_is_weight = (min_prob * self.size) ** (-self.beta)
            is_weights = (self.min_tree[batch_idx + self.leaf_start_idx] / self._get_sum() * self.size) ** (-self.beta)
            is_weights /= max_is_weight

        expr = Experience(
            self.states[batch_idx], self.actions[batch_idx], self.rewards[batch_idx],
            self.next_states[batch_idx], self.dones[batch_idx]
        )

        if batch_idx.max() >= self.size:
            self._check_trees()

        # increase beta since samples's prob are getting closer to real prob
        self.beta = min(self.beta + self.beta_inc, 1.0)

        # self._check_trees()
        return expr, is_weights, batch_idx
    

    def update_batch(self, batch_idx:np.ndarray, td_errors:np.ndarray):
        td_errors = np.squeeze(td_errors)
        if self.use_jit:
            max_p = _static_per_process_priorities_and_update_batch(
                self.sum_tree, self.min_tree, batch_idx, td_errors, 
                self.epsilon, self.prob_upper_bound, self.alpha
            )
            self.max_prob = max(self.max_prob, max_p)
        else:
            td_errors = np.minimum(np.abs(td_errors) + self.epsilon, self.prob_upper_bound)
            self.max_prob = max(td_errors.max(), self.max_prob)
            td_errors = np.power(td_errors, self.alpha)
            for idx, td_error in zip(batch_idx, td_errors):
                self._update_trees(idx, td_error)

        # self._check_trees()
        pass


    def _check_trees(self) -> bool:
        # debug purpose

        idx = 0
        while idx < self.leaf_start_idx:
            lc, rc = 2 * idx + 1, 2 * idx + 2
            assert self.sum_tree[idx] == self.sum_tree[lc] + self.sum_tree[rc], f'check sum at {idx=}'
            assert self.min_tree[idx] == min(self.min_tree[lc], self.min_tree[rc]), f'check min at {idx=}'
            idx += 1



@jit(nopython=True)
def _static_relo_process_and_update_batch(
    sum_tree:np.ndarray, min_tree:np.ndarray, batch_idx:np.ndarray, relo:np.ndarray,
    epsilon:float, prob_upper_bound:float, alpha:float
):
    relo = np.maximum(relo, 0) + epsilon
    max_prob = relo.max()
    relo = np.power(relo, alpha)

    leaf_start_idx = (len(sum_tree) + 1) // 2 - 1 

    for i in range(len(batch_idx)):
        _static_update_trees(sum_tree, min_tree, batch_idx[i] + leaf_start_idx, relo[i])

    return max_prob



class ReloPERBuffer(PERBuffer):

    def __init__(self, state_shape, act_shape, rew_shape, done_shape, max_size=2 ** 19) -> None:
        super().__init__(state_shape, act_shape, rew_shape, done_shape, max_size)


    def update_batch(self, batch_idx: np.ndarray, relo: np.ndarray):
        # using Max(relo, 0) + eps
        relo = np.squeeze(relo)
        
        if self.use_jit:
            max_p = _static_relo_process_and_update_batch(
                self.sum_tree, self.min_tree, batch_idx, relo, 
                self.epsilon, self.prob_upper_bound, self.alpha
            )
            self.max_prob = max(self.max_prob, max_p)
        else:
            relo = np.maximum(relo, 0) + self.epsilon
            self.max_prob = max(relo.max(), self.max_prob)
            relo = np.power(relo, self.alpha)
            for idx, priority in zip(batch_idx, relo):
                self._update_trees(idx, priority)
        

        pass
        
