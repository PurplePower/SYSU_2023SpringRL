import numpy as np
from abc import ABC, abstractmethod

class Policy:
    def __init__(self, n_action) -> None:
        self.n_action = n_action
        pass
    
    
    # @classmethod
    @abstractmethod
    def take_action(self, qs):
        pass
    
    pass


class GreedyPolicy(Policy):
    def __init__(self, n_action) -> None:
        super().__init__(n_action)
        
    
    def take_action(self, qs):
        return np.argmax(qs)
    

class EpsGreedyPolicy(GreedyPolicy):
    def __init__(self, n_action, epsilon=0.1) -> None:
        super().__init__(n_action)
        self.epsilon = epsilon
        
    def take_action(self, qs):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.n_action)
        else:
            return super().take_action()
