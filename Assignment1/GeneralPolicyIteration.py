from abc import ABC, abstractmethod

import numpy as np
from loguru import logger


class GeneralPolicyIteration(ABC):
    
    def __init__(self, gamma) -> None:
        self.gamma = gamma
        pass
    
    
    def _check_param(self):
        if self.gamma < 0.5:
            logger.warning(f'gamma value is lower than 0.5: {self.gamma=}')
    
    
    @abstractmethod
    def take_behavioural_action(self, s):
        pass
    
    
    @abstractmethod
    def take_target_action(self, s):
        pass
    
    
    
    
    
    pass


