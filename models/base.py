from .types_ import *
from torch import nn
from abc import abstractmethod


class BaseAE(nn.Module):
    
    def __init__(self) -> None:
        super(BaseAE, self).__init__()

    def encode(self, input: Tensor):
        raise NotImplementedError

    def decode(self, input: Tensor):
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs: Tensor) -> Tensor:
        pass




