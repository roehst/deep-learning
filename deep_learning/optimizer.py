from abc import ABC, abstractmethod


class Optimizer(ABC):
    def __init__(self, learning_rate: float = 0.01) -> None:
        self._learning_rate = learning_rate

    @abstractmethod
    def step(self) -> None:
        ...
