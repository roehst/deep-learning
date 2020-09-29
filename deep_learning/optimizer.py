from abc import ABC, abstractmethod


class Optimizer(ABC):
    def __init__(self, learning_rate: float = 0.01) -> None:
        self._learning_rate = learning_rate

    @abstractmethod
    def step(self) -> None:
        ...


class SGD(Optimizer):
    def step(self) -> None:
        for (param, param_grad) in zip(self.net.params(), self.net.param_grads()):
            param -= self._learning_rate * param_grad
