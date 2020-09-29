from abc import ABC, abstractmethod
from numpy import ndarray

import numpy as np


class Loss(ABC):
    def forward(self, prediction: ndarray, target: ndarray) -> float:
        self._prediction = prediction
        self._target = target
        self._loss_value = self.output()
        return self._loss_value

    def backward(self) -> ndarray:
        self._input_grad = self.input_grad()
        return self._input_grad

    @abstractmethod
    def output(self) -> float:
        ...

    @abstractmethod
    def input_grad(self) -> ndarray:
        ...


class MeanSquaredError(Loss):
    def output(self) -> float:
        return (
            np.sum(np.power(self._prediction - self._target, 2.0))
            / self._prediction.shape[0]
        )

    def input_grad(self) -> ndarray:
        return 2.0 * (self._prediction - self._target) / self._prediction.shape[0]
