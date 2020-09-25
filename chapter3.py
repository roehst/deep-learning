from abc import ABC, abstractmethod
import numpy as np
from numpy import ndarray


class Operation(ABC):
    def forward(self, x: ndarray) -> ndarray:
        self._input = x
        self._output = self.output(x)
        return self._output

    def backward(self, x) -> ndarray:
        self._input_grad = self.input_grad(x)
        return self._input_grad

    @abstractmethod
    def output(self, x: ndarray) -> ndarray:
        ...

    @abstractmethod
    def input_grad(self, x: ndarray) -> ndarray:
        ...