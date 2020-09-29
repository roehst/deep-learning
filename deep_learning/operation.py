from abc import ABC, abstractmethod

from numpy import ndarray

class Operation(ABC):
    def forward(self, input: ndarray) -> ndarray:
        self._input = input
        self._output = self.output()
        return self._output

    def backward(self, output_grad: ndarray) -> ndarray:
        self._input_grad = self.input_grad(output_grad)
        return self._input_grad

    @abstractmethod
    def output(self) -> ndarray:
        ...

    @abstractmethod
    def input_grad(self, output_grad: ndarray) -> ndarray:
        ...

